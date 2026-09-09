"""LLM contextual analysis + empathetic reply generation.

Calls Gemini (or, behind a flag, a local Ollama model) directly via
`requests` — no SDK. Any failure (network, timeout, malformed/missing JSON)
falls back to a canned template keyed off the local VADER label, so the
pipeline degrades gracefully instead of crashing.
"""

import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

import sentiment

load_dotenv(Path(__file__).parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# deepseek/deepseek-r1:free was removed from OpenRouter's catalog (checked
# live against /api/v1/models: 0 free deepseek models currently listed).
# Free-model availability rotates, so re-check the catalog if this one goes
# stale too.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nex-agi/nex-n2.5-mini:free")

TIMEOUT_SECONDS = 8
RETRY_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 1

FALLBACK_TEMPLATES = {
    "positive": "I'm so glad to hear that! Thanks so much for sharing your experience.",
    "neutral": "Thanks for reaching out - I'm looking into this for you right away.",
    "negative": "I'm really sorry for the trouble this has caused you. Let's get this sorted out.",
}

REQUIRED_KEYS = {"sentiment", "emotion", "intent", "urgency", "reply"}

PROMPT_TEMPLATE = """You are a retail customer support assistant analyzing a customer's message.

Conversation so far (most recent last):
{history_block}

Customer's latest message: "{message}"

Analyze the sentiment, emotion, and intent of this message, judge how urgently
it needs human attention, and write an empathetic reply.

Return ONLY raw JSON, no markdown code fences, no commentary, in exactly this shape:
{{
  "sentiment": "positive" | "neutral" | "negative",
  "emotion": "satisfied" | "frustrated" | "angry" | "worried" | "confused" | "neutral",
  "intent": "praise" | "product_issue" | "delivery_delay" | "refund_request" | "sizing_issue" | "billing_issue" | "fraud_or_security" | "general_query" | "other",
  "urgency": "low" | "medium" | "high",
  "reply": "1-3 sentence empathetic response, no internal jargon, no mention of scores or sentiment"
}}"""


def _build_prompt(message: str, history: list) -> str:
    recent = history[-4:]
    if recent:
        history_block = "\n".join(f'{turn["role"]}: {turn["text"]}' for turn in recent)
    else:
        history_block = "(no prior messages)"
    return PROMPT_TEMPLATE.format(history_block=history_block, message=message)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _call_gemini(prompt: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        # 500, not 200: gemini-flash-latest spends part of this budget on
        # internal reasoning tokens before writing the visible reply, so 200
        # was truncating the JSON mid-object.
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 500},
    }
    resp = requests.post(url, json=body, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_ollama(prompt: str) -> str:
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    resp = requests.post(
        "http://localhost:11434/api/generate", json=body, timeout=TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    data = resp.json()
    return data["response"]


def _call_openai(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.4,
        "max_tokens": 200,
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions", json=body, headers=headers, timeout=TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/Ayaan1911/TCS-Hackathon",
    }
    body = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 200,
    }
    # No response_format here: free OpenRouter models don't reliably honor
    # json_object mode, so this relies on the same fence-stripping + parsing
    # logic used for Gemini's occasional markdown-wrapped output.
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers, timeout=TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _parse_llm_json(raw_text: str):
    try:
        parsed = json.loads(_strip_fences(raw_text))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict) or not REQUIRED_KEYS.issubset(parsed.keys()):
        return None
    return parsed


def _fallback(message: str, reason: str) -> dict:
    local = sentiment.analyze(message)
    return {
        "sentiment": local["label"],
        "emotion": "neutral",
        "intent": "other",
        "urgency": "medium",
        "reply": FALLBACK_TEMPLATES[local["label"]],
        "source": f"fallback:{reason}",
    }


def analyze_and_reply(message: str, history: list) -> dict:
    prompt = _build_prompt(message, history)
    reason = "no_api_key"

    # Tier 1: primary provider (Gemini, or Ollama behind the flag), with a
    # short retry-with-backoff. We saw intermittent 503s from Gemini during
    # testing that cleared within a couple seconds, which a single attempt
    # can't ride out.
    if LLM_PROVIDER != "gemini" or GEMINI_API_KEY:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                raw_text = _call_ollama(prompt) if LLM_PROVIDER == "ollama" else _call_gemini(prompt)
                parsed = _parse_llm_json(raw_text)
                if parsed is not None:
                    parsed["source"] = LLM_PROVIDER
                    return parsed
                reason = "malformed_json"
                break  # malformed content won't fix itself on retry
            except requests.exceptions.Timeout:
                reason = "timeout"
            except requests.exceptions.RequestException:
                reason = "network_error"
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_SECONDS)

    # Tier 2: OpenAI as a hedge on independent infrastructure. Retries alone
    # only help if an outage is shorter than the retry window; a second cloud
    # provider covers the case where Gemini itself is down for longer than
    # that. Only one attempt here — the retry budget was already spent on
    # tier 1, and demo latency matters.
    if OPENAI_API_KEY:
        try:
            raw_text = _call_openai(prompt)
            parsed = _parse_llm_json(raw_text)
            if parsed is not None:
                parsed["source"] = "openai_fallback"
                return parsed
            reason = "openai_malformed_json"
        except requests.exceptions.Timeout:
            reason = "openai_timeout"
        except requests.exceptions.RequestException:
            reason = "openai_network_error"

    # Tier 3: OpenRouter (free-tier model) as a second hedge, on yet another
    # independent provider. Only one attempt, same reasoning as tier 2.
    if OPENROUTER_API_KEY:
        try:
            raw_text = _call_openrouter(prompt)
            parsed = _parse_llm_json(raw_text)
            if parsed is not None:
                parsed["source"] = "openrouter_fallback"
                return parsed
            reason = "openrouter_malformed_json"
        except requests.exceptions.Timeout:
            reason = "openrouter_timeout"
        except requests.exceptions.RequestException:
            reason = "openrouter_network_error"

    # Tier 4: local template, keyed off VADER's own read of the message.
    return _fallback(message, reason)
