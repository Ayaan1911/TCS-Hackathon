"""LLM contextual analysis + empathetic reply generation.

Calls Gemini (or, behind a flag, a local Ollama model) directly via
`requests` — no SDK. Any failure (network, timeout, malformed/missing JSON)
falls back to a canned template keyed off the local VADER label, so the
pipeline degrades gracefully instead of crashing.
"""

import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

import sentiment

load_dotenv(Path(__file__).parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

TIMEOUT_SECONDS = 8

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
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 200},
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
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        return _fallback(message, "no_api_key")

    prompt = _build_prompt(message, history)
    try:
        raw_text = _call_ollama(prompt) if LLM_PROVIDER == "ollama" else _call_gemini(prompt)
        parsed = json.loads(_strip_fences(raw_text))
        if not REQUIRED_KEYS.issubset(parsed.keys()):
            return _fallback(message, "missing_key")
        parsed["source"] = LLM_PROVIDER
        return parsed
    except requests.exceptions.Timeout:
        return _fallback(message, "timeout")
    except requests.exceptions.RequestException:
        return _fallback(message, "network_error")
    except (json.JSONDecodeError, KeyError, TypeError, IndexError, AttributeError):
        return _fallback(message, "malformed_json")
