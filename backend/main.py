"""FastAPI layer wiring the local scorer, LLM, agreement, and escalation engine together."""

import socket

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agreement
import escalation
import llm_client
import sentiment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: dict = {}

NEXT_BEST_ACTION = {
    "refund_request": ["Verify refund status", "Review previous contact history", "Prioritize if repeat request"],
    "delivery_delay": ["Check live shipment status", "Verify courier exception", "Escalate if SLA breached"],
    "fraud_or_security": ["Escalate immediately to security/payment team", "Do not expose account details in chat", "Confirm identity through a secure channel"],
    "sizing_issue": ["Offer exchange or return path", "Log product defect if recurring"],
    "product_issue": ["Offer exchange or return path", "Log product defect if recurring"],
    "billing_issue": ["Pull recent billing history", "Verify charge", "Offer correction or credit"],
    "praise": ["No action needed - standard reply is sufficient"],
    "general_query": ["No action needed - standard reply is sufficient"],
    "other": ["Review manually - intent unclear"],
}


class QuickRequest(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


@app.post("/analyze/quick")
def analyze_quick(req: QuickRequest):
    local = sentiment.analyze(req.message)
    history = SESSIONS.get(req.session_id, [])
    past_effective = [h["effective_compound"] for h in history if h["role"] == "user"]
    provisional_trajectory = past_effective + [local["compound"]]
    esc = escalation.compute_score(req.message, local["compound"], provisional_trajectory, llm_urgency=None)

    return {
        "sentiment_label": local["label"],
        "compound": local["compound"],
        "intensity": round(abs(local["compound"]) * 100, 1),
        "trajectory": provisional_trajectory,
        "trend": esc["trend"],
        "provisional_score": esc["score"],
        "provisional_band": esc["band"],
    }


@app.post("/chat")
def chat(req: ChatRequest):
    local = sentiment.analyze(req.message)
    history = SESSIONS.setdefault(req.session_id, [])
    gemini_history = [{"role": h["role"], "text": h["text"]} for h in history][-4:]

    llm_result = llm_client.analyze_and_reply(req.message, gemini_history)
    res = agreement.resolve(local, llm_result["sentiment"])

    past_effective = [h["effective_compound"] for h in history if h["role"] == "user"]
    full_trajectory = past_effective + [res["effective_compound"]]
    esc = escalation.compute_score(req.message, res["effective_compound"], full_trajectory, llm_urgency=llm_result["urgency"])
    nba = NEXT_BEST_ACTION.get(llm_result["intent"], NEXT_BEST_ACTION["other"])

    history.append({"role": "user", "text": req.message, "effective_compound": res["effective_compound"]})
    history.append({"role": "bot", "text": llm_result["reply"], "effective_compound": 0.0})

    return {
        "sentiment": {
            "label": res["final_label"],
            "compound": res["effective_compound"],
            "local_label": res["local_label"],
            "llm_label": res["llm_label"],
            "agreement": res["agreement"],
            "confidence": res["confidence"],
            "sarcasm_flag": res["sarcasm_flag"],
        },
        "emotion": llm_result["emotion"],
        "intent": llm_result["intent"],
        "urgency": llm_result["urgency"],
        "trajectory": {"scores": full_trajectory, "trend": esc["trend"]},
        "escalation": {
            "score": esc["score"],
            "band": esc["band"],
            "components": esc["components"],
            "reasons": esc["reasons"],
            "hard_override": esc["hard_override"],
        },
        "next_best_action": nba,
        "reply": llm_result["reply"],
        "reply_source": llm_result["source"],
    }


@app.post("/reset")
def reset(req: ResetRequest):
    SESSIONS.pop(req.session_id, None)
    return {"status": "reset"}


@app.get("/health")
def health():
    return {"status": "ok", "gemini_key_configured": bool(llm_client.GEMINI_API_KEY)}


def _find_available_port(candidates):
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"none of the candidate ports are available: {candidates}")


if __name__ == "__main__":
    # 8000 first, falling back to 8001/8002 if something else already holds
    # it. The frontend's API_BASE is not updated automatically, so the port
    # actually bound is printed loudly here rather than failing silently.
    port = _find_available_port([8000, 8001, 8002])
    # flush=True: stdout is fully buffered (not line-buffered) whenever it's
    # redirected rather than a live terminal, so without this the message
    # can sit invisible in the buffer for the life of the process.
    print(f"RetailSense backend running on http://127.0.0.1:{port}", flush=True)
    if port != 8000:
        print(f"*** NOT the default port 8000 — update frontend/index.html's API_BASE to :{port} if needed ***", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
