"""Transparent escalation decision engine.

Every number that feeds the final score is returned individually so the
decision is inspectable, not a black box.
"""

SENSITIVE_PHRASES = [
    "refund", "money back", "lawsuit", "sue", "legal action", "chargeback",
    "scam", "fraud", "cancel my", "cancel the", "unacceptable", "worst",
    "never again", "terrible",
    "called twice", "called three", "called again", "contacted support",
    "contacted you", "multiple times", "again and again", "still not fixed",
    "still broken", "nobody replied", "no one replied", "no response",
    "haven't heard back",
]

CRITICAL_PHRASES = [
    "unauthorized", "didn't authorize", "did not authorize", "someone used my card",
    "stolen card", "card was stolen", "account hacked", "hacked", "identity theft",
    "security breach", "lawsuit", "legal action", "sue you", "lawyer", "solicitor",
    "injured", "unsafe", "caught fire", "burned me", "allergic reaction", "hospital",
]

URGENCY_MAP = {"high": 100, "medium": 50, "low": 0}


def compute_trajectory(scores: list) -> dict:
    recent = scores[-3:]
    if len(recent) < 2:
        return {"trend": "stable", "trend_score": 0.0}

    current = recent[-1]
    declining = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))
    improving = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))

    if declining and current < 0:
        magnitude = recent[0] - recent[-1]
        return {"trend": "deteriorating", "trend_score": min(100.0, magnitude * 100)}
    if improving:
        return {"trend": "improving", "trend_score": 0.0}
    return {"trend": "stable", "trend_score": 0.0}


def compute_score(text: str, compound: float, trajectory_scores: list, llm_urgency: str = None) -> dict:
    lower = text.lower()

    severity = max(0.0, -compound) * 100
    # Fast path (llm_urgency=None) contributes 0 here, which makes the score a
    # LOWER BOUND: it can only go up once the LLM responds with a real urgency
    # signal, never down. This keeps the UI escalation badge from walking
    # backwards from red to green mid-conversation.
    urgency = URGENCY_MAP.get(llm_urgency, 0) if llm_urgency else 0
    traj = compute_trajectory(trajectory_scores)
    trend = traj["trend_score"]
    sensitive = 100 if any(p in lower for p in SENSITIVE_PHRASES) else 0

    score = 0.40 * severity + 0.25 * urgency + 0.20 * trend + 0.15 * sensitive

    if score >= 70:
        band = "escalate"
    elif score >= 40:
        band = "priority"
    else:
        band = "normal"

    reasons = []
    if severity >= 50:
        reasons.append("strongly negative sentiment")
    if urgency >= 100:
        reasons.append("high-urgency complaint")
    if trend >= 50:
        reasons.append("sentiment deteriorating across recent turns")
    if sensitive:
        reasons.append("sensitive phrase detected")

    hard_override = False
    # A purely sentiment-weighted score cannot catch a calm-but-critical
    # message (e.g. "someone used my card without authorization" said in a
    # flat tone). This override handles categorical risk as a hard floor
    # applied AFTER the weighted average, while the weighted score above
    # handles emotional escalation. It is deliberately not folded in as a
    # fifth weighted component, since a weighted average could still be
    # dragged down to a non-escalate band by low severity/urgency/trend.
    if any(p in lower for p in CRITICAL_PHRASES):
        score = max(score, 80.0)
        band = "escalate"
        hard_override = True
        reasons.insert(0, "CRITICAL: safety, fraud or legal issue detected")

    return {
        "score": round(score, 1),
        "band": band,
        "trend": traj["trend"],
        "hard_override": hard_override,
        "components": {
            "severity": severity,
            "urgency": urgency,
            "trend": trend,
            "sensitive": sensitive,
        },
        "reasons": reasons,
    }
