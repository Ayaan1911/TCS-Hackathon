"""Resolves disagreement between the local VADER scorer and the LLM's sentiment read.

VADER can't parse negation across clauses — e.g. "I've already contacted
support twice and nobody has helped me. I want my money back." scores
positive because of the word "helped", even though the sentence is clearly a
complaint. When the LLM disagrees with VADER, this module is what makes that
correction actually count toward escalation.compute_score's severity term,
instead of being noted and discarded.
"""

# First-pass tunable constant, not a derived number: when the LLM overrides
# the lexicon, disagreement itself is the signal (VADER's own magnitude is
# already known untrustworthy here), so treat it as at least
# moderately-strong evidence. Recalibrate once real Gemini output volume is
# available, same as the escalation weights.
DISAGREEMENT_SEVERITY_FLOOR = 0.65


def resolve(local_sentiment: dict, llm_label: str) -> dict:
    local_label = local_sentiment["label"]
    compound = local_sentiment["compound"]
    agreement = local_label == llm_label

    if agreement:
        confidence = round(0.75 + 0.25 * min(1.0, abs(compound)), 2)
        sarcasm_flag = False
        final_label = local_label
        effective_compound = compound
    else:
        confidence = 0.40
        sarcasm_flag = True
        final_label = llm_label
        magnitude = max(abs(compound), DISAGREEMENT_SEVERITY_FLOOR)
        if llm_label == "negative":
            effective_compound = -magnitude
        elif llm_label == "positive":
            effective_compound = magnitude
        else:
            effective_compound = 0.0

    return {
        "agreement": agreement,
        "confidence": confidence,
        "sarcasm_flag": sarcasm_flag,
        "final_label": final_label,
        "effective_compound": round(effective_compound, 3),
        "local_label": local_label,
        "llm_label": llm_label,
    }
