"""Plain script self-test for sentiment -> llm_client -> agreement -> escalation, chained
exactly as the real pipeline will run. No pytest."""

import agreement
import escalation
import llm_client
import sentiment

HAS_KEY = bool(llm_client.GEMINI_API_KEY) and llm_client.GEMINI_API_KEY != "your_key_here"


def run_turn(text, history, raw_compounds, effective_compounds):
    local = sentiment.analyze(text)
    raw_compounds.append(local["compound"])

    llm_result = llm_client.analyze_and_reply(text, history)
    agree = agreement.resolve(local, llm_result["sentiment"])
    effective_compounds.append(agree["effective_compound"])

    score = escalation.compute_score(
        text, agree["effective_compound"], effective_compounds, llm_result["urgency"]
    )

    history.append({"role": "user", "text": text})
    history.append({"role": "bot", "text": llm_result["reply"]})

    print(f'message: "{text}"')
    print(f'  local: label={local["label"]} compound={local["compound"]:.3f} (raw, reference only)')
    print(
        f'  llm:   sentiment={llm_result["sentiment"]} emotion={llm_result["emotion"]} '
        f'intent={llm_result["intent"]} urgency={llm_result["urgency"]} source={llm_result["source"]}'
    )
    print(f'  reply: {llm_result["reply"]}')
    print(
        f'  agreement={agree["agreement"]} confidence={agree["confidence"]} '
        f'sarcasm_flag={agree["sarcasm_flag"]} effective_compound={agree["effective_compound"]}'
    )
    c = score["components"]
    print(
        f'  components: severity={c["severity"]:.1f} urgency={c["urgency"]:.1f} '
        f'trend={c["trend"]:.1f} sensitive={c["sensitive"]:.1f}'
    )
    print(f'  hard_override: {score["hard_override"]}')
    print(f'  -> score={score["score"]} band={score["band"]}')
    print()
    return local, llm_result, agree, score


def main():
    if not HAS_KEY:
        print("=" * 70)
        print("WARNING: no real GEMINI_API_KEY found in backend/.env.")
        print("Running against llm_client's FALLBACK path only. This exercises")
        print("the plumbing but CANNOT verify the message-3 disagreement fix:")
        print("the fallback re-derives its label from the same local VADER")
        print("score, so it will always agree with itself.")
        print("Copy backend/.env.example to backend/.env and add a real key")
        print("to get a live verification.")
        print("=" * 70)
        print()

    print("=== Conversation A (messages 1-3) ===\n")
    history, raw, eff = [], [], []
    run_turn("The shoes I ordered are actually really good. Love them!", history, raw, eff)
    run_turn("Actually, there's a problem. The size is wrong.", history, raw, eff)

    print(">>> CRITICAL CASE (message 3) <<<")
    _, _, agree3, score3 = run_turn(
        "I've already contacted support twice and nobody has helped me. I want my money back.",
        history, raw, eff,
    )
    print(
        f'>>> local_label={agree3["local_label"]} llm_label={agree3["llm_label"]} '
        f'agreement={agree3["agreement"]} effective_compound={agree3["effective_compound"]} '
        f'band={score3["band"]} (expected: escalate)'
    )
    if HAS_KEY:
        print("PASS" if score3["band"] == "escalate" else "FAIL", "- message 3 disagreement fix")
    else:
        print("SKIPPED (no live key) - cannot verify disagreement fix, see warning above")
    print()

    print("=== Conversation B (fresh, message 4 - fraud) ===\n")
    history_b, raw_b, eff_b = [], [], []
    _, _, _, score4 = run_turn(
        "Hi, I noticed someone used my card for a purchase I didn't authorize.",
        history_b, raw_b, eff_b,
    )
    print(
        "PASS" if score4["band"] == "escalate" and score4["hard_override"] else "FAIL",
        "- message 4 hard override (expected: escalate via hard_override)",
    )
    print()

    print("=== Conversation C (fresh, message 5 - sarcasm) ===\n")
    history_c, raw_c, eff_c = [], [], []
    _, _, agree5, _ = run_turn(
        "oh great, ANOTHER broken package. just perfect.", history_c, raw_c, eff_c
    )
    print(
        f'>>> local_label={agree5["local_label"]} llm_label={agree5["llm_label"]} '
        f'sarcasm_flag={agree5["sarcasm_flag"]}'
    )
    if HAS_KEY:
        expected_ok = agree5["local_label"] == "positive" and agree5["sarcasm_flag"]
        print(
            "PASS" if expected_ok else "FAIL",
            "- message 5 sarcasm detection (expected: local positive, llm negative, sarcasm_flag True)",
        )
    else:
        print("SKIPPED (no live key) - fallback always agrees with local VADER, sarcasm can't be tested")
    print()


if __name__ == "__main__":
    main()
