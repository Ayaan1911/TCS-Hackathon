"""Plain script self-test for the sentiment + escalation pipeline. No pytest."""

import sentiment
import escalation


def run_message(text, llm_urgency, expected_band, trajectory_scores):
    result = sentiment.analyze(text)
    compound = result["compound"]
    trajectory_scores.append(compound)

    score = escalation.compute_score(text, compound, trajectory_scores, llm_urgency)

    passed = score["band"] == expected_band
    status = "PASS" if passed else "FAIL"

    print(f'  message: "{text}"')
    print(f"  compound: {compound:.3f}")
    c = score["components"]
    print(f"  severity={c['severity']:.1f} urgency={c['urgency']:.1f} trend={c['trend']:.1f} sensitive={c['sensitive']:.1f}")
    print(f"  hard_override: {score['hard_override']}")
    print(f"  total: {score['score']} -> band: {score['band']} (expected: {expected_band}) [{status}]")
    print()
    return passed


def main():
    results = []

    print("=== Conversation A (messages 1-3, trajectory accumulates) ===\n")
    trajectory = []
    results.append(run_message(
        "The shoes I ordered are actually really good. Love them!",
        None, "normal", trajectory,
    ))
    results.append(run_message(
        "Actually, there's a problem. The size is wrong.",
        "medium", "priority", trajectory,
    ))
    results.append(run_message(
        "I've already contacted support twice and nobody has helped me. I want my money back.",
        "high", "escalate", trajectory,
    ))

    print("=== Conversation B (fresh conversation, message 4) ===\n")
    trajectory_b = []
    results.append(run_message(
        "Hi, I noticed someone used my card for a purchase I didn't authorize.",
        None, "escalate", trajectory_b,
    ))

    passed_count = sum(results)
    total = len(results)
    print(f"{passed_count}/{total} PASSED" if passed_count == total else f"{passed_count}/{total} FAILED")


if __name__ == "__main__":
    main()
