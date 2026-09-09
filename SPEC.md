# RetailSense — Design Spec

## Problem Statement

RetailSense is a retail customer sentiment analysis chatbot. It detects
sentiment in a customer's messages in real time, replies empathetically, and
recommends escalation to a human agent when a conversation warrants it. This
document covers the sentiment/escalation core built in step 1 of the build.

## Three-Layer Architecture

1. **Local deterministic scorer** (`backend/sentiment.py`) — a lexicon-based
   VADER sentiment analyzer. Runs in-process, no network call, ~1ms latency.
   Gives every message a compound/pos/neu/neg score and a positive/neutral/
   negative label, instantly and reproducibly.
2. **LLM contextual analysis + generation** (later step) — an LLM reads the
   full conversation for context VADER can't see (sarcasm, multi-turn intent,
   urgency), generates the empathetic reply, and supplies an urgency signal
   (`high`/`medium`/`low`) that feeds the escalation engine.
3. **Escalation engine** (`backend/escalation.py`) — our own transparent
   decision logic. Combines the local score, the LLM's urgency signal,
   conversation trajectory, and sensitive-phrase detection into a single,
   inspectable escalation score and band.

## Escalation Formula

```
severity  = max(0, -compound) * 100          # weight 0.40
urgency   = URGENCY_MAP[llm_urgency] or 0     # weight 0.25
trend     = compute_trajectory(...).trend_score  # weight 0.20
sensitive = 100 if a sensitive phrase matched else 0  # weight 0.15

score = 0.40*severity + 0.25*urgency + 0.20*trend + 0.15*sensitive
```

Bands: `0-39 normal`, `40-69 priority`, `70-100 escalate`.

Every component is returned individually in the result (`components` dict)
so the score is fully inspectable — no hidden combination logic.

## Hard Override (separate from the weighted score)

A fixed list of `CRITICAL_PHRASES` (fraud, safety, legal, medical) forces
`score = max(score, 80.0)` and `band = "escalate"` *after* the weighted score
is computed — it is a floor, not a fifth weighted input.

This exists as a separate mechanism because a purely sentiment-weighted score
cannot catch a calm-but-critical message: "someone used my card without
authorization" can be phrased calmly, with near-zero severity, no LLM urgency
yet available, and no deteriorating trend. Weighted scoring measures
*emotional* escalation; the override catches *categorical risk* regardless of
tone. Both are needed — one alone misses either calm fraud reports or noisy-
but-low-risk venting.

## Trajectory Logic

`compute_trajectory` looks at the last 3 compound scores in the session
(oldest first, current message included). If they are strictly decreasing and
the current score is negative, the trend is `deteriorating` and its
trend_score scales with the drop magnitude (capped at 100). If strictly
increasing, `improving` (trend_score 0). Otherwise `stable` (trend_score 0).
Fewer than 2 scores available is always `stable`.

## Lower-Bound Property

When `llm_urgency` is `None` (fast path, LLM not yet responded), urgency
contributes 0 to the score. This makes the fast-path score a **lower bound**:
once the LLM responds with an urgency signal, the score can only rise, never
fall. This guarantees the UI escalation badge never walks backwards from red
to green mid-conversation — it can only get more urgent as more signal
arrives.

## Confidence (future work)

Confidence is not self-reported by the LLM (LLMs are poor judges of their own
calibration). It will instead be derived from **agreement** between the local
VADER classifier and the LLM's contextual read: when both agree on sentiment
direction, confidence is high; when they diverge, confidence is low and worth
surfacing to a human reviewer.
