"""Local deterministic sentiment scoring via VADER.

We run a lexicon-based scorer alongside the LLM (rather than relying on the
LLM alone) for four reasons: it gives a reproducible accuracy metric to
benchmark against, it responds in ~1ms so the UI can show instant feedback
as the customer types, it costs zero API quota on what will be the
highest-frequency operation in the system, and it keeps working even if the
network or LLM provider is down.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def analyze(text: str) -> dict:
    if not text or not text.strip():
        return {"label": "neutral", "compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}

    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "compound": compound,
        "pos": scores["pos"],
        "neu": scores["neu"],
        "neg": scores["neg"],
    }
