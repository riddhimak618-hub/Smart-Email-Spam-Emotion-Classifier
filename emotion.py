from textblob import TextBlob

# ── Keyword Lists ─────────────────────────────────────────────
URGENT_KEYWORDS = [
    "urgent", "immediately", "asap", "deadline", "action required",
    "critical", "emergency", "right away", "time sensitive", "cannot wait",
    "right now", "final notice", "hurry", "please act", "respond now",
    "act now", "last chance", "expires", "overdue", "time is running out"
]

ANGRY_KEYWORDS = [
    "unacceptable", "frustrated", "angry", "ridiculous", "outrageous",
    "terrible", "worst", "incompetent", "disgusting", "furious", "hate",
    "annoyed", "complaint", "demand", "pathetic", "disgraceful", "rude",
    "useless", "infuriated", "livid", "fed up", "appalled", "inexcusable",
    "unprofessional", "outraged", "this is a joke"
]

HAPPY_KEYWORDS = [
    "great", "thank", "thanks", "excellent", "wonderful", "congrats",
    "congratulations", "appreciate", "happy", "pleased", "delighted",
    "love", "amazing", "fantastic", "awesome", "good news", "thrilled",
    "excited", "proud", "brilliant", "outstanding", "well done", "perfect",
    "glad", "celebrate", "cheers", "impressive", "superb", "grateful"
]

SAD_KEYWORDS = [
    "sorry", "unfortunately", "regret", "sad", "disappointed", "miss",
    "loss", "difficult", "struggle", "bad news", "unable", "failed",
    "apologize", "apology", "condolences", "grief", "deeply sorry",
    "regrettably", "heartbroken", "distressed", "helpless", "unfortunate",
    "painful", "deeply regret", "cannot proceed", "rejected"
]


def _score(text: str, keywords: list) -> int:
    """Count how many keywords appear in text (case-insensitive)."""
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def detect_emotion(subject: str, body: str) -> dict:
    """
    Detect emotion from email subject + body.

    Returns:
        dict with keys: emotion, polarity, subjectivity, confidence, description
    """
    full_text    = f"{subject} {body}".strip()
    blob         = TextBlob(full_text)
    polarity     = round(blob.sentiment.polarity, 4)      # -1.0 to +1.0
    subjectivity = round(blob.sentiment.subjectivity, 4)  #  0.0 to +1.0

    # Keyword scores
    us = _score(full_text, URGENT_KEYWORDS)
    ans = _score(full_text, ANGRY_KEYWORDS)
    hs  = _score(full_text, HAPPY_KEYWORDS)
    ss  = _score(full_text, SAD_KEYWORDS)

    # ── Emotion Rules (priority order) ───────────────────────
    # Urgent: strong keyword signal OR at least 1 keyword + subjective negative tone
    if us >= 2 or (us >= 1 and (subjectivity > 0.3 or polarity < -0.1)):
        emotion     = "Urgent"
        description = "Email contains urgent language demanding immediate attention."

    # Angry: strong keyword signal OR negative polarity + high subjectivity + 1 keyword
    elif ans >= 2 or (polarity < -0.3 and subjectivity > 0.5 and ans >= 1):
        emotion     = "Angry"
        description = "Email expresses strong frustration or anger."

    # Happy: strong keyword signal OR clearly positive polarity
    elif hs >= 2 or polarity > 0.35:
        emotion     = "Happy"
        description = "Email carries a positive and cheerful tone."

    # Sad: strong keyword signal OR mildly negative + objective tone
    elif ss >= 2 or (polarity < -0.15 and subjectivity < 0.5):
        emotion     = "Sad"
        description = "Email conveys sadness, regret, or disappointment."

    # Neutral: calm, factual, professional
    else:
        emotion     = "Neutral"
        description = "Email has a calm, factual, and professional tone."

    # ── Confidence (based on combined sentiment signal strength) ──
    signal     = abs(polarity) + subjectivity
    confidence = "High" if signal > 1.0 else "Medium" if signal > 0.45 else "Low"

    return {
        "emotion":      emotion,
        "polarity":     polarity,
        "subjectivity": subjectivity,
        "confidence":   confidence,
        "description":  description
    }


# ── Quick Test ────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("URGENT Server Down Act Now",
         "The production server is completely down. We need someone to respond immediately. This is critical."),
        ("Congratulations on Your Promotion!",
         "I am absolutely thrilled to share the wonderful news. Your hard work has truly paid off. Well done!"),
        ("This service is completely unacceptable",
         "I am furious and extremely frustrated. This is outrageous. I demand a full refund immediately."),
        ("Meeting Notes Monday Review",
         "Hi team please find attached the notes from Mondays review. Next meeting is Thursday at 10 AM."),
        ("We regret to inform you",
         "Unfortunately we cannot offer you the position. We apologize for the disappointing news."),
        ("Act now before it expires",
         "Last chance — your offer expires tonight. Please respond right away."),
    ]

    print("=" * 60)
    print("   emotion.py — Test Run")
    print("=" * 60)
    for subj, body in tests:
        r = detect_emotion(subj, body)
        print(f"\n  Subject  : {subj}")
        print(f"  Emotion  : {r['emotion']}  [{r['confidence']} confidence]")
        print(f"  Polarity : {r['polarity']:+.3f}   Subjectivity: {r['subjectivity']:.3f}")
        print(f"  Insight  : {r['description']}")
    print("\n" + "=" * 60)
