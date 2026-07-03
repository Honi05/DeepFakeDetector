# src/forensic_text.py

ARTIFACT_DESCRIPTIONS = {
    "forehead": "texture discontinuities near the hairline",
    "eyes":     "unnatural eye reflection and blinking inconsistencies",
    "nose":     "geometric distortion in the nasal bridge region",
    "jaw":      "blending seam artifacts along the jaw boundary",
    "hairline": "synthetic hair strand generation artifacts",
}


def generate_forensic_report(confidence: float, zone1: str, zone2: str) -> str:
    """
    Deterministic forensic report from confidence score and top-2 activated facial zones.
    confidence: float in [0, 1] — probability of being fake.
    zone1, zone2: strings from ARTIFACT_DESCRIPTIONS keys.
    """
    hint1 = ARTIFACT_DESCRIPTIONS.get(zone1, "anomalous activation patterns")
    hint2 = ARTIFACT_DESCRIPTIONS.get(zone2, "anomalous activation patterns")
    pct = f"{confidence:.1%}"

    if confidence >= 0.85:
        return (
            f"HIGH CONFIDENCE FAKE ({pct}) — "
            f"Strong manipulation artifacts detected. "
            f"Primary anomalies localized to the {zone1} ({hint1}) "
            f"and {zone2} ({hint2}). "
            f"Pattern consistent with face-swap or neural texture synthesis."
        )
    elif confidence >= 0.65:
        return (
            f"LIKELY MANIPULATED ({pct}) — "
            f"Suspicious inconsistencies identified in the {zone1} region. "
            f"{hint1.capitalize()}. "
            f"Secondary anomalies present at the {zone2}. "
            f"Secondary human review recommended."
        )
    elif confidence >= 0.50:
        return (
            f"BORDERLINE CASE ({pct}) — "
            f"Low-confidence detection. Possible manipulation at the {zone1}. "
            f"Could reflect high-quality synthesis or severe compression loss. "
            f"Manual verification required."
        )
    else:
        return (
            f"LIKELY AUTHENTIC ({pct}) — "
            f"No significant manipulation artifacts detected. "
            f"Minor activation at {zone1} consistent with natural image noise or compression."
        )
