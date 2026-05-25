"""
text_processing.py
------------------
Text cleaning, sentence splitting, and extractive summarisation
utilities for the AI Paper Reading Assistant.
"""

import re
import string
from typing import List


# ──────────────────────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────────────────────

def clean_text(raw_text: str) -> str:
    """
    Clean raw PDF-extracted text for downstream NLP processing.

    Steps
    -----
    1. Collapse multiple newlines / carriage returns to single space.
    2. Remove non-ASCII / special symbols (keep alphanumerics + basic punctuation).
    3. Collapse multiple whitespace characters.
    4. Lower-case the result.
    5. Drop very short tokens (1–2 char) to reduce noise.

    Parameters
    ----------
    raw_text : str

    Returns
    -------
    str – cleaned text (may be empty string if input is empty)
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # Replace various line-break / form-feed combos with a space
    text = re.sub(r"[\r\n\f\v]+", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove standalone numbers (page numbers, figure refs, etc.)
    text = re.sub(r"\b\d+\b", " ", text)

    # Keep only letters, digits, spaces, and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\;\:\!\?\-\']", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Lower-case
    text = text.lower()

    # Drop very short "words" (likely OCR noise)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    text = " ".join(tokens)

    return text


def safe_clean(raw_text: str, min_length: int = 50) -> str:
    """
    Clean text and return empty string if result is too short to be useful.

    Parameters
    ----------
    raw_text   : str
    min_length : int – minimum acceptable cleaned-text length

    Returns
    -------
    str
    """
    cleaned = clean_text(raw_text)
    return cleaned if len(cleaned) >= min_length else ""


# ──────────────────────────────────────────────────────────────
# Sentence Splitting
# ──────────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    """
    Split text into individual sentences using a simple regex heuristic.

    Handles common abbreviations and numeric decimals to avoid false splits.

    Parameters
    ----------
    text : str

    Returns
    -------
    List[str] – list of sentence strings, stripped of extra whitespace
    """
    if not text:
        return []

    # Protect common abbreviations
    abbreviations = [
        "e.g", "i.e", "etc", "vs", "fig", "eq", "no", "vol",
        "dr", "mr", "mrs", "prof", "al", "approx", "dept",
    ]
    placeholder = "__ABBR__"
    protected = text
    for abbr in abbreviations:
        protected = re.sub(
            rf"\b{re.escape(abbr)}\.",
            abbr.replace(".", "_") + placeholder,
            protected,
            flags=re.IGNORECASE,
        )

    # Split on sentence-ending punctuation followed by whitespace + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)

    # Restore abbreviation dots
    sentences = [
        s.replace(placeholder, ".").strip()
        for s in parts
        if s.strip()
    ]
    return sentences


# ──────────────────────────────────────────────────────────────
# Extractive Summary
# ──────────────────────────────────────────────────────────────

def extractive_summary(abstract: str, cleaned_text: str,
                        num_sentences: int = 4,
                        max_chars: int = 600) -> str:
    """
    Generate an extractive summary without using any external LLM.

    Strategy
    --------
    1. If a non-trivial abstract is available, use it directly (truncated).
    2. Otherwise score sentences by simple heuristics (length, position,
       presence of informative words) and pick the top `num_sentences`.

    Parameters
    ----------
    abstract      : str – pre-extracted abstract (may be empty)
    cleaned_text  : str – full cleaned paper text
    num_sentences : int – number of sentences for fallback extractive summary
    max_chars     : int – maximum characters in the output

    Returns
    -------
    str – summary string
    """
    # ── Use abstract if available ────────────────────────────
    if abstract and len(abstract.strip()) > 100:
        # Light cleaning for display
        summary = re.sub(r"\s+", " ", abstract).strip()
        return summary[:max_chars]

    # ── Fallback: extract sentences from cleaned text ────────
    if not cleaned_text:
        return "No content available."

    sentences = split_sentences(cleaned_text)
    if not sentences:
        return cleaned_text[:max_chars]

    # Score each sentence
    informative_words = {
        "propose", "present", "introduce", "develop", "achieve", "demonstrate",
        "show", "result", "performance", "method", "approach", "model",
        "accuracy", "improve", "outperform", "dataset", "experiment",
        "framework", "system", "novel", "based", "using", "analysis",
        "evaluate", "algorithm", "deep", "learning", "network",
    }

    scored = []
    for idx, sent in enumerate(sentences):
        words = sent.lower().split()
        if len(words) < 5 or len(words) > 80:
            continue
        info_score = sum(1 for w in words if w in informative_words)
        # Favour early sentences (positional heuristic)
        pos_score = max(0, 1.0 - idx / max(len(sentences), 1) * 0.5)
        length_score = min(len(words) / 20, 1.0)
        total = info_score * 2 + pos_score + length_score
        scored.append((total, idx, sent))

    scored.sort(key=lambda x: (-x[0], x[1]))
    top = sorted(scored[:num_sentences], key=lambda x: x[1])  # restore order
    summary = " ".join(s for _, _, s in top)

    if not summary:
        summary = " ".join(sentences[:num_sentences])

    return summary[:max_chars]


# ──────────────────────────────────────────────────────────────
# Short Display Snippet
# ──────────────────────────────────────────────────────────────

def short_summary(summary: str, max_chars: int = 300) -> str:
    """
    Truncate a summary to `max_chars` at a word boundary for table display.

    Parameters
    ----------
    summary  : str
    max_chars: int

    Returns
    -------
    str
    """
    if not summary:
        return "—"
    if len(summary) <= max_chars:
        return summary
    truncated = summary[:max_chars].rsplit(" ", 1)[0]
    return truncated + " …"
