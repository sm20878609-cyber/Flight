"""
pdf_utils.py
------------
Utilities for extracting text, title, and abstract from PDF files.
Uses PyMuPDF (fitz) as the primary parser with pdfplumber as fallback.
"""

import re
import io
import warnings

# Primary parser
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

# Fallback parser
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Core Extraction
# ──────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_obj, filename: str = "unknown.pdf") -> dict:
    """
    Extract raw text from an uploaded PDF file object.

    Parameters
    ----------
    file_obj : file-like object
        The PDF file uploaded via Streamlit (BytesIO or similar).
    filename : str
        Original filename, used as fallback title.

    Returns
    -------
    dict with keys:
        'filename'  : str
        'raw_text'  : str   – full extracted text
        'page_count': int
        'success'   : bool
        'error'     : str or None
    """
    result = {
        "filename": filename,
        "raw_text": "",
        "page_count": 0,
        "success": False,
        "error": None,
    }

    pdf_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj

    # ── Try PyMuPDF ──────────────────────────────────────────
    if FITZ_AVAILABLE:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text("text"))
            doc.close()
            raw = "\n".join(pages_text)
            if len(raw.strip()) > 50:
                result["raw_text"] = raw
                result["page_count"] = len(pages_text)
                result["success"] = True
                return result
        except Exception as e:
            result["error"] = f"PyMuPDF error: {e}"

    # ── Fallback: pdfplumber ─────────────────────────────────
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages_text = [p.extract_text() or "" for p in pdf.pages]
            raw = "\n".join(pages_text)
            if len(raw.strip()) > 50:
                result["raw_text"] = raw
                result["page_count"] = len(pages_text)
                result["success"] = True
                result["error"] = None
                return result
        except Exception as e:
            result["error"] = (result["error"] or "") + f" | pdfplumber error: {e}"

    if not result["success"]:
        if not result["error"]:
            result["error"] = "No PDF parser available (install PyMuPDF or pdfplumber)."
        result["raw_text"] = ""

    return result


# ──────────────────────────────────────────────────────────────
# Title Extraction
# ──────────────────────────────────────────────────────────────

def extract_title(raw_text: str, filename: str = "unknown.pdf") -> str:
    """
    Attempt to extract a paper title from the first lines of raw_text.

    Strategy:
    1. Scan the first 30 non-empty lines.
    2. Skip common header noise (page numbers, emails, URLs, dates).
    3. Pick the longest line that looks like a real title (≥ 10 chars,
       not ALL_CAPS noise, not starting with a digit/bullet).
    4. Fall back to the filename stem.

    Parameters
    ----------
    raw_text : str
    filename : str

    Returns
    -------
    str – extracted or fallback title
    """
    if not raw_text or not raw_text.strip():
        return _filename_to_title(filename)

    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    candidates = []

    noise_patterns = [
        r"^\d+$",                       # pure numbers
        r"^https?://",                   # URLs
        r"@",                            # emails
        r"^\d{1,2}/\d{1,2}/\d{2,4}",   # dates
        r"^(vol\.|volume|issue|doi|arxiv|preprint|submitted|accepted|published|copyright|©|\(c\))",  # metadata
        r"^(abstract|introduction|keywords?|references?|acknowledgements?)",
    ]

    for line in lines[:40]:
        if len(line) < 8:
            continue
        if any(re.search(p, line, re.IGNORECASE) for p in noise_patterns):
            continue
        # Penalise all-caps lines unless they look like acronyms
        if line.isupper() and len(line) > 20:
            continue
        candidates.append(line)
        if len(candidates) >= 5:
            break

    if candidates:
        # Prefer the longest candidate in the first 3
        best = max(candidates[:3], key=len)
        # Truncate very long titles
        if len(best) > 200:
            best = best[:200].rsplit(" ", 1)[0] + " …"
        return best

    return _filename_to_title(filename)


def _filename_to_title(filename: str) -> str:
    """Convert a filename to a readable title."""
    stem = filename.rsplit(".", 1)[0]          # remove extension
    stem = re.sub(r"[_\-]+", " ", stem)        # underscores/hyphens → spaces
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem.title() if stem else filename


# ──────────────────────────────────────────────────────────────
# Abstract Extraction
# ──────────────────────────────────────────────────────────────

def extract_abstract(raw_text: str, max_fallback_chars: int = 1500) -> str:
    """
    Extract the abstract section from raw PDF text.

    Strategy:
    1. Search for text between 'Abstract' and 'Introduction' (case-insensitive).
    2. If 'Introduction' is not found, grab up to 2 000 chars after 'Abstract'.
    3. Fall back to the first `max_fallback_chars` characters of the text.

    Parameters
    ----------
    raw_text : str
    max_fallback_chars : int

    Returns
    -------
    str – extracted abstract or first-N-chars fallback
    """
    if not raw_text:
        return ""

    # Pattern: content between Abstract and Introduction
    pattern = re.compile(
        r"abstract[\s\-:]*\n?(.*?)(?=\n\s*\d*\.?\s*introduction|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(raw_text)
    if match:
        abstract = match.group(1).strip()
        # Abstract sanity check: should have some substance
        if len(abstract) > 100:
            return abstract[:3000]  # cap at 3 000 chars

    # Fallback: first meaningful characters
    cleaned_start = re.sub(r"\s+", " ", raw_text[:max_fallback_chars * 2]).strip()
    return cleaned_start[:max_fallback_chars]
