"""
rag/ingestion/metadata_extractor.py
Extract court metadata from judgment text using regex patterns.
"""
from __future__ import annotations

import re
from typing import Dict


_COURT_PATTERNS = [
    r"(Supreme Court of India)",
    r"(High Court of [A-Za-z\s]+)",
    r"(IN THE [A-Z\s]+ COURT)",
    r"(BEFORE THE [A-Z\s]+ COURT)",
    r"(National Company Law Tribunal)",
    r"(Income Tax Appellate Tribunal)",
    r"(National Consumer Disputes Redressal Commission)",
    r"(Consumer Disputes Redressal Commission)",
    r"(Armed Forces Tribunal)",
    r"(Central Administrative Tribunal)",
]

_CASE_PATTERNS = [
    r"(?:Civil Appeal|Criminal Appeal|W\.P\.|Writ Petition|SLP|Special Leave Petition|"
    r"C\.A\.|Crl\.A\.|O\.A\.|M\.A\.|Transfer Petition|Review Petition)\s*(?:No\.?|Nos?\.?)?\s*[\d/\(\)\-]+(?:/\d{4})?",
    r"[\w\s]+\s+vs?\s+[\w\s]+,\s+\d{4}",
    r"\(\d{4}\)\s+\d+\s+(?:SCC|AIR|SCR|SCJ|Bom\.?LR|MLJ|CLJ)\s+\d+",
    r"AIR\s+\d{4}\s+\w+\s+\d+",
    r"\d{4}\s+SCC\s+\(\w+\)\s+\d+",
]

_DATE_PATTERNS = [
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\,?\s+\d{4}",
    r"\d{1,2}/\d{1,2}/\d{4}",
    r"\d{4}-\d{2}-\d{2}",
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}",
]

_JUDGE_PATTERNS = [
    r"(?:HON['']?BLE\s+)?(?:MR\.|MRS\.|MS\.|DR\.|JUSTICE)\s+[A-Z][A-Za-z\.\s]+,?\s+J\.",
    r"(?:CORAM|Before)\s*:\s*([A-Z][A-Za-z\s\.,]+(?:J\.|C\.J\.))",
    r"J\.\s+(?:AND|&)\s+[A-Z][A-Za-z\s\.]+J\.",
]


def extract_court_metadata(text: str) -> Dict[str, str]:
    """
    Extract court, case_number, date_of_judgment, judges from text.
    Uses first 3000 chars (judgment header) for efficiency.
    """
    header = text[:3000] if text else ""

    court = ""
    for pattern in _COURT_PATTERNS:
        m = re.search(pattern, header, re.IGNORECASE)
        if m:
            court = m.group(0).strip().title()
            break

    case_number = ""
    for pattern in _CASE_PATTERNS:
        m = re.search(pattern, header, re.IGNORECASE)
        if m:
            case_number = m.group(0).strip()[:100]
            break

    date_of_judgment = ""
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, header, re.IGNORECASE)
        if m:
            date_of_judgment = m.group(0).strip()
            break

    judges: list = []
    for pattern in _JUDGE_PATTERNS:
        matches = re.findall(pattern, header, re.IGNORECASE)
        judges.extend(m.strip() for m in matches if m.strip())
    judges = list(dict.fromkeys(judges))[:5]  # deduplicate, keep order

    return {
        "court":            court,
        "case_number":      case_number,
        "date_of_judgment": date_of_judgment,
        "judges":           ", ".join(judges),
    }
