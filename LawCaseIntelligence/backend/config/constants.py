"""
backend/config/constants.py
Application-wide constants for legal domain. Groq-only.
"""
from __future__ import annotations

# ── Legal Categories ──────────────────────────────────────────────
LEGAL_CATEGORIES = [
    "Criminal", "Civil", "Tax", "Property", "Employment",
    "Constitutional", "Family", "Corporate", "Intellectual Property",
    "Environmental", "Administrative", "Consumer", "Banking", "General",
]

# ── Win Indicators ────────────────────────────────────────────────
WIN_INDICATORS = [
    "Petitioner-favourable",
    "Respondent-favourable",
    "Neutral",
    "Partially Allowed",
]

# ── Outcome Types ─────────────────────────────────────────────────
OUTCOME_TYPES = [
    "Appeal Allowed",
    "Appeal Dismissed",
    "Petition Allowed",
    "Petition Dismissed",
    "Partly Allowed",
    "Remanded",
    "Settled",
    "Withdrawn",
]

# ── Court Hierarchy (India) ───────────────────────────────────────
COURT_HIERARCHY = [
    "Supreme Court of India",
    "High Court",
    "District Court",
    "Sessions Court",
    "Magistrate Court",
    "National Company Law Tribunal",
    "Income Tax Appellate Tribunal",
    "National Consumer Disputes Redressal Commission",
]

# ── Agent Names ───────────────────────────────────────────────────
AGENT_NAMES = {
    "issue":     "Issue Extraction Agent",
    "argument":  "Argument Analysis Agent",
    "law":       "Law & Statute Agent",
    "precedent": "Precedent Analysis Agent",
    "reasoning": "Reasoning & Verdict Agent",
    "master":    "Master Supervisor Agent",
}

# ── Processing Status ─────────────────────────────────────────────
STATUS_PENDING    = "pending"
STATUS_QUEUED     = "queued"
STATUS_PROCESSING = "processing"
STATUS_DONE       = "done"
STATUS_FAILED     = "failed"

# ── Export Formats ────────────────────────────────────────────────
EXPORT_FORMATS = ["pdf", "json", "csv", "markdown"]

# ── System Prompt Base ────────────────────────────────────────────
LEGAL_ANALYST_PERSONA = (
    "You are an expert Indian legal analyst with deep knowledge of the "
    "Supreme Court and High Court jurisprudence, Indian Penal Code, "
    "Code of Civil Procedure, and all major Indian statutes. "
    "Extract ONLY information explicitly present in the provided text. "
    "Be precise, use exact legal terminology. Do not hallucinate. "
    "Respond with a valid JSON object only — no markdown, no explanation, no extra text."
)

# Chunk sizing / large-doc threshold: centralized in backend.config.settings
# (chunk_size=2_000, chunk_overlap=200, large_page_threshold=100). Previously
# this file declared its own conflicting copies (3000/300/100) that were
# never read by the ingestion pipeline — removed to avoid drift.

# ── ChromaDB ──────────────────────────────────────────────────────
CHROMA_LEGAL_COLLECTION = "legal_documents"
CHROMA_METADATA_FIELDS  = [
    "source_doc", "page_num", "court",
    "case_number", "date", "project_id",
]

# ── File Constraints ──────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_MB   = 100
