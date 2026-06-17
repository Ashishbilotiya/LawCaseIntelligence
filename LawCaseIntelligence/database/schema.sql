-- schema.sql
-- LawCaseIntelligence database schema (SQLite / PostgreSQL compatible)

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents_in_project (
    id                TEXT PRIMARY KEY,
    project_id        TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    document_name     TEXT NOT NULL,
    document_location TEXT DEFAULT '',
    file_size_bytes   INTEGER DEFAULT 0,
    page_count        INTEGER DEFAULT 0,
    status            TEXT DEFAULT 'pending',
    uploaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS processed_judgments (
    id                         TEXT PRIMARY KEY,
    project_id                 TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    document_name              TEXT NOT NULL,
    court                      TEXT DEFAULT '',
    case_number                TEXT DEFAULT '',
    date_of_judgment           TEXT DEFAULT '',
    case_category              TEXT DEFAULT 'General',
    win_indicator              TEXT DEFAULT 'Neutral',
    outcome                    TEXT DEFAULT '',
    issue_json                 TEXT DEFAULT '{}',
    petitioner_json            TEXT DEFAULT '{}',
    respondent_json            TEXT DEFAULT '{}',
    statutes_json              TEXT DEFAULT '{}',
    precedents_json            TEXT DEFAULT '{}',
    reasoning_json             TEXT DEFAULT '{}',
    trends_json                TEXT DEFAULT '{}',
    case_summary               TEXT DEFAULT '',
    frequently_cited_sections  TEXT DEFAULT '[]',
    processing_log             TEXT DEFAULT '[]',
    created_at                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_judgments_project    ON processed_judgments(project_id);
CREATE INDEX IF NOT EXISTS idx_judgments_category   ON processed_judgments(case_category);
CREATE INDEX IF NOT EXISTS idx_judgments_win        ON processed_judgments(win_indicator);
CREATE INDEX IF NOT EXISTS idx_docs_project         ON documents_in_project(project_id);
CREATE INDEX IF NOT EXISTS idx_docs_status          ON documents_in_project(status);
