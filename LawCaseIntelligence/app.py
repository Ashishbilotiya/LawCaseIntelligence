"""
app.py — LawCaseIntelligence Flask Application Entry Point
Run: python app.py  OR  gunicorn app:app
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from backend.config.logging_config import setup_logging
setup_logging(os.getenv("LOG_LEVEL", "INFO"))

from database.database import init_db
init_db()

from frontend.flask_app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    logger = logging.getLogger(__name__)
    logger.info(f"Starting LawCaseIntelligence on port {port}")
    print(f"\n{'='*60}")
    print(f"  ⚖️  LawCaseIntelligence — Legal AI Platform")
    print(f"  🌐  http://localhost:{port}")
    print(f"  🤖  Provider: Groq")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
