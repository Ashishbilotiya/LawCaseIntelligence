"""
frontend/flask_app.py — LawCaseIntelligence Flask + SocketIO Application
"""
from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from flask import (Flask, flash, jsonify, redirect, render_template,
                   request, send_from_directory, session, url_for)
from flask_socketio import SocketIO, emit, join_room
from werkzeug.utils import secure_filename

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_async_mode = "gevent" if os.getenv("RENDER") or os.getenv("ASYNC_MODE") == "gevent" else "threading"
socketio = SocketIO(cors_allowed_origins="*", async_mode=_async_mode, logger=False, engineio_logger=False)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
MAX_MB     = 100


def create_app() -> Flask:
    base = Path(__file__).parent
    app  = Flask(__name__, template_folder=str(base / "templates"), static_folder=str(base / "static"))
    app.secret_key = os.getenv("FLASK_SECRET", "lawcase-intelligence-secret-2026")
    app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @app.template_filter("regex_replace")
    def regex_replace(s, pattern, replacement=""):
        return re.sub(pattern, replacement, str(s))

    _register_routes(app)
    socketio.init_app(app)
    _register_socket_events()

    from database.database import init_db
    with app.app_context():
        init_db()
    return app


# ── DB helpers ────────────────────────────────────────────────────

def _db():
    from database.database import get_session
    return get_session()

def _models():
    from database.models import Project, DocumentInProject, ProcessedJudgment
    return Project, DocumentInProject, ProcessedJudgment

def _set_doc_status(doc_id: str, status: str) -> None:
    s = _db()
    _, DocumentInProject, _ = _models()
    d = s.query(DocumentInProject).filter_by(id=doc_id).first()
    if d:
        d.status = status
        s.commit()
    s.close()

def _validate_project_name(name: str) -> list:
    errors = []
    if not name:                errors.append("Project name is required.")
    elif len(name) < 3:         errors.append("Project name must be at least 3 characters.")
    elif len(name) > 100:       errors.append("Project name must be under 100 characters.")
    return errors


# ── Chat history helpers (SQLite-backed) ──────────────────────────

def _get_history(project_id: str) -> list:
    from database.chat_store import get_history
    return get_history(project_id)

def _get_summary(project_id: str) -> str:
    from database.chat_store import get_summary
    return get_summary(project_id)

def _save_history(project_id: str, history: list, summary: str = "") -> None:
    from database.chat_store import save_history
    save_history(project_id, history, summary)

def _clear_history(project_id: str) -> None:
    from database.chat_store import clear_history
    clear_history(project_id)


# ── Routes ────────────────────────────────────────────────────────

def _register_routes(app: Flask) -> None:

    @app.route("/")
    def index():
        s = _db(); Project, _, _ = _models()
        projects = s.query(Project).order_by(Project.created_at.desc()).all()
        result = [p.to_dict() for p in projects]; s.close()
        return render_template("index.html", projects=result)

    @app.route("/project/new", methods=["GET", "POST"])
    def new_project():
        if request.method == "POST":
            name = request.form.get("name","").strip()
            desc = request.form.get("description","").strip()
            side = request.form.get("our_argument_side","").strip()
            errors = _validate_project_name(name)
            if errors:
                for e in errors: flash(e, "error")
                return render_template("new_project.html", form=request.form, errors=errors)
            s = _db(); Project, _, _ = _models()
            p = Project(name=name, description=desc, our_argument_side=side)
            s.add(p); s.commit(); pid = p.id; s.close()
            flash(f'Project "{name}" created!', "success")
            return redirect(url_for("project_detail", project_id=pid))
        return render_template("new_project.html", form={}, errors=[])

    @app.route("/project/<project_id>")
    def project_detail(project_id):
        s = _db(); Project, DocumentInProject, ProcessedJudgment = _models()
        project = s.query(Project).filter_by(id=project_id).first()
        if not project:
            s.close(); flash("Project not found.", "error"); return redirect(url_for("index"))
        docs      = s.query(DocumentInProject).filter_by(project_id=project_id).all()
        judgments = s.query(ProcessedJudgment).filter_by(project_id=project_id).all()
        pd = project.to_dict(); dd = [d.to_dict() for d in docs]; jd = [j.to_dict() for j in judgments]
        s.close()
        return render_template("project.html", project=pd, documents=dd, judgments=jd)

    @app.route("/project/<project_id>/delete", methods=["DELETE"])
    def delete_project(project_id):
        s = _db(); Project, DocumentInProject, ProcessedJudgment = _models()
        project = s.query(Project).filter_by(id=project_id).first()
        if not project:
            s.close(); return jsonify({"success": False, "error": "Not found"}), 404
        pname = project.name
        try:
            # 1. Clean up embeddings from both project and global collections for all documents
            docs = s.query(DocumentInProject).filter_by(project_id=project_id).all()
            for doc in docs:
                # Delete PDF file from disk
                Path(doc.document_location).unlink(missing_ok=True)

                # Remove vectors from global and project collections
                try:
                    from rag.vectordb.collection_manager import delete_document_chunks
                    delete_document_chunks(doc.id, project_id, doc_name=doc.document_name)
                except Exception as ve:
                    logger.warning(f"Vector deletion failed for doc {doc.id}: {ve}")

            # 2. Delete database records
            s.query(ProcessedJudgment).filter_by(project_id=project_id).delete(synchronize_session=False)
            s.query(DocumentInProject).filter_by(project_id=project_id).delete(synchronize_session=False)
            s.delete(project)
            s.commit()
            s.close()

            # 3. Clear chat history and ensure the project collection is fully removed
            _clear_history(project_id)
            try:
                from rag.vectordb.collection_manager import delete_project_collection
                delete_project_collection(project_id)
                logger.info(f"Deleted ChromaDB collection for project {project_id}")
            except Exception as ve:
                logger.warning(f"Project collection deletion failed (non-fatal): {ve}")

            return jsonify({"success": True, "message": f'Project "{pname}" deleted.'})
        except Exception as e:
            s.rollback(); s.close()
            return jsonify({"success": False, "error": str(e)}), 500

    # ── Delete Document ───────────────────────────────────────────
    @app.route("/project/<project_id>/document/<doc_id>/delete", methods=["DELETE"])
    def delete_document(project_id, doc_id):
        s = _db(); _, DocumentInProject, ProcessedJudgment = _models()
        doc = s.query(DocumentInProject).filter_by(id=doc_id, project_id=project_id).first()
        if not doc:
            s.close(); return jsonify({"success": False, "error": "Document not found."}), 404

        doc_name     = doc.document_name
        doc_location = doc.document_location
        try:
            # 1. Delete PDF file from disk
            Path(doc_location).unlink(missing_ok=True)

            # 2. Cascade-delete judgment records (same session)
            deleted_j = s.query(ProcessedJudgment).filter_by(
                project_id=project_id, document_name=doc_name
            ).delete(synchronize_session=False)
            logger.info(f"Cascade-deleted {deleted_j} judgment(s) for '{doc_name}'")

            # 3. Delete document DB record
            s.delete(doc)
            s.commit()
            s.close()

            # 4. Delete ChromaDB vectors AFTER DB commit — pass doc_name for legacy chunks
            try:
                from rag.vectordb.collection_manager import delete_document_chunks
                deleted_v = delete_document_chunks(doc_id, project_id, doc_name=doc_name)
                logger.info(f"Removed {deleted_v} vectors for doc_id={doc_id} ('{doc_name}')")
            except Exception as ve:
                logger.warning(f"Vector deletion failed (non-fatal): {ve}")

            return jsonify({"success": True, "message": f'"{doc_name}" deleted.'})
        except Exception as e:
            s.rollback(); s.close()
            logger.error(f"delete_document failed: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # ── Delete Judgment ───────────────────────────────────────────
    @app.route("/project/<project_id>/judgment/<judgment_id>/delete", methods=["DELETE"])
    def delete_judgment(project_id, judgment_id):
        s = _db(); _, DocumentInProject, ProcessedJudgment = _models()
        j = s.query(ProcessedJudgment).filter_by(id=judgment_id, project_id=project_id).first()
        if not j:
            s.close(); return jsonify({"success": True, "message": "Already deleted."})

        jname = j.document_name
        try:
            # 1. Find associated doc_id to wipe embeddings
            doc = s.query(DocumentInProject).filter_by(
                project_id=project_id, document_name=jname
            ).first()
            if doc:
                from rag.vectordb.collection_manager import delete_document_chunks
                delete_document_chunks(doc.id, project_id, doc_name=jname)

            # 2. Delete the judgment record
            s.delete(j)
            s.commit()
            s.close()
            return jsonify({"success": True, "message": f'Judgment "{jname}" and associated embeddings deleted.'})
        except Exception as e:
            s.rollback(); s.close()
            return jsonify({"success": False, "error": str(e)}), 500

    # ── Resume Processing ─────────────────────────────────────────
    @app.route("/project/<project_id>/process/<doc_id>/resume", methods=["POST"])
    def resume_processing(project_id, doc_id):
        sid = request.json.get("sid") if request.is_json else None
        s = _db(); _, DocumentInProject, _ = _models()
        doc = s.query(DocumentInProject).filter_by(id=doc_id, project_id=project_id).first()
        if not doc:
            s.close(); return jsonify({"success": False, "error": "Document not found."}), 404
        doc_location = doc.document_location; doc_name = doc.document_name
        doc.status = "processing"; s.commit(); s.close()
        try:
            from rag.vectordb.collection_manager import delete_document_chunks
            delete_document_chunks(doc_id, project_id, doc_name=doc_name)
        except Exception as e:
            logger.warning(f"Resume vector cleanup failed: {e}")
        def run():
            from agents.graph.graph_builder import process_judgment
            try:
                result = process_judgment(pdf_path=doc_location, project_id=project_id,
                                          document_name=doc_name, socket_room=sid)
                if result.get("error"):
                    socketio.emit("pipeline_error", {"doc_id": doc_id, "error": result["error"]}, room=sid)
                    _set_doc_status(doc_id, "failed")
                else:
                    _set_doc_status(doc_id, "done")
                    socketio.emit("pipeline_complete", {"doc_id": doc_id, "doc_name": doc_name}, room=sid)
            except Exception as e:
                logger.error(f"Resume pipeline error: {e}")
                socketio.emit("pipeline_error", {"doc_id": doc_id, "error": str(e)}, room=sid)
                _set_doc_status(doc_id, "failed")
        threading.Thread(target=run, daemon=True).start()
        return jsonify({"success": True, "message": "Resume started."})

    # ── Upload PDF ────────────────────────────────────────────────
    @app.route("/project/<project_id>/upload", methods=["POST"])
    def upload_pdf(project_id):
        s = _db(); Project, _, _ = _models()
        project = s.query(Project).filter_by(id=project_id).first(); s.close()
        if not project:
            return jsonify({"success": False, "errors": ["Project not found."]}), 404
        file = request.files.get("pdf_file")
        if not file or file.filename == "":
            return jsonify({"success": False, "errors": ["No file selected."]}), 400
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"success": False, "errors": ["Only PDF files allowed."]}), 400

        safe_name = secure_filename(file.filename)

        # ── Duplicate check ───────────────────────────────────────
        s2 = _db(); _, DocumentInProject, _ = _models()
        existing = s2.query(DocumentInProject).filter_by(
            project_id=project_id, document_name=safe_name
        ).filter(DocumentInProject.status.in_(["queued","processing","done"])).first()
        s2.close()
        if existing:
            label = {"done":"already processed","processing":"currently being processed",
                     "queued":"already queued"}.get(existing.status, existing.status)
            return jsonify({"success":False,"errors":[f'"{safe_name}" is {label}. Delete it first to re-upload.'],
                            "duplicate":True,"existing_status":existing.status}), 409

        doc_id = str(uuid.uuid4())
        proj_dir = UPLOAD_DIR / project_id
        proj_dir.mkdir(parents=True, exist_ok=True)
        save_path = proj_dir / f"{doc_id}_{safe_name}"
        file.save(str(save_path))
        try:
            import fitz
            doc = fitz.open(str(save_path)); pages = len(doc); doc.close()
            if pages == 0:
                save_path.unlink(missing_ok=True)
                return jsonify({"success":False,"errors":["PDF has no pages."]}), 400
        except Exception:
            save_path.unlink(missing_ok=True)
            return jsonify({"success":False,"errors":["Invalid or corrupted PDF."]}), 400

        s3 = _db(); _, DocumentInProject, _ = _models()
        s3.add(DocumentInProject(id=doc_id, project_id=project_id, document_name=safe_name,
                                  document_location=str(save_path), status="queued",
                                  page_count=pages, file_size_bytes=save_path.stat().st_size))
        s3.commit(); s3.close()
        return jsonify({"success":True,"document_id":doc_id,"filename":safe_name,
                        "pages":pages,"message":f"Uploaded ({pages} pages). Ready to process."})

    # ── Process Document ──────────────────────────────────────────
    @app.route("/project/<project_id>/process/<doc_id>", methods=["POST"])
    def process_document(project_id, doc_id):
        sid = request.json.get("sid") if request.is_json else None
        s = _db(); _, DocumentInProject, _ = _models()
        doc = s.query(DocumentInProject).filter_by(id=doc_id, project_id=project_id).first()
        if not doc:
            s.close(); return jsonify({"success":False,"error":"Document not found."}), 404
        if doc.status == "processing":
            s.close(); return jsonify({"success":False,"error":"Already processing."}), 400
        doc_location = doc.document_location; doc_name = doc.document_name
        doc.status = "processing"; s.commit(); s.close()
        def run():
            from agents.graph.graph_builder import process_judgment
            try:
                result = process_judgment(pdf_path=doc_location, project_id=project_id,
                                          document_name=doc_name, socket_room=sid)
                if result.get("error"):
                    socketio.emit("pipeline_error", {"doc_id":doc_id,"error":result["error"]}, room=sid)
                    _set_doc_status(doc_id, "failed")
                else:
                    _set_doc_status(doc_id, "done")
                    socketio.emit("pipeline_complete", {"doc_id":doc_id,"doc_name":doc_name}, room=sid)
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                socketio.emit("pipeline_error", {"doc_id":doc_id,"error":str(e)}, room=sid)
                _set_doc_status(doc_id, "failed")
        threading.Thread(target=run, daemon=True).start()
        return jsonify({"success":True,"message":"Processing started."})

    # ── Judgment Detail ───────────────────────────────────────────
    @app.route("/judgment/<judgment_id>")
    def judgment_detail(judgment_id):
        s = _db(); _, _, ProcessedJudgment = _models()
        j = s.query(ProcessedJudgment).filter_by(id=judgment_id).first()
        jd = j.to_dict() if j else None; s.close()
        if not jd:
            flash("Judgment not found.", "error"); return redirect(url_for("index"))
        return render_template("judgment.html", judgment=jd)

    # ── Chat ──────────────────────────────────────────────────────
    @app.route("/project/<project_id>/chat", methods=["GET"])
    def chat(project_id):
        s = _db(); Project, _, _ = _models()
        project = s.query(Project).filter_by(id=project_id).first()
        pd = project.to_dict() if project else None; s.close()
        if not pd:
            flash("Project not found.", "error"); return redirect(url_for("index"))
        return render_template("chat.html", project=pd, chat_history=_get_history(project_id))

    @app.route("/project/<project_id>/chat/ask", methods=["POST"])
    def chat_ask(project_id):
        data = request.json or {}
        question = data.get("question","").strip()
        if not question:
            return jsonify({"error":"Empty question"}), 400
        history = _get_history(project_id); summary = _get_summary(project_id)
        try:
            from backend.services.chat.chat_router import chat_router
            result = chat_router(query=question, project_id=project_id,
                                 chat_history=history, summary=summary)
        except Exception as e:
            logger.error(f"[ChatAsk] Error: {e}"); return jsonify({"error":str(e)}), 500
        history.append({"role":"user","content":question})
        history.append({"role":"assistant","content":result.get("answer","")})
        _save_history(project_id, history, summary=result.get("summary", summary))
        return jsonify(result)

    @app.route("/project/<project_id>/chat/clear", methods=["POST"])
    def chat_clear(project_id):
        _clear_history(project_id); return jsonify({"success":True})

    @app.route("/api/query", methods=["POST"])
    def api_query():
        data = request.json or {}
        question = data.get("question","").strip(); pid = data.get("project_id","").strip()
        if not question or not pid:
            return jsonify({"error":"question and project_id required"}), 400
        try:
            from backend.services.chat.chat_router import chat_router
            return jsonify(chat_router(query=question, project_id=pid))
        except Exception as e:
            return jsonify({"error":str(e)}), 500

    @app.route("/health")
    def health():
        return jsonify({"status":"ok","app":"LawCaseIntelligence"})

    @app.route("/admin/reset-keys", methods=["POST"])
    def reset_keys():
        """Force-reset the API key manager state. Useful after a rate-limit cooldown expires
        or when new API keys are added."""
        try:
            from backend.services.llm.api_key_manager import get_api_key_manager
            mgr = get_api_key_manager()
            mgr.force_reset_all_keys()
            return jsonify({"success": True, "message": "All API keys reset to active state."})
        except Exception as e:
            logger.error(f"reset_keys failed: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(str(UPLOAD_DIR), filename)

    @app.errorhandler(404)
    def not_found(e): return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(e): return render_template("errors/500.html"), 500


# ── Socket events ─────────────────────────────────────────────────

def _register_socket_events() -> None:
    @socketio.on("connect")
    def on_connect(): logger.info(f"Socket connected: {request.sid}")

    @socketio.on("join")
    def on_join(data):
        room = data.get("room", request.sid); join_room(room); emit("joined", {"room":room})

    @socketio.on("disconnect")
    def on_disconnect(): logger.info(f"Socket disconnected: {request.sid}")


if __name__ == "__main__":
    app   = create_app()
    port  = int(os.getenv("PORT", 5001))
    debug = os.getenv("DEBUG","false").lower() == "true"
    print(f"\n{'='*55}\n  ⚖  LawCaseIntelligence\n  🌐  http://localhost:{port}\n{'='*55}\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=debug)
