"""
VIFHE Support chatbot — Flask backend.

Routes:
  GET  /                       -> UI
  GET  /api/health             -> health check (Railway uses this)
  POST /api/upload             -> upload files (admin only)
  POST /api/query              -> RAG query (public)
  GET  /api/sources            -> list knowledge base files (public)
  POST /api/delete             -> delete a single source (admin only)
  POST /api/clear              -> clear knowledge base (admin only)
  GET  /api/stats              -> vector count (public)

Admin (require Authorization: Bearer <ADMIN_TOKEN>):
  GET  /api/admin/namespaces   -> list ALL namespaces + vector counts
  POST /api/admin/clear        -> clear a specific namespace (json body: {namespace})
  POST /api/admin/clear-all    -> nuke every namespace in the index
"""
import os
import secrets
import traceback
from functools import wraps

from flask import Flask, request, jsonify, render_template
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

import rag

# ---------- Config ----------
MAX_FILE_BYTES = 5 * 1024 * 1024          # 5 MB per file
MAX_TOTAL_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB per request
MAX_CHUNKS_PER_UPLOAD = 80                 # caps Pinecone usage per upload
SHARED_NAMESPACE = os.getenv("SHARED_NAMESPACE", "vifhe-kb")

SECRET_KEY = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # if empty, admin endpoints are disabled

# ---------- App setup ----------
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_TOTAL_UPLOAD_BYTES
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("FLASK_ENV") == "production"
app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 24 * 30  # 30 days

# Trust Railway's reverse proxy for correct client IPs
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Rate limiter (in-memory; swap to Redis if you scale beyond 1 worker)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["120 per hour"],
    storage_uri="memory://",
)

# ---------- Auth helper ----------
def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not ADMIN_TOKEN:
            return jsonify({"error": "Admin endpoints disabled (no ADMIN_TOKEN configured)"}), 403
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != ADMIN_TOKEN:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*a, **kw)
    return wrapper


# ---------- Public routes ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/upload", methods=["POST"])
@limiter.limit("8 per hour")
@admin_required
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    ns = SHARED_NAMESPACE
    results = []
    total_chunks = 0

    for f in files:
        try:
            raw = f.read()
            if len(raw) > MAX_FILE_BYTES:
                results.append({
                    "filename": f.filename,
                    "error": f"File exceeds {MAX_FILE_BYTES // (1024*1024)}MB limit",
                })
                continue

            text = rag.read_file(f.filename, raw)
            chunks = rag.chunk_text(text, source=f.filename)

            if not chunks:
                results.append({"filename": f.filename, "error": "No extractable text"})
                continue

            remaining = MAX_CHUNKS_PER_UPLOAD - total_chunks
            if remaining <= 0:
                results.append({
                    "filename": f.filename,
                    "error": f"Per-upload chunk limit reached ({MAX_CHUNKS_PER_UPLOAD})",
                })
                continue

            if len(chunks) > remaining:
                chunks = chunks[:remaining]
                truncated = True
            else:
                truncated = False

            uploaded = rag.upload_chunks(chunks, namespace=ns)
            total_chunks += uploaded
            results.append({
                "filename": f.filename,
                "chunks": uploaded,
                "chars": len(text),
                "truncated": truncated,
            })
        except Exception as e:
            traceback.print_exc()
            results.append({"filename": f.filename, "error": str(e)})

    return jsonify({"results": results})


@app.route("/api/query", methods=["POST"])
@limiter.limit("30 per minute")
def query():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    top_k = max(1, min(int(data.get("top_k", 5)), 15))

    if not question:
        return jsonify({"error": "Empty question"}), 400
    if len(question) > 2000:
        return jsonify({"error": "Question too long (max 2000 chars)"}), 400

    # Sanitize history: only valid role + string content, cap length
    raw_history = data.get("history", [])
    if not isinstance(raw_history, list):
        raw_history = []
    clean_history = [
        {"role": h["role"], "content": str(h["content"])[:4000]}
        for h in raw_history
        if isinstance(h, dict)
        and h.get("role") in ("user", "assistant")
        and h.get("content")
    ][-20:]  # cap at last 20 messages

    ns = SHARED_NAMESPACE
    try:
        answer, sources = rag.rag_query(
            question, namespace=ns, top_k=top_k, history=clean_history
        )
        return jsonify({
            "answer": answer,
            "sources": [
                {
                    "source": s["source"],
                    "chunk_index": s["chunk_index"],
                    "score": round(s["score"], 4),
                    "preview": s["text"][:200] + ("..." if len(s["text"]) > 200 else ""),
                }
                for s in sources
            ],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
        
# ---------- Error handlers ----------
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": f"Rate limit exceeded: {e.description}"}), 429


@app.errorhandler(413)
def payload_too_large(e):
    return jsonify({"error": f"Upload too large (max {MAX_TOTAL_UPLOAD_BYTES // (1024*1024)}MB total)"}), 413


# ---------- Local dev entry ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", os.getenv("FLASK_PORT", 5000)))
    print(f"\n  -> Open http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=True)
