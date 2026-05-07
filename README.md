# VIFHE Support Chatbot

AI-powered support chatbot for VIFHE (Virtual Institute for Higher Education):
- **Pinecone** stores vectors + does embeddings (free tier, hosted model)
- **DeepSeek** generates answers
- **Flask + gunicorn** backend, Railway-ready
- **Shared knowledge base** — admin pre-loads files, visitors query
- **Rate limits + file caps** so a bot can't drain your wallet
- **Admin endpoints** so you can manage the knowledge base remotely

---

## Quick start (local dev)

```bash
# 1. Install
python -m venv .venv
source .venv/bin/activate         # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env — paste real keys for PINECONE_API_KEY and DEEPSEEK_API_KEY
# Generate a FLASK_SECRET_KEY:
python -c "import secrets; print(secrets.token_hex(32))"
# (Optional) Generate ADMIN_TOKEN the same way

# 3. Create Pinecone index (one time)
python setup_index.py

# 4. Run
python app.py
# open http://localhost:5000
```

---

## Loading the knowledge base

Use `admin_upload.py` to manage the shared knowledge base that all visitors query against.

```bash
# Upload files
python admin_upload.py vifhe_files/admissions.pdf vifhe_files/programs.md

# List what's loaded
python admin_upload.py --list

# Remove a specific file
python admin_upload.py --delete admissions.pdf

# Wipe the entire knowledge base
python admin_upload.py --clear
```

Visitors of the web app will query whatever is in this shared namespace (`vifhe-kb` by default). They cannot upload, delete, or modify files — only ask questions.

---

## Deploy to Railway

See **`RAILWAY.md`** for step-by-step deployment.

---

## Built-in protections

| Protection | Limit | Where to change |
|---|---|---|
| Admin-only uploads | Upload/delete/clear require `ADMIN_TOKEN` | `app.py` → `@admin_required` |
| Query rate limit | 30 / min / IP | `app.py` → `@limiter.limit("30 per minute")` |
| Upload rate limit | 8 / hour / IP | `app.py` → `@limiter.limit("8 per hour")` |
| File size | 5 MB / file | `app.py` → `MAX_FILE_BYTES` |
| Total upload size | 20 MB / request | `app.py` → `MAX_TOTAL_UPLOAD_BYTES` |
| Chunks per upload | 80 max | `app.py` → `MAX_CHUNKS_PER_UPLOAD` |
| Question length | 2000 chars | `app.py` → `query()` route |
| `top_k` clamp | 1–15 | `app.py` → `query()` route |

---

## Admin endpoints

Set `ADMIN_TOKEN` in `.env` (or Railway env vars) to enable. All require:
```
Authorization: Bearer <ADMIN_TOKEN>
```

```bash
# List every namespace + vector count
curl https://your-app.up.railway.app/api/admin/namespaces \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"

# Clear one namespace
curl -X POST https://your-app.up.railway.app/api/admin/clear \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"namespace": "vifhe-kb"}'

# Nuke ALL namespaces (use with care)
curl -X POST https://your-app.up.railway.app/api/admin/clear-all \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

---

## Project structure

```
pinecone-deepseek-rag/
├── README.md
├── RAILWAY.md             # deployment walkthrough
├── requirements.txt
├── runtime.txt            # Python version pin
├── Procfile               # gunicorn start command
├── .env.example
├── .gitignore
├── setup_index.py         # one-time index creation
├── admin_upload.py        # CLI tool to load knowledge base
├── app.py                 # Flask routes + rate limiting
├── rag.py                 # chunking + Pinecone + DeepSeek
├── static/
│   ├── style.css
│   └── app.js
└── templates/
    └── index.html
```

---

## How it works

```
Admin runs admin_upload.py  → files chunked → Pinecone embeds + stores in "vifhe-kb" namespace
Visitor asks a question     → Pinecone embeds query → searches shared namespace
                            → top-k chunks sent to DeepSeek as context
                            → answer returned (no source citations shown)
```

All visitors query the same shared knowledge base. No per-session isolation — the knowledge base is curated by the admin.

---

## Monitoring

Monitor these dashboards regularly:
- **DeepSeek dashboard:** https://platform.deepseek.com → check usage daily
- **Pinecone dashboard:** https://app.pinecone.io → vector count + namespace list
- **Railway dashboard:** logs + bandwidth

Set up email alerts on DeepSeek when balance drops below a threshold.

---

## Troubleshooting

**`PineconeApiException: 404 not found`** → run `python setup_index.py`

**`429 Rate limit exceeded`** during normal use → bump limits in `app.py`

**Sessions not persisting after restart** → set `FLASK_SECRET_KEY` to a fixed value (not the auto-generated one)

**`413 Payload too large`** → file > 5MB, or total upload > 20MB
