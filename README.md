# Pinecone + DeepSeek RAG — Production Build

Public-facing RAG app:
- **Pinecone** stores vectors + does embeddings (free tier, hosted model)
- **DeepSeek** generates answers
- **Flask + gunicorn** backend, Railway-ready
- **Per-session namespaces** so users only see their own files
- **Rate limits + file caps** so a bot can't drain your wallet
- **Admin endpoints** so you can wipe namespaces remotely

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

## Deploy to Railway

See **`RAILWAY.md`** for step-by-step deployment.

---

## Built-in protections

| Protection | Limit | Where to change |
|---|---|---|
| Per-session privacy | Each browser gets isolated Pinecone namespace | `app.py` → `get_session_namespace()` |
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
  -d '{"namespace": "u-abc123def456"}'

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
├── app.py                 # Flask routes + sessions + rate limiting
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
First visit       → Flask sets a session cookie → namespace "u-<random>" assigned
Upload file       → chunked → Pinecone embeds + stores in YOUR namespace
Query             → Pinecone embeds query → searches YOUR namespace only
                  → top-k chunks sent to DeepSeek as context
                  → answer streamed back with source citations
```

Different visitors = different namespaces = total isolation.

---

## Monitoring

You said no kill switch — so monitor manually:
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
