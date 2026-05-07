# Railway Deployment Guide

End-to-end walkthrough from local code → public URL.

---

## Step 1 — Push project to GitHub (private repo)

```bash
cd pinecone-deepseek-rag
git init
git add .
git commit -m "Initial commit"
```

Create a **private** repo on GitHub, then:

```bash
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/pinecone-deepseek-rag.git
git push -u origin main
```

> ⚠️ Confirm `.env` is in `.gitignore` and was NOT committed. Run `git status` first if unsure.

---

## Step 2 — Create Railway project

1. Go to https://railway.com → sign in with GitHub
2. **New Project** → **Deploy from GitHub repo**
3. Select your repo
4. Railway auto-detects Python and reads `Procfile` + `requirements.txt` + `runtime.txt`

The first deploy will likely **fail** because env vars aren't set yet — that's expected.

---

## Step 3 — Set environment variables

In Railway → your service → **Variables** tab → add each:

| Name | Value |
|---|---|
| `PINECONE_API_KEY` | your real key |
| `PINECONE_INDEX_NAME` | `rag-docs` |
| `PINECONE_CLOUD` | `aws` |
| `PINECONE_REGION` | `us-east-1` |
| `DEEPSEEK_API_KEY` | your real key |
| `FLASK_SECRET_KEY` | run `python -c "import secrets; print(secrets.token_hex(32))"` and paste the output |
| `FLASK_ENV` | `production` |
| `ADMIN_TOKEN` | run `python -c "import secrets; print(secrets.token_urlsafe(32))"` and paste it |

> Don't set `PORT` — Railway injects it automatically.

After saving, Railway will redeploy automatically.

---

## Step 4 — Create the Pinecone index

This needs to happen ONCE, before the first request. Two options:

**Option A — locally (recommended):**
```bash
# from your local machine, with .env filled in
python setup_index.py
```

**Option B — on Railway via shell:**
1. Railway service → **Settings** → **Service** → ensure deploy succeeded
2. Click the three dots → **Open shell**
3. Run: `python setup_index.py`

---

## Step 5 — Get your public URL

1. Railway service → **Settings** → **Networking** → **Generate Domain**
2. You'll get something like `your-app-production.up.railway.app`
3. Open it in a browser

You should see the UI with a green status dot. Upload a file, ask a question, confirm it works.

---

## Step 6 — Test isolation

This is critical for public deployment:

1. Open the app in your normal browser → upload `test1.txt`
2. Open the app in **incognito/private window** → you should see **0 documents**
3. Upload `test2.txt` in incognito → query
4. Switch back to normal browser → query → should still only see `test1.txt`

If both windows see each other's files, sessions aren't working — check `FLASK_SECRET_KEY` is set.

---

## Step 7 — Test rate limits

```bash
# Try 35 queries in quick succession; the 31st+ should return 429
for i in {1..35}; do
  curl -s -o /dev/null -w "%{http_code} " https://your-app.up.railway.app/api/query \
    -H "Content-Type: application/json" -d '{"question":"hi"}'
done
```

Expect `200 200 200 ... 429 429 429` after the 30th request.

---

## Step 8 — Monitor

Bookmark these tabs and check daily for the first week:

- **DeepSeek usage:** https://platform.deepseek.com → Usage
- **Pinecone vectors:** https://app.pinecone.io → your index → namespaces tab
- **Railway logs:** your service → **Deploy Logs** + **HTTP Logs**

---

## Pulling the plug (if abuse happens)

If you wake up to a $50 DeepSeek bill:

1. **Railway dashboard** → your service → **Settings** → **Pause Service** (instant stop, no deletion)
2. Then investigate:
   ```bash
   # See who uploaded what
   curl https://your-app.up.railway.app/api/admin/namespaces \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
   ```
3. Wipe everything if needed:
   ```bash
   curl -X POST https://your-app.up.railway.app/api/admin/clear-all \
     -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
   ```

---

## Costs (approximate)

| Service | Free tier | What hits the limit |
|---|---|---|
| Railway | $5/month credit (Hobby plan) | App running 24/7 ≈ $3-5/month |
| Pinecone | 2GB storage, ~100K vectors | Total uploads across all visitors |
| DeepSeek | Pay-as-you-go (no free tier) | Total questions × tokens |

Realistically: with light public traffic, expect **$3-10/month total**.

---

## Updating the app later

```bash
# Make changes locally
git add .
git commit -m "what changed"
git push
```

Railway auto-deploys on every push to `main`. No manual step needed.
