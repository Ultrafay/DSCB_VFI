"""
Core RAG logic: chunking, Pinecone upserts, retrieval, DeepSeek chat.
All operations are scoped to a namespace (one per session in production).
"""
import os
import io
import re
import uuid
import requests
from typing import List, Dict, Tuple
from pinecone import Pinecone

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
BATCH_SIZE = 90  # Pinecone integrated-embed upsert hard limit is 96


# -------- Pinecone client (lazy singleton) --------
_pc = None
_index = None

def get_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = _pc.Index(os.getenv("PINECONE_INDEX_NAME", "rag-docs"))
    return _index


# -------- File reading --------
def read_file(filename: str, raw_bytes: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw_bytes))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise ValueError(f"Could not read PDF: {e}")
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="ignore")


# -------- Chunking --------
def chunk_text(text: str, source: str) -> List[Dict]:
    text = re.sub(r"\s+\n", "\n", text)
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                cut = text.rfind(sep, start + CHUNK_SIZE // 2, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        content = text[start:end].strip()
        if len(content) > 30:
            chunks.append({
                "id": f"{source}-{uuid.uuid4().hex[:8]}-{idx}",
                "text": content,
                "source": source,
                "chunk_index": idx,
            })
            idx += 1
        start = end - CHUNK_OVERLAP if end < len(text) else end
    return chunks


# -------- Upload --------
def upload_chunks(chunks: List[Dict], namespace: str) -> int:
    index = get_index()
    records = [
        {
            "_id": c["id"],
            "text": c["text"],
            "source": c["source"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        index.upsert_records(namespace=namespace, records=batch)
    return len(records)


# -------- Search --------
def search(query: str, namespace: str, top_k: int = 5) -> List[Dict]:
    index = get_index()
    res = index.search_records(
        namespace=namespace,
        top_k=top_k,
        inputs={"text": query},
        fields=["text", "source", "chunk_index"],
    )
    data = res.to_dict() if hasattr(res, "to_dict") else res
    hits = data.get("result", {}).get("hits", [])
    return [
        {
            "id": h.get("id_", h.get("_id", "")),
            "score": h.get("score_", h.get("_score", 0)),
            "text": h.get("fields", {}).get("text", ""),
            "source": h.get("fields", {}).get("source", "unknown"),
            "chunk_index": h.get("fields", {}).get("chunk_index", 0),
        }
        for h in hits
    ]


# -------- Stats --------
def namespace_stats(namespace: str) -> Dict:
    index = get_index()
    stats = index.describe_index_stats()
    ns_data = stats.get("namespaces", {}).get(namespace, {})
    all_namespaces = list(stats.get("namespaces", {}).keys())
    return {
        "total_vectors": ns_data.get("vector_count", 0),
        "namespaces_total": len(all_namespaces),
    }


def list_sources(namespace: str) -> List[Dict]:
    index = get_index()
    sources = {}
    try:
        for page in index.list(namespace=namespace, limit=99):
            if not page or not page.vectors:
                continue
            ids = [v.id for v in page.vectors]
            fetched = index.fetch(ids=ids, namespace=namespace)
            for vid, vec in fetched.vectors.items():
                meta = vec.metadata or {}
                src = meta.get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
    except Exception as e:
        print(f"list_sources error: {e}")
    return [{"source": k, "chunk_count": v} for k, v in sources.items()]


def delete_source(source: str, namespace: str) -> int:
    index = get_index()
    deleted = 0
    try:
        for page in index.list(namespace=namespace, limit=99):
            if not page or not page.vectors:
                continue
            ids = [v.id for v in page.vectors]
            fetched = index.fetch(ids=ids, namespace=namespace)
            to_delete = [
                vid for vid, vec in fetched.vectors.items()
                if (vec.metadata or {}).get("source") == source
            ]
            if to_delete:
                index.delete(ids=to_delete, namespace=namespace)
                deleted += len(to_delete)
    except Exception as e:
        print(f"delete_source error: {e}")
    return deleted


def clear_namespace(namespace: str) -> bool:
    index = get_index()
    try:
        index.delete(delete_all=True, namespace=namespace)
        return True
    except Exception as e:
        print(f"clear_namespace error: {e}")
        return False


def list_all_namespaces() -> List[Dict]:
    """Admin: list every namespace with vector counts."""
    index = get_index()
    stats = index.describe_index_stats()
    return [
        {"namespace": name, "vector_count": data.get("vector_count", 0)}
        for name, data in stats.get("namespaces", {}).items()
    ]


# -------- DeepSeek chat --------
def chat_with_llm(question: str, context_chunks: List[Dict]) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing in environment")

    if context_chunks:
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']} | chunk {c['chunk_index']}]\n{c['text']}"
            for c in context_chunks
        )
    else:
        context = "(No relevant documents found.)"

    system_prompt = (
        "You are the VIFHE Support assistant \u2014 friendly, concise, and helpful. "
        "Answer questions about VIFHE programs, admissions, fees, courses, and policies "
        "using ONLY the information in the provided context below. "
        "If the answer is not in the context, politely say you don't have that information "
        "and suggest the user contact VIFHE directly. "
        "Do not include citations, source filenames, or chunk references in your answer. "
        "Keep responses short and natural \u2014 like a helpful staff member, not a search engine.\n\n"
        f"CONTEXT:\n{context}"
    )

    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        timeout=60,
    )

    if not resp.ok:
        try:
            err = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            err = resp.text
        raise RuntimeError(f"DeepSeek API error ({resp.status_code}): {err}")

    return resp.json()["choices"][0]["message"]["content"]


def rag_query(question: str, namespace: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
    chunks = search(question, namespace=namespace, top_k=top_k)
    answer = chat_with_llm(question, chunks)
    return answer, chunks
