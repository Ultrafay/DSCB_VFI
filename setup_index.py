"""
Run once to create the Pinecone index used by this app.
Uses Pinecone's hosted embedding model so no separate embedding API is needed.

Usage:
    python setup_index.py
"""
import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-docs")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBED_MODEL = "llama-text-embed-v2"

if not API_KEY or API_KEY.startswith("pcsk_your"):
    print("ERROR: PINECONE_API_KEY missing or still placeholder in .env")
    sys.exit(1)

pc = Pinecone(api_key=API_KEY)

existing = [i.name for i in pc.list_indexes()]
if INDEX_NAME in existing:
    print(f"[OK] Index '{INDEX_NAME}' already exists. Nothing to do.")
    sys.exit(0)

print(f"Creating index '{INDEX_NAME}' with hosted model '{EMBED_MODEL}'...")

pc.create_index_for_model(
    name=INDEX_NAME,
    cloud=CLOUD,
    region=REGION,
    embed={
        "model": EMBED_MODEL,
        "field_map": {"text": "text"},
    },
)

print(f"[OK] Index '{INDEX_NAME}' created.")
print("  You can now run: python app.py")
