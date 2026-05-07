"""
Admin tool: upload files to the shared VIFHE knowledge base namespace.

Usage:
    python admin_upload.py path/to/file1.pdf path/to/file2.txt ...
    python admin_upload.py --list           # see what's currently loaded
    python admin_upload.py --clear          # wipe the knowledge base
    python admin_upload.py --delete <name>  # remove one file
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import rag

NAMESPACE = os.getenv("SHARED_NAMESPACE", "vifhe-kb")


def upload_file(path: Path):
    if not path.exists():
        print(f"  [X] {path}: file not found")
        return
    raw = path.read_bytes()
    try:
        text = rag.read_file(path.name, raw)
        chunks = rag.chunk_text(text, source=path.name)
        if not chunks:
            print(f"  [X] {path.name}: no extractable text")
            return
        rag.upload_chunks(chunks, namespace=NAMESPACE)
        print(f"  [OK] {path.name}: {len(chunks)} chunks uploaded")
    except Exception as e:
        print(f"  [X] {path.name}: {e}")


def list_sources():
    sources = rag.list_sources(NAMESPACE)
    if not sources:
        print("Knowledge base is empty.")
        return
    print(f"Knowledge base ({NAMESPACE}):")
    for s in sources:
        print(f"  - {s['source']:<40}  {s['chunk_count']} chunks")


def clear_all():
    confirm = input(f"Wipe all files from '{NAMESPACE}'? Type 'yes': ")
    if confirm.strip().lower() == "yes":
        rag.clear_namespace(NAMESPACE)
        print("[OK] Cleared.")
    else:
        print("Aborted.")


def delete_one(name: str):
    deleted = rag.delete_source(name, namespace=NAMESPACE)
    print(f"[OK] Deleted {deleted} chunks for '{name}'")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    if args[0] == "--list":
        list_sources()
    elif args[0] == "--clear":
        clear_all()
    elif args[0] == "--delete" and len(args) >= 2:
        delete_one(args[1])
    else:
        for arg in args:
            upload_file(Path(arg))
        print()
        list_sources()
