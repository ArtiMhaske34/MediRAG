from __future__ import annotations

import glob
import os
from pathlib import Path

import click

from src.document_loaders import get_loader
from src.ingest_helpers import extract_fhir_json
from src.semantic_processor import SemanticProcessor

SOURCE_DIR = "source_documents"
INDEX_DIR = "faissindex"
SUPPORTED = {".json", ".pdf", ".docx", ".jpg", ".jpeg", ".png", ".mp3", ".wav"}


@click.command()
def main():
    """Extract supported files and rebuild a clean index from the source folder."""
    Path(SOURCE_DIR).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.*")))
    if not files:
        print(f"No files found in {SOURCE_DIR}/.")
        return

    corpus, errors = [], []
    for filepath in files:
        name = os.path.basename(filepath)
        try:
            if Path(name).suffix.lower() not in SUPPORTED:
                print(f"Skipping unsupported file: {name}")
                continue
            if name.lower().endswith(".json"):
                text = extract_fhir_json(filepath)
            else:
                text = "\n".join(get_loader(filepath).load(filepath))
            if text.strip():
                corpus.append({"text": text, "source": name})
                print(f"Read: {name}")
            else:
                errors.append(f"No readable text: {name}")
        except Exception as error:
            errors.append(f"{name}: {error}")
            print(f"Could not read {name}: {error}")

    if not corpus:
        print("No readable source documents were found; the existing index was left untouched.")
        return

    processor = SemanticProcessor(index_dir=INDEX_DIR)
    processor.chunk_and_embed(corpus, replace=True)
    print(f"Rebuilt the index from {len(corpus)} documents.")
    if errors:
        print("Files needing attention:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
