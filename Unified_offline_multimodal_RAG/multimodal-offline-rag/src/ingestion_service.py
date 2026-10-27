from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from src.document_loaders import get_loader
from src.ingest_helpers import extract_fhir_json
from src.semantic_processor import SemanticProcessor

BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = BASE_DIR / "source_documents"
INDEX_DIR = BASE_DIR / "faissindex"
META_PATH = BASE_DIR / "document_metadata.json"
SUPPORTED = {".json", ".pdf", ".docx"}
MAX_UPLOAD_BYTES = 60 * 1024 * 1024
MAX_RECORD_BYTES = 15 * 1024 * 1024
MAX_ARCHIVE_EXPANSION_BYTES = 100 * 1024 * 1024


def _safe_name(raw: str) -> str:
    name = PurePosixPath(raw.replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    if not name or name.startswith("."):
        raise ValueError("The uploaded file name is not valid.")
    return name[:180]


def _metadata() -> dict[str, Any]:
    try:
        import json
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_metadata(data: dict[str, Any]) -> None:
    import json
    META_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _extract(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return extract_fhir_json(str(path))
    loader = get_loader(str(path))
    return "\n".join(loader.load(str(path)))


def list_documents() -> list[dict[str, Any]]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = _metadata()
    result = []
    for path in sorted(SOURCE_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        result.append({
            "filename": path.name,
            "type": path.suffix.lower().lstrip("."),
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "synthetic": bool(metadata.get(path.name, {}).get("synthetic", False)),
        })
    return result


def _install_index(temp_index: Path) -> None:
    backup = BASE_DIR / ("faissindex_backup_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if INDEX_DIR.exists():
        INDEX_DIR.replace(backup)
    try:
        temp_index.replace(INDEX_DIR)
    except Exception:
        if backup.exists() and not INDEX_DIR.exists():
            backup.replace(INDEX_DIR)
        raise


def _append_index(files: list[Path]) -> int:
    """Append only new files to a copied index, then atomically install the result."""
    corpus = []
    for path in files:
        text = _extract(path).strip()
        if text:
            corpus.append({"text": text, "source": path.name})
    if not corpus:
        raise ValueError("No readable supported documents were found to index.")

    build_root = Path(tempfile.mkdtemp(prefix=".medirag-index-append-", dir=BASE_DIR))
    temp_index = build_root / "index"
    try:
        shutil.copytree(INDEX_DIR, temp_index)
        processor = SemanticProcessor(index_dir=str(temp_index))
        processor.chunk_and_embed(corpus, replace=False)
        _install_index(temp_index)
    finally:
        if build_root.exists():
            shutil.rmtree(build_root, ignore_errors=True)
    return len(corpus)


def _rebuild_index() -> int:
    """Build a fresh index off to the side, then swap it in so re-runs do not duplicate chunks."""
    files = [p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED]
    corpus = []
    for path in sorted(files):
        text = _extract(path).strip()
        if text:
            corpus.append({"text": text, "source": path.name})
    if not corpus:
        raise ValueError("No readable supported documents were found to index.")

    temp_index = Path(tempfile.mkdtemp(prefix=".medirag-index-", dir=BASE_DIR))
    try:
        processor = SemanticProcessor(index_dir=str(temp_index))
        processor.chunk_and_embed(corpus, replace=True)
        _install_index(temp_index)
    finally:
        if temp_index.exists():
            shutil.rmtree(temp_index, ignore_errors=True)
    return len(corpus)


def add_documents(files: list[tuple[str, bytes]], synthetic: bool = False) -> dict[str, Any]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = _metadata()
    saved: list[Path] = []
    skipped_duplicates: list[str] = []
    known_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED
    }
    try:
        for raw_name, content in files:
            name = _safe_name(raw_name)
            path = Path(name)
            if path.suffix.lower() not in SUPPORTED:
                raise ValueError(f"Unsupported file type: {path.suffix or '(no extension)'}")
            if not content or len(content) > MAX_RECORD_BYTES:
                raise ValueError(f"{name} is empty or exceeds the per-file size limit.")
            content_hash = hashlib.sha256(content).hexdigest()
            if content_hash in known_hashes:
                skipped_duplicates.append(name)
                continue
            known_hashes.add(content_hash)
            destination = SOURCE_DIR / name
            if destination.exists():
                stem, suffix = destination.stem, destination.suffix
                number = 2
                while destination.exists():
                    destination = SOURCE_DIR / f"{stem}_{number}{suffix}"
                    number += 1
            destination.write_bytes(content)
            try:
                if not _extract(destination).strip():
                    raise ValueError(f"No readable text was found in {destination.name}.")
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            saved.append(destination)
            metadata[destination.name] = {"synthetic": synthetic}
        if not saved:
            return {"added": [], "skipped_duplicates": skipped_duplicates, "indexed_documents": 0, "index_reused": True}
        _save_metadata(metadata)
        if (INDEX_DIR / "index.faiss").exists():
            indexed = _append_index(saved)
        else:
            indexed = _rebuild_index()
        return {"added": [item.name for item in saved], "skipped_duplicates": skipped_duplicates, "indexed_documents": indexed}
    except Exception:
        for path in saved:
            path.unlink(missing_ok=True)
            metadata.pop(path.name, None)
        _save_metadata(metadata)
        raise


def import_synthea_zip(archive_bytes: bytes, patient_limit: int = 5) -> dict[str, Any]:
    if len(archive_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("The ZIP archive exceeds the 60 MB upload limit.")
    if not 1 <= patient_limit <= 20:
        raise ValueError("Choose between 1 and 20 demo records per import.")
    selected: list[tuple[str, bytes]] = []
    expanded_bytes = 0
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(archive_bytes)) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".json"):
                    continue
                if info.file_size > MAX_RECORD_BYTES:
                    continue
                expanded_bytes += info.file_size
                if expanded_bytes > MAX_ARCHIVE_EXPANSION_BYTES:
                    raise ValueError("The ZIP expands beyond the 100 MB import limit.")
                name = _safe_name(info.filename)
                raw = archive.read(info)
                import json
                data = json.loads(raw)
                if not isinstance(data, dict) or data.get("resourceType") != "Bundle":
                    continue
                entries = data.get("entry", [])
                if not any(entry.get("resource", {}).get("resourceType") == "Patient" for entry in entries if isinstance(entry, dict)):
                    continue
                selected.append((name, raw))
                if len(selected) >= patient_limit:
                    break
    except zipfile.BadZipFile as error:
        raise ValueError("That file is not a readable ZIP archive.") from error
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("The archive contains an invalid JSON record.") from error
    if not selected:
        raise ValueError("No per-patient FHIR Bundle files were found in the archive.")
    return add_documents(selected, synthetic=True)


def delete_document(filename: str) -> dict[str, Any]:
    name = _safe_name(filename)
    if name != filename:
        raise ValueError("Invalid document name.")
    path = SOURCE_DIR / name
    if not path.is_file() or path.suffix.lower() not in SUPPORTED:
        raise FileNotFoundError("Document not found.")
    metadata = _metadata()
    trash = BASE_DIR / "removed_documents"
    trash.mkdir(exist_ok=True)
    destination = trash / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + name)
    path.replace(destination)
    metadata.pop(name, None)
    _save_metadata(metadata)
    try:
        indexed = _rebuild_index()
    except Exception:
        destination.replace(path)
        raise
    return {"deleted": name, "indexed_documents": indexed}


