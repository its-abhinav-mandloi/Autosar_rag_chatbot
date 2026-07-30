"""
Document Metadata Store.

Tracks metadata about ingested documents:
- Document name, file path, page count
- Chunk count, ingestion time
- Embedding model version used

Stored as a JSON file for simplicity (no external DB dependency).
In production, this would be a proper database.

Continuation Note:
    This module is complete. Metadata is persisted at config.METADATA_STORE_PATH/documents.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.config import get_settings
from app.monitoring.logging_config import get_logger

logger = get_logger("metadata_store")


class MetadataStore:
    """JSON-file-backed metadata store for ingested documents."""

    def __init__(self):
        settings = get_settings()
        self._store_dir = settings.metadata_store_absolute_path
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._file_path = self._store_dir / "documents.json"
        self._data: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        """Load metadata from JSON file."""
        if self._file_path.exists():
            try:
                with open(self._file_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error("metadata_load_error", error=str(e))
                return {}
        return {}

    def _save(self) -> None:
        """Persist metadata to JSON file."""
        with open(self._file_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def record_ingestion(
        self,
        document_name: str,
        file_path: str,
        page_count: int,
        chunk_count: int,
        processing_time_seconds: float,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """Record metadata for a newly ingested document."""
        self._data[document_name] = {
            "document_name": document_name,
            "file_path": file_path,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "processing_time_seconds": round(processing_time_seconds, 2),
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "version": 1,
        }
        self._save()
        logger.info(
            "ingestion_recorded",
            document=document_name,
            pages=page_count,
            chunks=chunk_count,
        )

    def get_document(self, document_name: str) -> Optional[dict]:
        """Get metadata for a specific document."""
        return self._data.get(document_name)

    def get_all_documents(self) -> List[dict]:
        """Get metadata for all ingested documents."""
        return list(self._data.values())

    def delete_document(self, document_name: str) -> bool:
        """Remove metadata for a document (called during re-ingestion)."""
        if document_name in self._data:
            del self._data[document_name]
            self._save()
            return True
        return False

    def get_stats(self) -> dict:
        """Summary statistics for all ingested documents."""
        docs = self._data.values()
        return {
            "total_documents": len(self._data),
            "total_pages": sum(d.get("page_count", 0) for d in docs),
            "total_chunks": sum(d.get("chunk_count", 0) for d in docs),
            "documents": [d["document_name"] for d in docs],
        }


# Module-level singleton
_metadata_store: Optional[MetadataStore] = None


def get_metadata_store() -> MetadataStore:
    """Get or create the global MetadataStore singleton."""
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store
