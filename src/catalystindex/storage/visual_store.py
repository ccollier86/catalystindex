"""Storage for visual elements (images and tables extracted from documents).

Visual elements are stored separately from text chunks since they don't need embedding.
They're retrieved based on visual_element_ids in chunk metadata during search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..models.common import VisualElement, Tenant

logger = logging.getLogger(__name__)


class VisualStore:
    """Base class for visual element storage."""

    def store_visuals(
        self,
        tenant: Tenant,
        document_id: str,
        knowledge_base_id: str,
        visual_elements: List[VisualElement],
    ) -> None:
        """Store visual elements for a document."""
        raise NotImplementedError

    def get_visuals(
        self,
        tenant: Tenant,
        visual_element_ids: List[str],
        knowledge_base_id: str,
    ) -> List[VisualElement]:
        """Retrieve visual elements by their IDs."""
        raise NotImplementedError

    def get_document_visuals(
        self,
        tenant: Tenant,
        document_id: str,
        knowledge_base_id: str,
    ) -> List[VisualElement]:
        """Retrieve all visual elements for a document."""
        raise NotImplementedError


class InMemoryVisualStore(VisualStore):
    """In-memory visual element storage for development/testing."""

    def __init__(self) -> None:
        # Key: (tenant_key, kb_id, element_id) -> VisualElement
        self._elements: Dict[tuple[str, str, str], VisualElement] = {}
        # Index: (tenant_key, kb_id, document_id) -> List[element_id]
        self._doc_index: Dict[tuple[str, str, str], List[str]] = {}

    def _tenant_key(self, tenant: Tenant) -> str:
        return f"{tenant.org_id}:{tenant.workspace_id}"

    def store_visuals(
        self,
        tenant: Tenant,
        document_id: str,
        knowledge_base_id: str,
        visual_elements: List[VisualElement],
    ) -> None:
        """Store visual elements in memory."""
        if not visual_elements:
            return

        tenant_key = self._tenant_key(tenant)
        element_ids: List[str] = []

        for visual in visual_elements:
            key = (tenant_key, knowledge_base_id, visual.element_id)
            self._elements[key] = visual
            element_ids.append(visual.element_id)

        # Update document index
        doc_key = (tenant_key, knowledge_base_id, document_id)
        self._doc_index[doc_key] = element_ids

        logger.info(
            f"Stored {len(visual_elements)} visual elements for document '{document_id}' "
            f"in knowledge base '{knowledge_base_id}'"
        )

    def get_visuals(
        self,
        tenant: Tenant,
        visual_element_ids: List[str],
        knowledge_base_id: str,
    ) -> List[VisualElement]:
        """Retrieve visual elements by their IDs."""
        if not visual_element_ids:
            return []

        tenant_key = self._tenant_key(tenant)
        visuals: List[VisualElement] = []

        for element_id in visual_element_ids:
            key = (tenant_key, knowledge_base_id, element_id)
            visual = self._elements.get(key)
            if visual:
                visuals.append(visual)

        return visuals

    def get_document_visuals(
        self,
        tenant: Tenant,
        document_id: str,
        knowledge_base_id: str,
    ) -> List[VisualElement]:
        """Retrieve all visual elements for a document."""
        tenant_key = self._tenant_key(tenant)
        doc_key = (tenant_key, knowledge_base_id, document_id)

        element_ids = self._doc_index.get(doc_key, [])
        if not element_ids:
            return []

        return self.get_visuals(tenant, element_ids, knowledge_base_id)


class FileBasedVisualStore(VisualStore):
    """File-based visual element storage with JSON persistence."""

    def __init__(self, storage_dir: str | Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileBasedVisualStore initialized at {self.storage_dir}")

    def _tenant_dir(self, tenant: Tenant, knowledge_base_id: str) -> Path:
        """Get directory for tenant's knowledge base visuals."""
        tenant_key = f"{tenant.org_id}_{tenant.workspace_id}"
        kb_dir = self.storage_dir / tenant_key / knowledge_base_id / "visuals"
        kb_dir.mkdir(parents=True, exist_ok=True)
        return kb_dir

    def _visual_file(self, tenant: Tenant, knowledge_base_id: str, document_id: str) -> Path:
        """Get JSON file path for document's visuals."""
        return self._tenant_dir(tenant, knowledge_base_id) / f"{document_id}.json"

    def store_visuals(
        self,
        tenant: Tenant,
        document_id: str,
        knowledge_base_id: str,
        visual_elements: List[VisualElement],
    ) -> None:
        """Store visual elements to JSON file."""
        if not visual_elements:
            return

        file_path = self._visual_file(tenant, knowledge_base_id, document_id)

        # Convert VisualElements to dict for JSON serialization
        visuals_data = [
            {
                "element_id": v.element_id,
                "element_type": v.element_type,
                "artifact_uri": v.artifact_uri,
                "image_base64": v.image_base64,
                "page_number": v.page_number,
                "document_id": v.document_id,
                "coordinates": v.coordinates,
                "caption": v.caption,
                "related_chunk_ids": v.related_chunk_ids,
                "spatial_position": v.spatial_position,
            }
            for v in visual_elements
        ]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(visuals_data, f, indent=2)

        logger.info(
            f"Stored {len(visual_elements)} visual elements to {file_path}"
        )

    def get_visuals(
        self,
        tenant: Tenant,
        visual_element_ids: List[str],
        knowledge_base_id: str,
    ) -> List[VisualElement]:
        """
        Retrieve visual elements by their IDs.

        Note: This requires scanning document files since we don't have a reverse index.
        For production, consider using a database or adding an index file.
        """
        if not visual_element_ids:
            return []

        element_id_set = set(visual_element_ids)
        visuals: List[VisualElement] = []

        # Scan all visual files in the knowledge base
        kb_dir = self._tenant_dir(tenant, knowledge_base_id)
        for visual_file in kb_dir.glob("*.json"):
            try:
                with open(visual_file, "r", encoding="utf-8") as f:
                    visuals_data = json.load(f)

                for data in visuals_data:
                    if data["element_id"] in element_id_set:
                        visuals.append(
                            VisualElement(
                                element_id=data["element_id"],
                                element_type=data["element_type"],
                                page_number=data["page_number"],
                                document_id=data["document_id"],
                                artifact_uri=data.get("artifact_uri"),
                                image_base64=data.get("image_base64"),
                                coordinates=data.get("coordinates"),
                                caption=data.get("caption"),
                                related_chunk_ids=data.get("related_chunk_ids", []),
                                spatial_position=data.get("spatial_position"),
                            )
                        )

                        # Early exit if we found all requested visuals
                        if len(visuals) == len(visual_element_ids):
                            return visuals

            except Exception as exc:
                logger.warning(f"Failed to read visual file {visual_file}: {exc}")
                continue

        return visuals

    def get_document_visuals(
        self,
        tenant: Tenant,
        document_id: str,
        knowledge_base_id: str,
    ) -> List[VisualElement]:
        """Retrieve all visual elements for a document."""
        file_path = self._visual_file(tenant, knowledge_base_id, document_id)

        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                visuals_data = json.load(f)

            return [
                VisualElement(
                    element_id=data["element_id"],
                    element_type=data["element_type"],
                    page_number=data["page_number"],
                    document_id=data["document_id"],
                    artifact_uri=data.get("artifact_uri"),
                    image_base64=data.get("image_base64"),
                    coordinates=data.get("coordinates"),
                    caption=data.get("caption"),
                    related_chunk_ids=data.get("related_chunk_ids", []),
                    spatial_position=data.get("spatial_position"),
                )
                for data in visuals_data
            ]

        except Exception as exc:
            logger.error(f"Failed to load visuals from {file_path}: {exc}")
            return []


__all__ = ["VisualStore", "InMemoryVisualStore", "FileBasedVisualStore"]
