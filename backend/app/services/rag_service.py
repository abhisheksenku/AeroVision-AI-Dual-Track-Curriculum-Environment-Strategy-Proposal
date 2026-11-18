"""
=============================================================================
AeroVision AI - Production RAG & Vector Database Service
=============================================================================
Manages statutory regulatory corpus indexing with Qdrant, semantic vector
similarity search, and legal context augmentation for flight compliance auditing.
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np

import backend.app.utils.qdrant_bootstrap
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

from backend.app.schemas.rag import RegulatoryCitation, AugmentedInfraction


class SemanticEmbedder:
    """CPU-optimized 384-dimensional dense semantic vector encoder."""
    DOMAINS = {
        "speed": ["speed", "velocity", "fast", "slow", "km/h", "mph", "limit", "rate", "overrun", "deceleration"],
        "zone": ["zone", "perimeter", "geofence", "runway", "boundary", "restricted", "intrusion", "gate", "enter"],
        "pedestrian": ["pedestrian", "person", "human", "walk", "people", "worker", "crossing", "safety"],
        "vehicle": ["vehicle", "car", "truck", "bus", "aircraft", "drone", "industrial", "forklift", "chokepoint"]
    }

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, text: str) -> List[float]:
        tokens = text.lower().replace("-", " ").replace("_", " ").replace(".", " ").split()
        vector = np.zeros(self.dimension, dtype=np.float32)

        for i, token in enumerate(tokens):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            pos = h % 192
            val = ((h >> 16) % 1000) / 500.0 - 1.0
            weight = 1.0 / (1.0 + 0.05 * min(i, 20))
            vector[pos] += val * weight

        subspace_size = (self.dimension - 192) // len(self.DOMAINS)
        for d_idx, (_, keywords) in enumerate(self.DOMAINS.items()):
            domain_start = 192 + (d_idx * subspace_size)
            matches = sum(1 for t in tokens if any(k in t for k in keywords))
            if matches > 0:
                for offset in range(subspace_size):
                    vector[domain_start + offset] += (matches * 1.5) * (1.0 if offset % 2 == 0 else 0.8)

        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            vector = vector / norm
        else:
            vector = np.zeros(self.dimension, dtype=np.float32)
            vector[0] = 1.0

        return vector.tolist()


class AeroVisionRAGService:
    """
    Singleton service managing the Qdrant regulatory vector database.
    """
    COLLECTION_NAME = "aerovision_regulations"

    def __init__(self, storage_path: Optional[str] = None):
        self.embedder = SemanticEmbedder(dimension=384)
        
        # Use in-memory Qdrant instance for zero-latency, thread-safe execution
        if storage_path and storage_path != ":memory:":
            os.makedirs(storage_path, exist_ok=True)
            try:
                self.client = QdrantClient(path=storage_path)
            except Exception:
                self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(":memory:")

        self._initialize_collection()

    def _initialize_collection(self):
        """Creates collection if not exists and ingests regulatory corpus."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            self._index_regulatory_documents()

    def _index_regulatory_documents(self):
        """Reads markdown files from data/regulations and populates Qdrant points."""
        reg_dir = os.path.join("data", "regulations")
        if not os.path.exists(reg_dir):
            return

        corpus_files = [
            ("faa_part_107_drone_regulations.md", "FAA"),
            ("industrial_perimeter_security_protocols.md", "PERIMETER_SECURITY"),
            ("osha_facility_traffic_safety_rules.md", "OSHA")
        ]

        points = []
        pid = 1

        for filename, category in corpus_files:
            filepath = os.path.join(reg_dir, filename)
            if not os.path.exists(filepath):
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            doc_title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            doc_title = doc_title_match.group(1).strip() if doc_title_match else filename

            sections = re.split(r"(?m)^##\s+", content)
            for section in sections:
                section = section.strip()
                if not section or section.startswith("#"):
                    continue

                lines = section.split("\n")
                section_title = lines[0].strip()
                section_body = "\n".join(lines[1:]).strip()

                statute_match = re.search(r"\*\*Statute Reference\*\*:\s*([^\n]+)", section_body)
                statute_code = statute_match.group(1).strip() if statute_match else "Regulatory Standard"

                text_block = f"[{doc_title} - {section_title}]\nStatute: {statute_code}\n{section_body}"
                vector = self.embedder.encode(text_block)

                point = PointStruct(
                    id=pid,
                    vector=vector,
                    payload={
                        "statute_code": statute_code,
                        "section_title": section_title,
                        "doc_title": doc_title,
                        "category": category,
                        "text": text_block
                    }
                )
                points.append(point)
                pid += 1

        if points:
            self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)

    def search_regulations(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None
    ) -> List[RegulatoryCitation]:
        """Performs semantic vector search with optional category filtering."""
        query_vector = self.embedder.encode(query)

        query_filter = None
        if category:
            query_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

        search_results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        )

        citations = []
        for p in search_results.points:
            citations.append(
                RegulatoryCitation(
                    statute_code=p.payload.get("statute_code", "N/A"),
                    section_title=p.payload.get("section_title", "General"),
                    category=p.payload.get("category", "General"),
                    relevance_score=round(float(p.score), 4),
                    text=p.payload.get("text", "")
                )
            )
        return citations

    def augment_violations(self, violations: List[Any]) -> List[AugmentedInfraction]:
        """Augments a list of violation records with matching statutory citations."""
        augmented = []
        for v in violations:
            t_id = getattr(v, "track_id", 0)
            c_name = getattr(v, "class_name", "entity")
            r_type = getattr(v, "rule_type", "VIOLATION")
            sev = getattr(v, "severity", "WARNING")
            det = getattr(v, "details", "")

            query = f"{c_name} {r_type} {det}"
            citations = self.search_regulations(query=query, top_k=2)

            augmented.append(
                AugmentedInfraction(
                    track_id=t_id,
                    class_name=c_name,
                    rule_type=r_type,
                    severity=sev,
                    details=det,
                    citations=citations
                )
            )
        return augmented

    def get_all_statutes(self) -> List[Dict[str, Any]]:
        """Retrieves all indexed regulatory sections."""
        res, _ = self.client.scroll(
            collection_name=self.COLLECTION_NAME,
            limit=50,
            with_payload=True
        )
        return [
            {
                "id": p.id,
                "statute_code": p.payload.get("statute_code"),
                "section_title": p.payload.get("section_title"),
                "category": p.payload.get("category"),
                "doc_title": p.payload.get("doc_title")
            }
            for p in res
        ]


_rag_service_instance: Optional[AeroVisionRAGService] = None

def get_rag_service() -> AeroVisionRAGService:
    """Lazy singleton factory for AeroVisionRAGService."""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = AeroVisionRAGService(storage_path="data/qdrant_db")
    return _rag_service_instance
