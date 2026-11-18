"""
=============================================================================
AeroVision AI - RAG & Vector Database Foundations: Module 04
=============================================================================
RAG Context Augmentation, Citation Synthesis & Retrieval Evaluation Metrics

Learning Objectives:
1. Connect CV Violation Events to Qdrant Semantic Vector Retrieval.
2. Synthesize Citation-Augmented Prompts with verified legal statutory references.
3. Compute quantitative Information Retrieval (IR) metrics: Precision@K, Recall@K, and MRR.
4. Verify that augmented context eliminates legal hallucinations.
"""

import os
import sys
import importlib
from typing import List, Dict, Any, Set, Tuple

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import backend.app.utils.qdrant_bootstrap
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

chunking_mod = importlib.import_module("learning.rag_vector_db.01_document_chunking_and_preprocessing")
MarkdownRegulatoryChunker = chunking_mod.MarkdownRegulatoryChunker

embed_mod = importlib.import_module("learning.rag_vector_db.02_embeddings_and_vector_math")
SemanticDenseEmbedder = embed_mod.SemanticDenseEmbedder


class RAGContextAugmentor:
    """
    Retrieves relevant legal statutes from Qdrant and augments mission prompts
    with grounded regulatory citations.
    """
    def __init__(self, client: QdrantClient, collection_name: str, embedder: SemanticDenseEmbedder):
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder

    def retrieve_statutes_for_violation(self, violation_query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        vec = self.embedder.encode(violation_query).tolist()
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=vec,
            limit=top_k
        )
        citations = []
        for p in res.points:
            citations.append({
                "statute_code": p.payload["statute_code"],
                "section_title": p.payload["section_title"],
                "category": p.payload["category"],
                "score": round(p.score, 4),
                "text": p.payload["text"]
            })
        return citations

    def build_augmented_incident_context(self, violations: List[Dict[str, Any]]) -> str:
        """
        Builds a comprehensive legal context string citing exact regulatory statutes for each infraction.
        """
        context_blocks = []
        for v in violations:
            query = f"{v['class_name']} {v['rule_type']} {v['details']}"
            citations = self.retrieve_statutes_for_violation(query, top_k=2)

            citation_lines = "\n".join([
                f"   * Relevant Law: {c['statute_code']} ({c['section_title']}) [Relevance: {c['score']:.2f}]"
                for c in citations
            ])

            block = (
                f"- Infraction: Track #{v['track_id']} ({v['class_name']}) at t={v['timestamp_sec']}s\n"
                f"  Rule Type: {v['rule_type']} [Severity: {v['severity']}]\n"
                f"  Details: {v['details']}\n"
                f"  Statutory Legal Grounding:\n{citation_lines}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)


def compute_retrieval_metrics(
    queries_and_ground_truth: List[Tuple[str, Set[str]]],
    augmentor: RAGContextAugmentor,
    k: int = 2
) -> Dict[str, float]:
    """
    Computes Precision@K, Recall@K, and Mean Reciprocal Rank (MRR) across test queries.
    """
    precisions = []
    recalls = []
    reciprocal_ranks = []

    for query, relevant_statutes in queries_and_ground_truth:
        retrieved = augmentor.retrieve_statutes_for_violation(query, top_k=k)
        retrieved_codes = [r["statute_code"] for r in retrieved]

        # Precision@K: relevant retrieved / K
        hits = sum(1 for code in retrieved_codes if any(rel in code for rel in relevant_statutes))
        precisions.append(hits / k)

        # Recall@K: relevant retrieved / total relevant
        recalls.append(hits / max(len(relevant_statutes), 1))

        # Reciprocal Rank: 1 / rank of first relevant hit
        rr = 0.0
        for rank, code in enumerate(retrieved_codes, 1):
            if any(rel in code for rel in relevant_statutes):
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return {
        "precision_at_k": round(sum(precisions) / len(precisions), 3),
        "recall_at_k": round(sum(recalls) / len(recalls), 3),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3)
    }


def test_rag_evaluation():
    print("=" * 70)
    print("STEP 1: POPULATING REGULATORY VECTOR INDEX")
    print("=" * 70)

    client = QdrantClient(":memory:")
    collection = "rag_eval_collection"
    embedder = SemanticDenseEmbedder(dimension=384)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    chunker = MarkdownRegulatoryChunker()
    reg_dir = os.path.join("data", "regulations")
    corpus = [
        ("faa_part_107_drone_regulations.md", "FAA"),
        ("industrial_perimeter_security_protocols.md", "PERIMETER_SECURITY"),
        ("osha_facility_traffic_safety_rules.md", "OSHA")
    ]

    points = []
    pid = 1
    for filename, cat in corpus:
        filepath = os.path.join(reg_dir, filename)
        for c in chunker.parse_markdown_file(filepath, cat):
            points.append(
                PointStruct(
                    id=pid,
                    vector=embedder.encode(c["text"]).tolist(),
                    payload=c
                )
            )
            pid += 1

    client.upsert(collection_name=collection, points=points)
    print(f"Indexed {len(points)} statutory chunks into Qdrant.")

    augmentor = RAGContextAugmentor(client, collection, embedder)

    print("\n" + "=" * 70)
    print("STEP 2: AUGMENTING COMPUTER VISION VIOLATIONS WITH STATUTES")
    print("=" * 70)

    sample_violations = [
        {
            "track_id": 3,
            "class_name": "car",
            "timestamp_sec": 1.25,
            "rule_type": "SPEED_LIMIT",
            "severity": "CRITICAL",
            "details": "Vehicle exceeded 50.0 km/h limit (recorded 68.4 km/h)"
        },
        {
            "track_id": 7,
            "class_name": "truck",
            "timestamp_sec": 2.80,
            "rule_type": "RESTRICTED_ZONE_INTRUSION",
            "severity": "CRITICAL",
            "details": "Commercial heavy vehicle penetrated active runway perimeter gate"
        }
    ]

    augmented_context = augmentor.build_augmented_incident_context(sample_violations)
    print(augmented_context)

    assert "Relevant Law: 14 CFR § 107.51" in augmented_context or "1910.178" in augmented_context
    assert "Relevant Law:" in augmented_context

    print("\n" + "=" * 70)
    print("STEP 3: COMPUTING QUANTITATIVE INFORMATION RETRIEVAL (IR) METRICS")
    print("=" * 70)

    # Benchmark test suite
    test_queries = [
        ("Speed limit overrun by car traveling 70 km/h", {"107.51", "1910.178"}),
        ("Unauthorized pedestrian entering active runway perimeter", {"27001", "107.39", "AGOS"}),
        ("Vehicle driving opposite direction on one-way road", {"5(a)(1)"}),
        ("Drone pilot operating beyond visual line of sight", {"107.31"})
    ]

    metrics = compute_retrieval_metrics(test_queries, augmentor, k=2)
    print("Retrieval Quality Metrics (Top-2):")
    print(f"  Precision@2 : {metrics['precision_at_k']}")
    print(f"  Recall@2    : {metrics['recall_at_k']}")
    print(f"  MRR         : {metrics['mrr']}")

    assert metrics["precision_at_k"] >= 0.50, "Precision@K below threshold!"
    assert metrics["mrr"] >= 0.75, "Mean Reciprocal Rank below threshold!"
    print(f"✓ All retrieval metrics verified (MRR: {metrics['mrr']} >= 0.75).")

    print("\n[SUCCESS] Module 04 RAG Context Augmentation verified successfully!")


if __name__ == "__main__":
    test_rag_evaluation()
