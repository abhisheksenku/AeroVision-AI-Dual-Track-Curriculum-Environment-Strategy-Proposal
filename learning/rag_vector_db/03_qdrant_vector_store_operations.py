"""
=============================================================================
AeroVision AI - RAG & Vector Database Foundations: Module 03
=============================================================================
Qdrant Vector Database Operations, Indexing, and Payload Filtering

Learning Objectives:
1. Initialize local in-memory QdrantClient with cosine distance configuration.
2. Ingest and embed statutory regulatory documents into a Qdrant collection.
3. Execute semantic vector searches with Top-K ranked similarity retrieval.
4. Apply structured metadata payload filters (e.g. category matching) in conjunction with vector distance.
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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

import importlib

chunking_mod = importlib.import_module("learning.rag_vector_db.01_document_chunking_and_preprocessing")
MarkdownRegulatoryChunker = chunking_mod.MarkdownRegulatoryChunker

embed_mod = importlib.import_module("learning.rag_vector_db.02_embeddings_and_vector_math")
SemanticDenseEmbedder = embed_mod.SemanticDenseEmbedder


def test_qdrant_operations():
    print("=" * 70)
    print("STEP 1: INITIALIZING IN-MEMORY QDRANT VECTOR DATABASE")
    print("=" * 70)

    client = QdrantClient(":memory:")
    collection_name = "aerovision_regulations"
    embedder = SemanticDenseEmbedder(dimension=384)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Collection '{collection_name}' created with Cosine metric in R^384.")

    print("\n" + "=" * 70)
    print("STEP 2: INGESTING & EMBEDDING REGULATORY CORPUS")
    print("=" * 70)

    chunker = MarkdownRegulatoryChunker()
    reg_dir = os.path.join("data", "regulations")
    corpus = [
        ("faa_part_107_drone_regulations.md", "FAA"),
        ("industrial_perimeter_security_protocols.md", "PERIMETER_SECURITY"),
        ("osha_facility_traffic_safety_rules.md", "OSHA")
    ]

    all_chunks = []
    for filename, cat in corpus:
        filepath = os.path.join(reg_dir, filename)
        all_chunks.extend(chunker.parse_markdown_file(filepath, cat))

    points = []
    for idx, c in enumerate(all_chunks):
        vec = embedder.encode(c["text"]).tolist()
        point = PointStruct(
            id=idx + 1,
            vector=vec,
            payload={
                "chunk_id": c["chunk_id"],
                "doc_title": c["doc_title"],
                "section_title": c["section_title"],
                "statute_code": c["statute_code"],
                "category": c["category"],
                "text": c["text"]
            }
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    print(f"Successfully upserted {len(points)} regulatory points with payloads.")

    print("\n" + "=" * 70)
    print("STEP 3: RUNNING SEMANTIC SEARCH QUERIES")
    print("=" * 70)

    # Query 1: Speeding infraction
    query_speed = "Automobile recorded traveling at high velocity exceeding the statutory speed limit"
    q_vec_speed = embedder.encode(query_speed).tolist()

    res_speed = client.query_points(
        collection_name=collection_name,
        query=q_vec_speed,
        limit=2
    )

    print(f"Query: '{query_speed}'")
    top_hit = res_speed.points[0]
    print(f"  Top Match (Score: {top_hit.score:.4f}):")
    print(f"  Statute  : {top_hit.payload['statute_code']}")
    print(f"  Section  : {top_hit.payload['section_title']}")
    print(f"  Category : {top_hit.payload['category']}")

    assert "OSHA" in top_hit.payload["category"] or "1910.178" in top_hit.payload["statute_code"] or "107.51" in top_hit.payload["statute_code"]
    assert top_hit.score > 0.70

    # Query 2: Geofence runway intrusion
    query_geo = "Heavy truck entered active runway perimeter gate without clearance"
    q_vec_geo = embedder.encode(query_geo).tolist()

    res_geo = client.query_points(
        collection_name=collection_name,
        query=q_vec_geo,
        limit=2
    )

    print(f"\nQuery: '{query_geo}'")
    top_geo = res_geo.points[0]
    print(f"  Top Match (Score: {top_geo.score:.4f}):")
    print(f"  Statute  : {top_geo.payload['statute_code']}")
    print(f"  Section  : {top_geo.payload['section_title']}")
    print(f"  Category : {top_geo.payload['category']}")

    assert "PERIMETER_SECURITY" in top_geo.payload["category"] or "AGOS" in top_geo.payload["statute_code"]
    assert top_geo.score > 0.70

    print("\n" + "=" * 70)
    print("STEP 4: SEMANTIC SEARCH WITH PAYLOAD METADATA FILTER")
    print("=" * 70)

    # Search for speed regulations strictly filtered by FAA category
    faa_filter = Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="FAA")
            )
        ]
    )

    res_filtered = client.query_points(
        collection_name=collection_name,
        query=q_vec_speed,
        query_filter=faa_filter,
        limit=2
    )

    filtered_hit = res_filtered.points[0]
    print("Query: Speed limits with FILTER: category == 'FAA'")
    print(f"  Filtered Hit Statute  : {filtered_hit.payload['statute_code']}")
    print(f"  Filtered Hit Category : {filtered_hit.payload['category']}")

    assert filtered_hit.payload["category"] == "FAA"
    assert "107.51" in filtered_hit.payload["statute_code"] or "107" in filtered_hit.payload["statute_code"]

    print("\n[SUCCESS] Module 03 Qdrant Vector Store Operations verified successfully!")


if __name__ == "__main__":
    test_qdrant_operations()
