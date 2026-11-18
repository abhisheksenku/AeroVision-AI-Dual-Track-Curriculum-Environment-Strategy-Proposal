"""
=============================================================================
AeroVision AI - RAG & Vector Database Foundations: Module 02
=============================================================================
Dense Semantic Embeddings & High-Speed Vector Mathematics

Learning Objectives:
1. Understand vector space representations (Dense Embeddings in R^D).
2. Formulate and benchmark Cosine Similarity, Dot Product, and Euclidean (L2) Distance.
3. Prove mathematical equivalence between L2 distance and Cosine similarity on unit spheres.
4. Implement CPU-optimized normalized semantic embedding encoder.
"""

import math
import hashlib
import numpy as np
from typing import List, Tuple


class SemanticDenseEmbedder:
    """
    High-performance, CPU-optimized semantic dense vector embedder.
    Generates unit-normalized 384-dimensional dense semantic vectors
    combining domain semantic clusters with lexical projections.
    """
    DOMAINS = {
        "speed": ["speed", "velocity", "fast", "slow", "km/h", "mph", "limit", "rate", "overrun"],
        "zone": ["zone", "perimeter", "geofence", "runway", "boundary", "restricted", "intrusion", "gate", "enter"],
        "pedestrian": ["pedestrian", "person", "human", "walk", "people", "worker", "crossing"],
        "vehicle": ["vehicle", "car", "truck", "bus", "aircraft", "drone", "industrial", "forklift"]
    }

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, text: str) -> np.ndarray:
        """
        Converts text string into unit-normalized dense float32 vector in R^384.
        """
        tokens = text.lower().replace("-", " ").replace("_", " ").replace(".", " ").split()
        vector = np.zeros(self.dimension, dtype=np.float32)

        # 1. Lexical hash projection (first 192 dimensions)
        for i, token in enumerate(tokens):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            pos = h % 192
            val = ((h >> 16) % 1000) / 500.0 - 1.0
            weight = 1.0 / (1.0 + 0.05 * min(i, 20))
            vector[pos] += val * weight

        # 2. Domain semantic clustering (subspaces in 192..384)
        subspace_size = (self.dimension - 192) // len(self.DOMAINS)
        for d_idx, (domain_name, keywords) in enumerate(self.DOMAINS.items()):
            domain_start = 192 + (d_idx * subspace_size)
            # Count matches
            matches = sum(1 for t in tokens if any(k in t for k in keywords))
            if matches > 0:
                # Activate domain subspace with coherent semantic signature
                for offset in range(subspace_size):
                    vector[domain_start + offset] += (matches * 1.5) * (1.0 if offset % 2 == 0 else 0.8)

        # L2 Unit Normalization: ||v|| = 1.0
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            vector = vector / norm
        else:
            vector = np.zeros(self.dimension, dtype=np.float32)
            vector[0] = 1.0

        return vector


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Computes Cosine Similarity: (u . v) / (||u|| * ||v||)"""
    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(dot / (norm_u * norm_v))


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Computes Euclidean (L2) Distance: sqrt(sum((u_i - v_i)^2))"""
    return float(np.linalg.norm(u - v))


def test_vector_math():
    print("=" * 70)
    print("STEP 1: DENSE VECTOR GENERATION & UNIT SPHERE NORMALIZATION")
    print("=" * 70)

    embedder = SemanticDenseEmbedder(dimension=384)

    text_speed = "Vehicle exceeded external roadway speed limit of 50 km/h."
    text_fast = "Fast moving passenger car observed at excessive velocity."
    text_geofence = "Unauthorized pedestrian entered restricted hazardous runway perimeter."

    v_speed = embedder.encode(text_speed)
    v_fast = embedder.encode(text_fast)
    v_geofence = embedder.encode(text_geofence)

    print(f"Embedding Dimension : {v_speed.shape[0]} (Type: {v_speed.dtype})")
    print(f"Norm v_speed        : {np.linalg.norm(v_speed):.6f}")
    print(f"Norm v_fast         : {np.linalg.norm(v_fast):.6f}")
    print(f"Norm v_geofence     : {np.linalg.norm(v_geofence):.6f}")

    assert abs(np.linalg.norm(v_speed) - 1.0) < 1e-5
    assert abs(np.linalg.norm(v_fast) - 1.0) < 1e-5

    print("\n" + "=" * 70)
    print("STEP 2: COMPUTING COSINE SIMILARITY VS EUCLIDEAN DISTANCE")
    print("=" * 70)

    cos_speed_fast = cosine_similarity(v_speed, v_fast)
    cos_speed_geo = cosine_similarity(v_speed, v_geofence)

    dist_speed_fast = euclidean_distance(v_speed, v_fast)
    dist_speed_geo = euclidean_distance(v_speed, v_geofence)

    print(f"Cos(Speed, Fast)    : {cos_speed_fast:+.4f} (Euclidean: {dist_speed_fast:.4f})")
    print(f"Cos(Speed, Geofence): {cos_speed_geo:+.4f} (Euclidean: {dist_speed_geo:.4f})")

    # Semantic relevance check: speed is closer to fast than to geofence
    assert cos_speed_fast > cos_speed_geo, "Semantic clustering failed!"
    assert dist_speed_fast < dist_speed_geo, "Distance metric ordering mismatch!"

    print("\n" + "=" * 70)
    print("STEP 3: MATHEMATICAL PROOF OF COSINE-L2 EQUIVALENCE")
    print("=" * 70)
    # Proof: for unit vectors: d_L2^2 = 2 - 2 * cos(theta) => cos = 1 - 0.5 * d_L2^2
    derived_cos = 1.0 - 0.5 * (dist_speed_fast ** 2)
    print(f"Direct Cosine       : {cos_speed_fast:.6f}")
    print(f"Derived from L2     : {derived_cos:.6f}")
    assert abs(cos_speed_fast - derived_cos) < 1e-5, "Spherical equivalence violated!"
    print("✓ Mathematical proof verified: Cosine distance and L2 ranking are strictly equivalent on unit spheres.")

    print("\n[SUCCESS] Module 02 Vector Mathematics verified successfully!")


if __name__ == "__main__":
    test_vector_math()
