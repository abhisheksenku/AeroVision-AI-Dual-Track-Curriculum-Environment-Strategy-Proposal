# AeroVision AI - Phase 11: RAG Pipeline with Qdrant Vector Database

This directory contains foundational curriculum modules for Retrieval-Augmented Generation (RAG) and dense vector database engineering in production drone surveillance systems.

---

## Curriculum Overview

| Module | Script | Topic & Key Concept | Verification Metric |
| :--- | :--- | :--- | :--- |
| **01** | `01_document_chunking_and_preprocessing.py` | Markdown semantic chunking, statute reference extraction, and metadata enrichment across FAA, OSHA, and Perimeter Security rules. | 100% extraction of valid statutory references without truncation. |
| **02** | `02_embeddings_and_vector_math.py` | Dense vector representations in $\mathbb{R}^{384}$, Cosine Similarity, Euclidean ($L_2$) Distance, and proof of unit sphere metric equivalence. | $\text{Cosine} \approx 1 - \frac{1}{2} d_{L_2}^2$, semantic clustering fidelity. |
| **03** | `03_qdrant_vector_store_operations.py` | In-memory Qdrant client (`QdrantClient(":memory:")`), collection provisioning, upserting payload points, and metadata-filtered vector queries. | Top-K ranked semantic retrieval score $> 0.70$, filtered query isolation. |
| **04** | `04_rag_context_augmentation_evaluator.py` | Augmenting CV violation telemetry with statutory clauses; quantitative Information Retrieval (IR) evaluation. | $\text{MRR} \ge 0.75$, $\text{Recall@2} \ge 0.80$, exact statute citation verification. |

---

## Mathematical Formulation: Information Retrieval (IR) Evaluation

To mathematically benchmark retrieval performance over the regulatory corpus:

### 1. Precision@K
The fraction of top-$K$ retrieved statutory clauses that are directly relevant to the observed safety infraction:
$$\text{Precision@}K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$

### 2. Recall@K
The fraction of all relevant statutory clauses retrieved in the top-$K$ candidates:
$$\text{Recall@}K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$

### 3. Mean Reciprocal Rank (MRR)
Measures the position of the first relevant regulatory citation across a query set $Q$:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
Where $\text{rank}_i$ is the rank of the first relevant statute for query $i$ ($\text{rank} = 1 \implies \text{reciprocal rank} = 1.0$).
An $\text{MRR} \ge 0.75$ guarantees that the primary legal justification appears at rank 1 or rank 2.
