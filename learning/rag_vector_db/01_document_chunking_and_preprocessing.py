"""
=============================================================================
AeroVision AI - RAG & Vector Database Foundations: Module 01
=============================================================================
Document Chunking, Markdown Parsing, and Metadata Enrichment

Learning Objectives:
1. Understand trade-offs between fixed-window chunking vs semantic markdown chunking.
2. Ingest statutory regulatory files (FAA Part 107, OSHA, Industrial Perimeter).
3. Extract structured metadata: statute code, section title, regulatory category.
4. Normalize text and verify chunk size boundaries to prevent token truncation.
"""

import os
import re
from typing import List, Dict, Any


class MarkdownRegulatoryChunker:
    """
    Semantic chunker that splits regulatory markdown documents along section boundaries,
    preserving legal statute codes and enforcement procedures in chunk metadata.
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def parse_markdown_file(self, file_path: str, category: str) -> List[Dict[str, Any]]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Regulatory file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        doc_title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        doc_title = doc_title_match.group(1).strip() if doc_title_match else os.path.basename(file_path)

        # Split into sections based on '## '
        sections = re.split(r"(?m)^##\s+", content)
        chunks = []
        chunk_idx = 0

        for section in sections:
            section = section.strip()
            if not section or section.startswith("#"):
                continue

            lines = section.split("\n")
            section_title = lines[0].strip()
            section_body = "\n".join(lines[1:]).strip()

            # Extract statute reference if present
            statute_match = re.search(r"\*\*Statute Reference\*\*:\s*([^\n]+)", section_body)
            statute_code = statute_match.group(1).strip() if statute_match else "N/A"

            # Create structured text block
            text_block = f"[{doc_title} - {section_title}]\nStatute: {statute_code}\n{section_body}"

            chunk_meta = {
                "chunk_id": f"{category.lower()}_{chunk_idx:03d}",
                "doc_title": doc_title,
                "section_title": section_title,
                "statute_code": statute_code,
                "category": category,
                "text": text_block,
                "char_length": len(text_block)
            }
            chunks.append(chunk_meta)
            chunk_idx += 1

        return chunks


def test_chunking():
    print("=" * 70)
    print("STEP 1: PARSING REGULATORY CORPUS")
    print("=" * 70)

    chunker = MarkdownRegulatoryChunker()
    reg_dir = os.path.join("data", "regulations")

    corpus = [
        ("faa_part_107_drone_regulations.md", "FAA"),
        ("industrial_perimeter_security_protocols.md", "PERIMETER_SECURITY"),
        ("osha_facility_traffic_safety_rules.md", "OSHA")
    ]

    all_chunks: List[Dict[str, Any]] = []
    for filename, cat in corpus:
        filepath = os.path.join(reg_dir, filename)
        chunks = chunker.parse_markdown_file(filepath, cat)
        print(f"File: {filename:45s} -> Chunks Extracted: {len(chunks)} (Category: {cat})")
        all_chunks.extend(chunks)

    print("\n" + "=" * 70)
    print("STEP 2: VERIFYING CHUNK METADATA & REASONING BOUNDARIES")
    print("=" * 70)

    sample = all_chunks[0]
    print(f"Sample Chunk ID   : {sample['chunk_id']}")
    print(f"Doc Title         : {sample['doc_title']}")
    print(f"Section           : {sample['section_title']}")
    print(f"Statute Code      : {sample['statute_code']}")
    print(f"Character Length  : {sample['char_length']} chars")
    print("-" * 70)
    print("Sample Content:\n" + sample['text'][:250] + "...")
    print("-" * 70)

    # Validations
    assert len(all_chunks) >= 10, f"Expected at least 10 regulatory chunks, found {len(all_chunks)}"
    for c in all_chunks:
        assert c["statute_code"] != "N/A", f"Chunk {c['chunk_id']} failed to parse statute reference!"
        assert c["char_length"] > 100, f"Chunk {c['chunk_id']} is too brief!"
        assert c["category"] in ["FAA", "PERIMETER_SECURITY", "OSHA"]

    print(f"\n[SUCCESS] Extracted {len(all_chunks)} high-fidelity regulatory chunks across 3 domains!")


if __name__ == "__main__":
    test_chunking()
