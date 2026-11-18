# AeroVision AI - Phase 10: LLM & Generative AI Foundations

This directory contains foundational curriculum modules for integrating Large Language Models (LLMs) and Multimodal Generative AI into production drone surveillance systems.

---

## Curriculum Overview

| Module | Script | Topic & Key Concept | Verification Metric |
| :--- | :--- | :--- | :--- |
| **01** | `01_gemini_api_client_and_models.py` | Google GenAI SDK (`google-genai`), API key resilience, token usage tracking, and offline deterministic fallback. | Client initialization, model selection (`gemini-2.5-flash`), offline resilience. |
| **02** | `02_prompt_engineering_and_cot.py` | Persona modeling, spatio-temporal telemetry injection, and 4-step Chain-of-Thought (CoT) causality reasoning. | Step-by-step reasoning validation, threat classification, and operational countermeasures. |
| **03** | `03_structured_outputs_and_schema_validation.py` | Structured JSON generation via Pydantic (`response_schema`, `response_mime_type`), validation error recovery. | Complete schema compliance, strict type casting, error boundary handling. |
| **04** | `04_grounding_and_hallucination_evaluator.py` | Multimodal Grounding Engine cross-referencing LLM outputs against Computer Vision ground truth. | Entity score, speed fidelity delta, incident matching, composite Grounding Score. |

---

## Mathematical Formulation: Grounding Fidelity Score

To ensure that autonomous flight reports never fabricate incidents or misreport speeds, the **Grounding Auditor** computes a composite metric:

$$\text{Grounding Score} = w_e \cdot S_{\text{entity}} + w_s \cdot S_{\text{speed}} + w_i \cdot S_{\text{incident}}$$

Where:
- $S_{\text{entity}} = 1.0 - \frac{|\text{Reported Classes} \setminus \text{CV Detected Classes}|}{|\text{Reported Classes}|}$
- $S_{\text{speed}} = \max\left(0, 1.0 - \frac{|v_{\text{reported}} - v_{\text{actual}}|}{v_{\text{actual}}}\right)$ (with $1.5\text{ km/h}$ tolerance)
- $S_{\text{incident}} = \frac{|\text{Valid Citing Tracks} \cap \text{Actual Violators}|}{|\text{Reported Incidents}|}$
- Weights: $w_e = 0.35$, $w_s = 0.35$, $w_i = 0.30$.

**Verdict Thresholds**:
- $\ge 0.90$: `VERIFIED_GROUNDED` (safe for automated operations)
- $0.70 - 0.89$: `MINOR_DEVIATION` (requires manual supervisor sign-off)
- $< 0.70$: `UNRELIABLE_HALLUCINATION` (rejected)
