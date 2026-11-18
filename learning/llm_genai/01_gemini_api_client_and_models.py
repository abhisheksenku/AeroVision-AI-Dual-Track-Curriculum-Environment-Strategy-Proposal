"""
=============================================================================
AeroVision AI - LLM & Generative AI Foundations: Module 01
=============================================================================
Google Gemini API Client, Architecture, and Parameter Mechanics

Learning Objectives:
1. Understand Google GenAI Python SDK (google-genai) client lifecycle.
2. Configure generation hyperparameters (temperature, top_p, top_k, max_output_tokens).
3. Implement resilient client with automatic offline simulation fallback when
   GEMINI_API_KEY is not configured, enabling reproducible CI/CD execution.
4. Measure prompt and response token counts for cost and context window profiling.
"""

import os
import sys
import time
from typing import Dict, Any, Optional

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class AeroVisionGeminiClient:
    """
    Production-ready wrapper for Google Gemini API with automatic fallback.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.is_live = False

        if self.api_key and GENAI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.is_live = True
            except Exception as e:
                print(f"[WARN] Failed to initialize live Gemini Client: {e}")
                self.client = None
        else:
            self.client = None

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Generates text completion with metadata tracking (tokens, latency, provider).
        """
        start_time = time.perf_counter()

        if self.is_live and self.client:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Approximate token counting if usage_metadata is present
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", len(prompt.split()) * 1.3)
            response_tokens = getattr(response.usage_metadata, "candidates_token_count", len(response.text.split()) * 1.3)

            return {
                "text": response.text,
                "model": self.model_name,
                "latency_ms": round(latency_ms, 2),
                "is_live": True,
                "prompt_tokens": int(prompt_tokens),
                "response_tokens": int(response_tokens)
            }
        else:
            # Deterministic simulation fallback for environments without API key
            time.sleep(0.05)  # simulate network hop
            latency_ms = (time.perf_counter() - start_time) * 1000
            simulated_response = (
                f"[AEROVISION SIMULATED BRIEFING - {self.model_name.upper()}]\n"
                f"Surveillance Mission Telemetry Evaluated.\n"
                f"Core finding: Drone video analytics detected normal traffic flow with 2 minor speed infractions.\n"
                f"Operational Status: COMPLIANT WITH OBSERVATIONS."
            )
            prompt_tokens = int(len(prompt.split()) * 1.3)
            response_tokens = int(len(simulated_response.split()) * 1.3)

            return {
                "text": simulated_response,
                "model": f"{self.model_name} (Simulation Fallback)",
                "latency_ms": round(latency_ms, 2),
                "is_live": False,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens
            }


def test_gemini_client():
    print("=" * 70)
    print("STEP 1: INITIALIZING AEROVISION GEMINI CLIENT")
    print("=" * 70)

    client = AeroVisionGeminiClient()
    print(f"GenAI SDK Available : {GENAI_AVAILABLE}")
    print(f"Live API Mode       : {client.is_live} (Key Present: {bool(client.api_key)})")
    print(f"Selected Model      : {client.model_name}")

    print("\n" + "=" * 70)
    print("STEP 2: RUNNING GENERATION TEST WITH SYSTEM INSTRUCTION")
    print("=" * 70)

    system_instruction = (
        "You are AeroVision AI Incident Commander, an expert aerial reconnaissance "
        "and traffic safety auditor. You produce concise, grounded, high-precision briefings."
    )
    prompt = (
        "Mission ID: MS-402\n"
        "Duration: 4.0 seconds\n"
        "Total Vehicles: 10\n"
        "Peak Speed: 42.5 km/h\n"
        "Violations: 0\n"
        "Summarize this flight mission in 2 sentences."
    )

    result = client.generate_text(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=0.2,
        max_output_tokens=256
    )

    print(f"Latency             : {result['latency_ms']} ms")
    print(f"Mode                : {'LIVE GOOGLE API' if result['is_live'] else 'DETERMINISTIC FALLBACK'}")
    print(f"Prompt Tokens       : {result['prompt_tokens']}")
    print(f"Response Tokens     : {result['response_tokens']}")
    print("-" * 70)
    print("Generated Briefing:\n", result["text"])
    print("-" * 70)

    assert len(result["text"]) > 20, "Generated text is unexpectedly empty!"
    assert result["prompt_tokens"] > 0
    assert result["response_tokens"] > 0
    print("\n[SUCCESS] Module 01 verified successfully!")


if __name__ == "__main__":
    test_gemini_client()
