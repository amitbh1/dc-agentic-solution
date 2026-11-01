from __future__ import annotations

from ..observability import audit
from ..query.plan import PlannerOutput
from .base import PlanRequest
from .prompts import PROMPT_VERSION, SYSTEM, build_prompt


class GeminiLLM:
    """Gemini on Vertex AI with structured output. Untested against a live project."""

    def __init__(self, model: str, project: str, location: str, timeout_s: int = 30):
        from google import genai
        from google.genai import types

        self._types = types
        self._model = model
        self._client = genai.Client(
            vertexai=True, project=project, location=location,
            http_options=types.HttpOptions(timeout=timeout_s * 1000),
        )

    def plan(self, req: PlanRequest) -> PlannerOutput:
        resp = self._client.models.generate_content(
            model=self._model,
            contents=build_prompt(req),
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM,
                response_mime_type="application/json",
                response_schema=PlannerOutput,
                temperature=0.0,
                max_output_tokens=1024,
            ),
        )
        usage = getattr(resp, "usage_metadata", None)
        audit(
            "llm_call", model=self._model, prompt_version=PROMPT_VERSION,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            tokens=getattr(usage, "total_token_count", None),
        )
        if isinstance(resp.parsed, PlannerOutput):
            return resp.parsed
        # Raises pydantic.ValidationError on malformed output; the planner node treats that as repairable.
        return PlannerOutput.model_validate_json(resp.text or "")
