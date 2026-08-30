import json
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMExtractor:
    def __init__(self) -> None:
        self.model = "gpt-4o"
        from core.llm_provider import build_provider
        self.client = build_provider()
        if self.client is None:
            logger.warning(
                "No LLM provider configured (OPENAI_API_KEY / PERPLEXITY_API_KEY) "
                "— LLM extraction disabled. Reddit posts will be skipped."
            )

    def extract(self, timeline_json: str):
        if self.client is None:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Cannot perform LLM extraction."
            )

        from core.schemas import ExtractedClinicalProfile

        system_prompt = (
            "You are a clinical information extraction engine.\n"
            "Your only task is to transform the received text into a JSON object "
            "that strictly conforms to the provided schema.\n"
            "Rules:\n"
            "- Extract ONLY information explicitly present in the text.\n"
            "- Do NOT invent data or make clinical inferences.\n"
            "- For missing values use null; for missing lists use [].\n"
            "- episode_id: generate a short unique slug from the text (e.g. 'ep-breast-cancer-2023').\n"
            "- user_id: use 'anonymous' if no username is present.\n"
            "- baseline_status.value must be one of: assente, lieve, moderata, elevata, non_deducibile.\n"
            "- post_treatment_status.value must be one of: tornata_come_prima, inferiore_a_prima, "
            "superiore_a_prima, non_stabilizzata, non_deducibile.\n"
            "- supporting_quotes: include verbatim excerpts from the text with their source.\n"
            "- Return ONLY a valid JSON object matching the schema."
        )

        schema = ExtractedClinicalProfile.model_json_schema()

        from core.llm_guard import call_llm_json
        data = call_llm_json(
            self.client,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Timeline:\n{timeline_json}\n\n"
                        f"Schema JSON:\n{json.dumps(schema, ensure_ascii=False)}"
                    ),
                },
            ],
            model=self.model,
            temperature=0.0,
            operation="llm_extractor",
        )

        from core.schemas import ExtractedClinicalProfile
        return ExtractedClinicalProfile.model_validate(data)
