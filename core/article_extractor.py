"""
LLM extractor for scientific articles (PubMed / EuropePMC / ClinicalTrials).
Uses the general ClinicalProfile schema rather than the hair-loss-specific
ExtractedClinicalProfile schema.
"""
import json
import logging

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ArticleExtractor:
    MODEL = "gpt-4o"

    def __init__(self):
        from core.llm_provider import build_provider
        self.client = build_provider()
        if self.client is None:
            logger.warning("No LLM provider configured — ArticleExtractor disabled.")

    def extract(self, title: str, abstract: str, source: str) -> dict:
        """
        Extract a structured clinical profile from an article title + abstract.
        Returns a plain dict matching ClinicalProfile (clinical_schema.py).
        Raises RuntimeError if no API key.
        """
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        from core.clinical_schema import ClinicalProfile

        schema = ClinicalProfile.model_json_schema()

        text = f"Source: {source}\nTitle: {title}\n\nAbstract:\n{abstract or '(no abstract available)'}"

        system_prompt = (
            "You are a clinical information extraction engine. "
            "Extract a structured clinical profile from the given article title and abstract.\n\n"
            "Rules:\n"
            "- Extract ONLY information explicitly stated in the text.\n"
            "- Do NOT invent data, make inferences, or fill fields not mentioned.\n"
            "- Use null for missing scalar fields; [] for missing list fields.\n"
            "- For 'treatments', list drug names, procedures, and interventions in the order they appear.\n"
            "- For 'dosages', provide the dose/route/frequency for each treatment in the same order.\n"
            "- For 'treatment_duration', provide the duration for each treatment in the same order.\n"
            "  (treatments, dosages, and treatment_duration are index-aligned parallel arrays.)\n"
            "- For 'outcomes', list efficacy results, response rates, survival data.\n"
            "- For 'adverse_effects', list all side effects and toxicities mentioned.\n"
            "- For 'study_population', state the sample size and study design concisely, "
            "  e.g. 'n=240, Phase III RCT' or 'n=18, retrospective cohort'.\n"
            "- For 'evidence_level', state the study design label only, "
            "  e.g. 'Phase III RCT', 'meta-analysis', 'case report', 'cohort study'.\n"
            "- Return ONLY a valid JSON object conforming strictly to the schema."
        )

        from core.llm_guard import call_llm_json
        raw = call_llm_json(
            self.client,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{text}\n\n"
                        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}"
                    ),
                },
            ],
            model=self.MODEL,
            temperature=0.0,
            operation="article_extract",
        )

        # Normalize LLM output to match ClinicalProfile expectations before validation
        if not isinstance(raw, dict):
            # LLM should return a dict; if not, raise to surface the error
            raise RuntimeError("Article extractor expected a JSON object from the LLM")

        # Fields that must be lists of strings per ClinicalProfile schema
        list_fields = [
            "diagnosis", "triggers", "symptoms",
            "treatments", "dosages", "treatment_duration",
            "outcomes", "adverse_effects",
            "laboratory_findings", "biopsy_findings", "imaging_findings",
            "timeline",
        ]

        for k in list_fields:
            v = raw.get(k)
            # Normalise None -> [] and scalars -> [str(value)]
            if v is None:
                raw[k] = []
            elif isinstance(v, list):
                normalized = []
                for item in v:
                    if item is None:
                        normalized.append("")
                    else:
                        normalized.append(str(item))
                raw[k] = normalized
            else:
                # Scalar provided (string/number) -> wrap into list
                raw[k] = [str(v)]

        # Ensure parallel arrays (treatments, dosages, treatment_duration) are index-aligned
        t_len = len(raw.get("treatments", []))
        for arr_name in ("dosages", "treatment_duration"):
            arr = raw.get(arr_name, [])
            if not isinstance(arr, list):
                arr = [str(arr)] if arr is not None else []
            if len(arr) < t_len:
                arr = arr + [""] * (t_len - len(arr))
            raw[arr_name] = [str(x) if x is not None else "" for x in arr]

        # Finally validate against Pydantic model to ensure shape correctness
        profile = ClinicalProfile.model_validate(raw)
        return profile.model_dump()
