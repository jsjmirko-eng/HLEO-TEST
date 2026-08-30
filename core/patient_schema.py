"""
Pydantic schema for patient-reported experience extracted from Reddit posts.
Captures the personal health journey: condition, treatments tried,
outcomes, side effects, emotional impact, and verbatim key quotes.
"""
from typing import List, Optional
from pydantic import BaseModel


class PatientExperienceProfile(BaseModel):
    """
    Structured representation of a patient's self-reported clinical experience.
    Extracted by LLM from free-text posts (primarily Reddit).
    """

    # Demographics (extracted from text, never inferred)
    age: Optional[str] = None
    sex: Optional[str] = None
    country: Optional[str] = None

    # Core condition and history
    condition: Optional[str] = None               # e.g. "androgenetic alopecia"
    condition_duration: Optional[str] = None      # e.g. "3 years"
    family_history: Optional[str] = None

    # Treatment journey
    treatments_tried: List[str] = []              # all treatments mentioned historically
    current_treatment: Optional[str] = None       # most recent / ongoing
    dosages: List[str] = []
    treatment_duration: List[str] = []

    # Outcomes and effects
    reported_outcomes: List[str] = []             # what the patient says happened
    side_effects: List[str] = []
    discontinuation_reason: Optional[str] = None  # why they stopped a treatment

    # Subjective experience
    emotional_impact: Optional[str] = None        # brief summary of psychological impact
    quality_of_life_notes: Optional[str] = None
    unmet_needs: List[str] = []                   # things they wish existed or hadn't happened

    # Source material
    key_quotes: List[str] = []                    # verbatim phrases from the post (max 3)
    experience_summary: Optional[str] = None      # 1-2 sentence neutral summary of the post

    # Evidence metadata
    post_length_chars: Optional[int] = None       # filled by code, not LLM
    extraction_confidence: Optional[str] = None   # "high" | "medium" | "low" (LLM-assessed)
