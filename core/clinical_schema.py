from pydantic import BaseModel
from typing import List, Optional


class ClinicalProfile(BaseModel):
    """
    Structured clinical profile extracted by LLM from a scientific article abstract.
    All fields reflect information explicitly stated in the source text; nothing is inferred.
    """

    # --- Demographics (study population, not individual patient) ---
    patient_age: Optional[str] = None        # e.g. "mean 45 years", "adults 18-65"
    patient_sex: Optional[str] = None        # e.g. "predominantly female", "mixed"

    # --- Condition ---
    diagnosis: List[str] = []                # ICD-10-style condition names
    disease_stage: Optional[str] = None      # e.g. "moderate-to-severe", "Stage III"

    # --- Aetiology & presentation ---
    triggers: List[str] = []                 # precipitating / causative factors
    symptoms: List[str] = []                 # reported signs and symptoms

    # --- Intervention ---
    treatments: List[str] = []               # drug names, procedures, interventions (parallel to dosages)
    dosages: List[str] = []                  # dose/route/frequency; index-aligned with treatments
    treatment_duration: List[str] = []       # duration per treatment; index-aligned with treatments

    # --- Outcomes ---
    outcomes: List[str] = []                 # efficacy results, response rates, survival data
    adverse_effects: List[str] = []          # side effects and toxicities

    # --- Investigations ---
    laboratory_findings: List[str] = []      # blood tests, biomarkers
    biopsy_findings: List[str] = []          # histopathology
    imaging_findings: List[str] = []         # radiology, dermoscopy, trichoscopy

    # --- Study context ---
    timeline: List[str] = []                 # key dates, follow-up milestones
    study_population: Optional[str] = None   # sample size and design, e.g. "n=240 RCT"
    evidence_level: Optional[str] = None     # study design label, e.g. "Phase III RCT", "meta-analysis"