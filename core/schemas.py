from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class BaselineEnum(str, Enum):
    ASSENTE = "assente"
    LIEVE = "lieve"
    MODERATA = "moderata"
    ELEVATA = "elevata"
    NON_DEDUCIBILE = "non_deducibile"

class PostTreatmentEnum(str, Enum):
    TORNATA_COME_PRIMA = "tornata_come_prima"
    INFERIORE = "inferiore_a_prima"
    SUPERIORE = "superiore_a_prima"
    NON_STABILIZZATA = "non_stabilizzata"
    NON_DEDUCIBILE = "non_deducibile"

class ClinicalCategory(str, Enum):
    CAT_A = "A"
    CAT_B = "B"
    CAT_C = "C"
    CAT_D = "D"
    CAT_E = "E"

class EvidenceQuote(BaseModel):
    verbatim_text: str
    source_url: str
    post_date: str

class ClinicalStatus(BaseModel):
    value: BaselineEnum | PostTreatmentEnum
    supporting_quotes: List[EvidenceQuote] = Field(default_factory=list)
    support_strength: float = Field(ge=0.0, le=1.0)

class ExtractedClinicalProfile(BaseModel):
    episode_id: str
    user_id: str
    baseline_status: ClinicalStatus
    post_treatment_status: ClinicalStatus
    conflict_detected: bool

class ValidationItemResult(BaseModel):
    is_valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None

class ValidationReport(BaseModel):
    passed_validation: bool
    errors: List[ValidationItemResult] = Field(default_factory=list)

class JudgeResult(BaseModel):
    episode_id: str
    assigned_category: ClinicalCategory
    adjudication_required: bool
    final_confidence_score: float
