import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from datetime import datetime, timezone
from core.validator import HLEOValidator
from core.schemas import ExtractedClinicalProfile, ClinicalStatus, EvidenceQuote
from core.judge import HLEOJudge, ClinicalCategory

def test_validator_detects_hallucination():
    profile = ExtractedClinicalProfile(
        episode_id="T1",
        user_id="U1",
        conflict_detected=False,
        baseline_status=ClinicalStatus(
            value="moderata",
            support_strength=0.9,
            supporting_quotes=[
                EvidenceQuote(
                    verbatim_text="capelli perfetti",
                    source_url="http://x.com",
                    post_date="2026-07-20T12:00:00Z",
                )
            ],
        ),
        post_treatment_status=ClinicalStatus(
            value="tornata_come_prima",
            support_strength=0.9,
            supporting_quotes=[],
        ),
    )

    raw = {"http://x.com": "ho perso i capelli"}

    report = HLEOValidator.validate(profile, raw, datetime.now(timezone.utc))

    print(report)
    print(report.errors)

    assert not report.passed_validation
    assert report.errors[0].error_code == "VAL_E01"
