from core.schemas import ClinicalCategory, JudgeResult, BaselineEnum, PostTreatmentEnum

class HLEOJudge:
    @staticmethod
    def evaluate(base: str, post: str, valid: bool, strength: float, conflict: bool, ep_id: str) -> JudgeResult:
        if not valid or conflict:
            return JudgeResult(episode_id=ep_id, assigned_category=ClinicalCategory.CAT_E, adjudication_required=True, final_confidence_score=0.0)
        cat = ClinicalCategory.CAT_E
        if base in [BaselineEnum.ASSENTE.value, BaselineEnum.LIEVE.value]:
            if post == PostTreatmentEnum.TORNATA_COME_PRIMA.value: cat = ClinicalCategory.CAT_A
            elif post in [PostTreatmentEnum.SUPERIORE.value, PostTreatmentEnum.NON_STABILIZZATA.value]: cat = ClinicalCategory.CAT_D
        elif base == BaselineEnum.MODERATA.value:
            if post == PostTreatmentEnum.TORNATA_COME_PRIMA.value: cat = ClinicalCategory.CAT_B
            elif post in [PostTreatmentEnum.SUPERIORE.value, PostTreatmentEnum.NON_STABILIZZATA.value]: cat = ClinicalCategory.CAT_C
        elif base == BaselineEnum.ELEVATA.value:
            cat = ClinicalCategory.CAT_C
        adj = cat == ClinicalCategory.CAT_E
        return JudgeResult(episode_id=ep_id, assigned_category=cat, adjudication_required=adj, final_confidence_score=strength)
