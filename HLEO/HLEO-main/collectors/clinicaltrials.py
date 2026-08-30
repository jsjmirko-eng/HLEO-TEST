import requests
from typing import Optional

from core.search_result import SearchResult


class ClinicalTrialsCollector:
    API_URL = "https://clinicaltrials.gov/api/v2/studies"

    def search(self, query: str, limit: Optional[int] = None):
        target = limit if limit is not None else 400
        page_size = max(1, min(target, 100))
        fields = ("NCTId,BriefTitle,BriefSummary,DetailedDescription,"
                  "OverallStatus,Condition,InterventionName,Phase,EnrollmentCount,"
                  "PrimaryOutcomeMeasure,StartDate,PrimaryCompletionDate,LeadSponsorName")
        studies = []
        token = None
        while len(studies) < target:
            params = {"query.term": query, "pageSize": page_size, "fields": fields}
            if token:
                params["pageToken"] = token
            r = requests.get(self.API_URL, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            batch = data.get("studies", []) or []
            studies.extend(batch)
            if limit is not None or not batch:
                break
            token = data.get("nextPageToken")
            if not token:
                break
        if limit is not None:
            studies = studies[:limit]

        results = []
        for study in studies:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            cond_mod = proto.get("conditionsModule", {})
            desc_mod = proto.get("descriptionModule", {})
            interv_mod = proto.get("armsInterventionsModule", {})
            design_mod = proto.get("designModule", {})
            outcomes_mod = proto.get("outcomesModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

            brief = desc_mod.get("briefSummary", "")
            detailed = desc_mod.get("detailedDescription", "")
            abstract = (brief + "\n\n" + detailed).strip()

            interventions = [
                i.get("interventionName", "")
                for i in interv_mod.get("interventions", [])
            ]

            phases = design_mod.get("phases", [])
            phase_str = ", ".join(phases) if phases else ""

            enrollment = design_mod.get("enrollmentInfo", {}).get("count")

            primary_outcomes = [
                o.get("measure", "")
                for o in outcomes_mod.get("primaryOutcomes", [])
            ]

            start_date = status_mod.get("startDateStruct", {}).get("date", "")
            completion_date = status_mod.get("primaryCompletionDateStruct", {}).get("date", "")

            lead_sponsor = sponsor_mod.get("leadSponsor", {}).get("name", "")

            results.append(
                SearchResult(
                    title=ident.get("briefTitle", ""),
                    source="ClinicalTrials.gov",
                    abstract=abstract,
                    year=int(start_date[:4]) if start_date and start_date[:4].isdigit() else None,
                    metadata={
                        "nct_id":           ident.get("nctId", ""),
                        "condition":        cond_mod.get("conditions", []),
                        "status":           status_mod.get("overallStatus", ""),
                        "interventions":    interventions,
                        "phase":            phase_str,
                        "enrollment":       enrollment,
                        "primary_outcomes": primary_outcomes,
                        "start_date":       start_date,
                        "completion_date":  completion_date,
                        "lead_sponsor":     lead_sponsor,
                    },
                )
            )
        return results
