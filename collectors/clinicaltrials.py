from typing import Optional

from core.search_result import SearchResult


def _lim():
    try:
        from core.llm_limits import get_limits
        return get_limits()
    except Exception:
        from core.llm_limits import HLEOLimits
        return HLEOLimits()


class ClinicalTrialsCollector:
    API_URL = "https://clinicaltrials.gov/api/v2/studies"

    def search_page(
        self,
        query: str,
        cursor: Optional[dict] = None,
        limit: Optional[int] = None,
    ):
        from core.http_retry import http_get
        from core.search_page import SearchPage

        lim = _lim()
        timeout = lim.collector_timeout_s
        max_retries = lim.collector_max_retries
        backoff_base = lim.backoff_base_s
        backoff_max = lim.backoff_max_s

        target = limit if limit is not None else 400
        page_size = max(1, min(target, 100))
        fields = ("NCTId,BriefTitle,BriefSummary,DetailedDescription,"
                  "OverallStatus,Condition,InterventionName,Phase,EnrollmentCount,"
                  "PrimaryOutcomeMeasure,StartDate,PrimaryCompletionDate,LeadSponsorName")
        state = cursor or {"page_token": None, "collected": 0}
        params = {"query.term": query, "pageSize": page_size, "fields": fields}
        if state.get("page_token"):
            params["pageToken"] = state["page_token"]

        r = http_get(
            self.API_URL,
            params=params,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base_s=backoff_base,
            backoff_max_s=backoff_max,
        )
        data = r.json()
        studies = data.get("studies", []) or []
        if limit is not None:
            remaining = max(0, limit - int(state.get("collected", 0)))
            studies = studies[:remaining]

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
                        "nct_id": ident.get("nctId", ""),
                        "condition": cond_mod.get("conditions", []),
                        "status": status_mod.get("overallStatus", ""),
                        "interventions": interventions,
                        "phase": phase_str,
                        "enrollment": enrollment,
                        "primary_outcomes": primary_outcomes,
                        "start_date": start_date,
                        "completion_date": completion_date,
                        "lead_sponsor": lead_sponsor,
                    },
                )
            )

        next_token = data.get("nextPageToken")
        collected = int(state.get("collected", 0)) + len(results)
        has_more = bool(next_token and studies)
        if limit is not None:
            has_more = has_more and collected < limit
        next_cursor = (
            {"page_token": next_token, "collected": collected}
            if has_more else None
        )
        return SearchPage(results, next_cursor, has_more)

    def search(self, query: str, limit: Optional[int] = None):
        from core.search_page import collect_search_pages

        return collect_search_pages(self, query, limit=limit)
