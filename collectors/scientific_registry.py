"""Registry of opt-in scientific metadata collectors."""
from collectors.openalex import OpenAlexCollector
from collectors.crossref import CrossrefCollector
from collectors.doaj import DOAJCollector
from collectors.openaire import OpenAIRECollector
from collectors.hal import HALCollector
from collectors.cinii import CiNiiResearchCollector
from collectors.jstage import JStageCollector


NEW_SCIENTIFIC_COLLECTORS = {
    "openalex": OpenAlexCollector,
    "crossref": CrossrefCollector,
    "doaj": DOAJCollector,
    "openaire": OpenAIRECollector,
    "hal": HALCollector,
    "cinii": CiNiiResearchCollector,
    "jstage": JStageCollector,
}
