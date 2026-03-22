from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List

SEPARATOR = "!@#$%^&*()"

OPERATION_TO_ID = {
    "definition": 0,
    "evidence_status": 1,
    "causes_effects": 2,
    "actions_solutions": 3,
    "comparison_evaluation": 4,
}
ID_TO_OPERATION = {value: key for key, value in OPERATION_TO_ID.items()}
UNKNOWN_OPERATION_ID = -100

_LOCATION_TERMS = {
    "africa", "asia", "europe", "america", "us", "u.s.", "usa", "united states",
    "china", "india", "canada", "uk", "united kingdom", "australia", "japan",
    "germany", "france", "russia", "brazil", "mexico"
}
_POPULATION_TERMS = {
    "children", "child", "adults", "adult", "women", "woman", "men", "man",
    "adolescents", "adolescent", "teenagers", "elderly", "older adults", "patients",
    "students", "teachers", "pregnant women", "infants"
}
_COMPARISON_PATTERNS = [
    r"\bcompare\b", r"\bcomparison\b", r"\bpros and cons\b", r"\badvantages and disadvantages\b",
    r"\bdifference(?:s)? between\b", r"\bhow do .* differ\b", r"\bversus\b", r"\bvs\.?\b", r"\bbetter than\b",
]
_ACTION_PATTERNS = [
    r"\btreatment(?:s)? for\b", r"\bhow to\b", r"\bprevent(?:ion)?\b", r"\bmanage(?:ment)?\b",
    r"\bintervention(?:s)?\b", r"\bsolution(?:s)?\b", r"\bwhat can be done\b", r"\breduce\b",
]
_CAUSE_PATTERNS = [
    r"\bcause(?:s)? of\b", r"\beffect(?:s)? of\b", r"\bimpact of\b", r"\bmechanism(?:s)? of\b",
    r"\bwhy does\b", r"\bwhy do\b", r"\bhow .* affect(?:s)?\b",
]
_EVIDENCE_PATTERNS = [
    r"\bevidence for\b", r"\bfact(?:s)? vs\.? fiction\b", r"\bmyth(?:s)? and fact(?:s)?\b",
    r"\bis it true\b", r"\bscientific consensus\b", r"\bscientifically proven\b", r"\bdoes .* really\b",
]
_TEMPORAL_PATTERNS = [
    r"\bsince \d{4}\b", r"\bafter \d{4}\b", r"\bbefore \d{4}\b", r"\bin the \d{4}s\b",
    r"\brecent\b", r"\bcurrent\b", r"\bhistorical\b", r"\blatest\b", r"\bpost[- ]pandemic\b",
]


@dataclass
class QueryInstruction:
    topic_text: str
    operation: str
    operation_id: int
    constraints: Dict[str, List[str] | bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _normalize_query_text(query: str, separator: str = SEPARATOR) -> str:
    text = query.split(separator, 1)[-1] if separator in query else query
    if ";" in text:
        text = text.split(";", 1)[-1]
    return re.sub(r"\s+", " ", text).strip()


def _extract_topic(text: str) -> str:
    lowered = text.lower().strip(" ?.")
    prefixes = [
        r"^what is ", r"^what are ", r"^explain ", r"^compare ", r"^difference between ",
        r"^pros and cons of ", r"^causes? of ", r"^effects? of ", r"^impact of ",
        r"^treatment for ", r"^how to ", r"^evidence for ",
    ]
    topic = lowered
    for prefix in prefixes:
        topic = re.sub(prefix, "", topic)
    topic = re.sub(r"\bin (?:the )?[a-z ]+$", "", topic)
    topic = re.sub(r"\bsince \d{4}.*$", "", topic)
    topic = re.sub(r"\bafter \d{4}.*$", "", topic)
    return topic.strip(" ,") or lowered


def _detect_operation(text: str) -> str:
    lowered = text.lower()
    if any(re.search(p, lowered) for p in _COMPARISON_PATTERNS):
        return "comparison_evaluation"
    if any(re.search(p, lowered) for p in _ACTION_PATTERNS):
        return "actions_solutions"
    if any(re.search(p, lowered) for p in _CAUSE_PATTERNS):
        return "causes_effects"
    if any(re.search(p, lowered) for p in _EVIDENCE_PATTERNS):
        return "evidence_status"
    return "definition"


def _extract_constraints(text: str) -> Dict[str, List[str] | bool]:
    lowered = text.lower()
    geo = sorted(term for term in _LOCATION_TERMS if term in lowered)
    population = sorted(term for term in _POPULATION_TERMS if term in lowered)
    temporal = sorted({match.group(0) for pattern in _TEMPORAL_PATTERNS for match in re.finditer(pattern, lowered)})
    return {
        "geographic": geo,
        "temporal": temporal,
        "population_entity": population,
    }


def label_query_instruction(query: str, separator: str = SEPARATOR) -> QueryInstruction:
    normalized = _normalize_query_text(query, separator=separator)
    operation = _detect_operation(normalized)
    return QueryInstruction(
        topic_text=_extract_topic(normalized),
        operation=operation,
        operation_id=OPERATION_TO_ID.get(operation, UNKNOWN_OPERATION_ID),
        constraints=_extract_constraints(normalized),
    )


def has_instruction_conflict(query_label: QueryInstruction, document_label: QueryInstruction) -> bool:
    if query_label.operation != document_label.operation:
        return True

    for key in ("geographic", "temporal", "population_entity"):
        query_values = set(query_label.constraints.get(key, []))
        doc_values = set(document_label.constraints.get(key, []))
        if query_values and doc_values and query_values.isdisjoint(doc_values):
            return True

    return False
