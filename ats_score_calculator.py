
"""
ats_score_calculator.py

Deterministic scoring + persistence layer for the ATS Evaluation Engine.

Design principle (per spec): the LLM NEVER computes the final ATS score.
It only returns category sub-scores (0-100) in `category_scores_for_ats`.
This module:
  1. Validates the LLM's JSON output against the schema.
  2. Normalizes the JD's ats_weights (so they always sum to 100, regardless
     of how the JD author entered them) so results stay comparable across
     job descriptions with slightly different weight totals.
  3. Computes weighted_final_ats_score = sum(category_score * normalized_weight) / 100.
  4. Cross-checks the LLM's qualitative hiring_recommendation against the
     numeric score band, flagging disagreement for recruiter attention
     rather than silently overriding the LLM.
  5. Produces a flat record ready for PostgreSQL insertion and a
     Streamlit-friendly dict.

Usage:
    from ats_score_calculator import evaluate_candidate

    record = evaluate_candidate(jd_json, llm_output_json)
    # record["weighted_final_ats_score"] -> float
    # record["postgres_row"]            -> flat dict for INSERT
    # record["score_recommendation_alignment"] -> "consistent" | "inconsistent"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import jsonschema  # optional dependency: pip install jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

SCHEMA_PATH = Path(__file__).parent / "ats_evaluation_output_schema.json"

# Used only when the `jsonschema` package isn't installed — a much lighter
# sanity check that the LLM returned all required top-level sections.
REQUIRED_TOP_LEVEL_KEYS = [
    "skill_match", "experience_analysis", "education_analysis", "certification_analysis",
    "project_analysis", "responsibility_alignment", "category_scores_for_ats",
    "risk_analysis", "strength_analysis", "weakness_analysis", "interview_intelligence",
    "hiring_recommendation", "ranking_summary",
]

# Maps category_scores_for_ats keys -> the weight key expected in JD.ats_weights
CATEGORY_TO_WEIGHT_KEY = {
    "skills_score": "skills_weight",
    "experience_score": "experience_weight",
    "education_score": "education_weight",
    "certifications_score": "certification_weight",
    "project_score": "project_weight",
    "responsibilities_score": "responsibility_weight",
    "keyword_score": "keyword_weight",
    "leadership_score": "leadership_weight",
    "communication_score": "communication_weight",
    "domain_score": "domain_weight",
}

# Fallback weights if the JD does not specify ats_weights at all.
# These are illustrative defaults — tune per organization.
DEFAULT_WEIGHTS = {
    "skills_weight": 25,
    "experience_weight": 20,
    "education_weight": 10,
    "certification_weight": 5,
    "project_weight": 10,
    "responsibility_weight": 10,
    "keyword_weight": 5,
    "leadership_weight": 5,
    "communication_weight": 5,
    "domain_weight": 5,
}

RECOMMENDATION_BANDS = [
    (85, 100, "Strong Hire"),
    (70, 84.999, "Hire"),
    (50, 69.999, "Maybe"),
    (0, 49.999, "No Hire"),
]


def load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r") as f:
        return json.load(f)


def validate_llm_output(llm_output: Dict[str, Any]) -> None:
    """
    Validates the LLM output against the JSON schema contract.
    If the `jsonschema` package is installed, performs full schema validation
    (types, enums, ranges, nested required fields). Otherwise falls back to
    checking that all required top-level sections are present.
    """
    if _HAS_JSONSCHEMA:
        schema = load_schema()
        jsonschema.validate(instance=llm_output, schema=schema)
    else:
        missing = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in llm_output]
        if missing:
            raise ValueError(
                f"LLM output is missing required top-level keys: {missing}. "
                "(Install `jsonschema` for full structural validation.)"
            )


def normalize_weights(raw_weights: Dict[str, Any]) -> Dict[str, float]:
    """
    Ensures the 10 category weights sum to exactly 100, regardless of how
    they were entered in the JD JSON (e.g. they might sum to 95, or 100.4
    due to manual entry). Missing individual weights default to 0 before
    normalization. If the JD has no usable weights at all, falls back to
    DEFAULT_WEIGHTS.
    """
    weights = {}
    total = 0.0
    any_present = False

    for cat_key, weight_key in CATEGORY_TO_WEIGHT_KEY.items():
        val = raw_weights.get(weight_key) if raw_weights else None
        if val is not None:
            any_present = True
        val = float(val) if isinstance(val, (int, float)) else 0.0
        weights[weight_key] = val
        total += val

    if not any_present or total <= 0:
        weights = dict(DEFAULT_WEIGHTS)
        total = sum(weights.values())

    return {k: (v / total) * 100.0 for k, v in weights.items()}


def compute_weighted_final_score(
    category_scores: Dict[str, float], normalized_weights: Dict[str, float]
) -> float:
    """
    final_score = sum(category_score * normalized_weight) / 100
    Equivalent to a weighted average since normalized_weights sum to 100.
    """
    total = 0.0
    for cat_key, weight_key in CATEGORY_TO_WEIGHT_KEY.items():
        score = float(category_scores.get(cat_key, 0) or 0)
        weight = normalized_weights.get(weight_key, 0.0)
        total += score * weight
    return round(total / 100.0, 2)


def resolve_communication_score(
    candidate_json: Dict[str, Any] | None, llm_category_scores: Dict[str, Any]
) -> float:
    """
    The prompt's communication_score rule ('use candidate.communication_score
    if provided, else default 50') is a deterministic lookup, not a judgment
    call — so it's enforced here rather than trusted to the LLM every time.

    If `candidate_json` is supplied, its `communication_score` always wins.
    If it's missing/None, default to 50. Only if `candidate_json` itself is
    not passed in at all does the LLM's own communication_score value pass
    through unchanged (e.g. if Python doesn't have access to the raw
    candidate record at scoring time).
    """
    if candidate_json is not None:
        val = candidate_json.get("communication_score")
        return float(val) if isinstance(val, (int, float)) else 50.0
    return float(llm_category_scores.get("communication_score") or 50)


def band_recommendation(score: float) -> str:
    for low, high, label in RECOMMENDATION_BANDS:
        if low <= score <= high:
            return label
    return "No Hire"


def evaluate_candidate(
    job_description: Dict[str, Any],
    llm_output: Dict[str, Any],
    candidate_json: Dict[str, Any] | None = None,
    injected_weights: Dict[str, float] | None = None,
    skip_validation: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point. Combines the LLM's qualitative analysis with the
    deterministic weighted score, and returns:
      - the full evaluation payload (LLM output + computed score)
      - a flat row suitable for a PostgreSQL UPSERT
      - an alignment flag comparing the LLM's stated recommendation against
        the numeric score band (useful as a QA signal, not an override)

    `candidate_json` is optional but recommended: if supplied, its
    `communication_score` deterministically overrides whatever the LLM
    returned for that category (per the prompt's explicit rule), so the
    final weighted score never depends on the LLM correctly following a
    rule that didn't require any judgment in the first place.
    """
    if not skip_validation:
        validate_llm_output(llm_output)

    if injected_weights is None:
        raise ValueError("ATS requires injected_weights from pipeline. No fallback allowed.")

    normalized_weights = normalize_weights(injected_weights)

    category_scores = dict(llm_output["category_scores_for_ats"])
    category_scores["communication_score"] = resolve_communication_score(candidate_json, category_scores)

    final_score = compute_weighted_final_score(category_scores, normalized_weights)

    score_band_recommendation = band_recommendation(final_score)
    llm_recommendation = llm_output["hiring_recommendation"]["recommendation"]
    alignment = "consistent" if score_band_recommendation == llm_recommendation else "inconsistent"

    result = dict(llm_output)  # shallow copy, don't mutate caller's dict
    result["category_scores_for_ats"] = category_scores  # reflects the resolved communication_score
    result["weighted_final_ats_score"] = final_score
    result["score_band_recommendation"] = score_band_recommendation
    result["score_recommendation_alignment"] = alignment
    result["normalized_weights_used"] = normalized_weights

    result["postgres_row"] = build_postgres_row(job_description, result, candidate_json)

    return result


def build_postgres_row(
    job_description: Dict[str, Any], result: Dict[str, Any], candidate_json: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Flattens the nested evaluation into a single-level dict for DB insertion.
    The v2 LLM output no longer carries candidate_name/job_title itself, so
    those come from the orchestration layer's own records (candidate_json,
    job_description) rather than from the LLM response.
    """
    cs = result["category_scores_for_ats"]
    candidate_name = (candidate_json or {}).get("name", "")
    return {
        "candidate_name": candidate_name,
        "job_title": job_description.get("job_title", ""),
        "department": job_description.get("department", ""),
        "matched_skills": json.dumps(result["skill_match"]["matched_skills"]),
        "missing_skills": json.dumps(result["skill_match"]["missing_skills"]),
        "skills_score": cs["skills_score"],
        "experience_score": cs["experience_score"],
        "education_score": cs["education_score"],
        "certifications_score": cs["certification_score"],
        "project_score": cs["project_score"],
        "responsibility_score": cs["responsibility_score"],
        "keyword_score": cs["keyword_score"],
        "leadership_score": cs["leadership_score"],
        "communication_score": cs["communication_score"],
        "domain_score": cs["domain_score"],
        "weighted_final_ats_score": result["weighted_final_ats_score"],
        "llm_recommendation": result["hiring_recommendation"]["recommendation"],
        "score_band_recommendation": result["score_band_recommendation"],
        "recommendation_alignment": result["score_recommendation_alignment"],
        "risk_flags": json.dumps(result["risk_analysis"]["risk_flags"]),
        "top_strengths": json.dumps(result["strength_analysis"]["top_strengths"]),
        "top_weaknesses": json.dumps(result["weakness_analysis"]["top_weaknesses"]),
        "ranking_summary": result.get("ranking_summary", ""),
        "full_evaluation_json": json.dumps(result),
    }


def rank_candidates(evaluated_results: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Sorts already-evaluated candidate records descending by final score for dashboard ranking."""
    return sorted(evaluated_results, key=lambda r: r["weighted_final_ats_score"], reverse=True)


if __name__ == "__main__":
    # Minimal smoke test with illustrative data, matching the v2 strict schema
    jd = {
        "job_title": "Senior Structural Engineer",
        "department": "Infrastructure",
        "weights": {
            "skills_weight": 25, "experience_weight": 20, "education_weight": 10,
            "certification_weight": 5, "project_weight": 10, "responsibility_weight": 10,
            "keyword_weight": 5, "leadership_weight": 5, "communication_weight": 5, "domain_weight": 5,
        },
    }

    candidate = {
        "name": "Jane Doe",
        "skills": ["AutoCAD", "STAAD Pro"],
        "communication_score": 72,  # should override whatever the LLM guessed
    }

    llm_output = {
        "skill_match": {
            "matched_skills": ["AutoCAD", "STAAD Pro"], "missing_skills": ["Revit"],
            "partial_skills": [], "keyword_matches": ["seismic design"],
        },
        "experience_analysis": {
            "required_experience_years": 8, "candidate_experience_years": 9, "experience_gap_years": 1,
            "experience_score": 90, "experience_relevance_score": 85, "job_role_similarity_score": 88,
            "industry_similarity_score": 90,
        },
        "education_analysis": {
            "degree_match": True, "major_match": True,
            "education_score": 95, "education_reasoning": "BSc Civil Engineering matches required degree and major.",
        },
        "certification_analysis": {
            "matched_certifications": ["PE License"], "missing_certifications": [], "certification_score": 90,
        },
        "project_analysis": {
            "project_relevance_score": 88, "project_complexity_score": 85, "project_scale_score": 90,
            "most_relevant_projects": ["Karachi Elevated Corridor"],
        },
        "responsibility_alignment": {
            "responsibility_match_percentage": 82, "matched_responsibilities": ["Structural design review"],
            "missing_responsibilities": [],
        },
        "category_scores_for_ats": {
            "skills_score": 80, "experience_score": 90, "education_score": 95, "certification_score": 90,
            "project_score": 88, "responsibility_score": 82, "keyword_score": 85, "leadership_score": 88,
            "communication_score": 40,  # deliberately wrong — should get overridden to 72 by candidate_json
            "domain_score": 92,
        },
        "risk_analysis": {"risk_flags": []},
        "strength_analysis": {"top_strengths": ["Mega-project experience", "Strong certifications"]},
        "weakness_analysis": {"top_weaknesses": ["No Revit experience"]},
        "interview_intelligence": {"interview_focus_points": ["Validate hands-on Revit exposure"]},
        "hiring_recommendation": {"recommendation": "Strong Hire", "reasoning": "Exceeds requirements with relevant infrastructure experience."},
        "ranking_summary": "Strong overall fit with deep infrastructure experience; minor gap in Revit proficiency.",
    }

    out = evaluate_candidate(jd, llm_output, candidate_json=candidate, injected_weights=jd["weights"],)
    print(json.dumps({
        "communication_score_used": out["category_scores_for_ats"]["communication_score"],
        "weighted_final_ats_score": out["weighted_final_ats_score"],
        "score_band_recommendation": out["score_band_recommendation"],
        "alignment": out["score_recommendation_alignment"],
    }, indent=2))

