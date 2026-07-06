"""
ranking.py — Production-Grade ATS Candidate Ranking Engine
===========================================================
Sits on top of the ATS evaluation pipeline as the final recruiter decision layer.

Responsibilities
----------------
1. Fetch pre-evaluated candidates from `ats_evaluations` (PostgreSQL) by JD.
2. Apply deterministic, risk-adjusted ranking (no LLM calls, no score recomputation).
3. Assign recruiter-ready tiers, explainability notes, and structured output.
4. Expose a clean public API consumed by the Streamlit dashboard and CLI.

Integration contracts
---------------------
- Reads ONLY from `ats_evaluations`; never writes to it.
- Never calls Gemini / any LLM.
- Never recomputes `weighted_final_ats_score`.
- All SQL queries use parameterised placeholders (psycopg2-style %s).

Usage (library)
---------------
    from ranking import get_top_candidates, rank_all_candidates, RankingFilter

    top10 = get_top_candidates(jd_id=3, k=10)
    ranked = rank_all_candidates(
        jd_id=3,
        filters=RankingFilter(min_score=50, only_hire_or_better=True),
    )

Usage (CLI)
-----------
    python ranking.py --jd-id 3 --top 10
    python ranking.py --jd-id 3 --min-score 60 --exclude-risk excessive_job_hopping
"""

"""
ranking.py — Production-Grade ATS Candidate Ranking Engine
===========================================================
Sits on top of the ATS evaluation pipeline as the final recruiter decision layer.

Responsibilities
----------------
1. Fetch pre-evaluated candidates from `ats_evaluations` (PostgreSQL) by JD.
2. Apply deterministic, risk-adjusted ranking (no LLM calls, no score recomputation).
3. Assign recruiter-ready tiers, explainability notes, and structured output.
4. Expose a clean public API consumed by the Streamlit dashboard and CLI.

Integration contracts
---------------------
- Reads ONLY from `ats_evaluations`; never writes to it.
- Never calls Gemini / any LLM.
- Never recomputes `weighted_final_ats_score`.
- All SQL queries use parameterised placeholders (psycopg2-style %s).

Usage (library)
---------------
    from ranking import get_top_candidates, rank_all_candidates, RankingFilter

    top10 = get_top_candidates(jd_id=3, k=10)
    ranked = rank_all_candidates(
        jd_id=3,
        filters=RankingFilter(min_score=50, only_hire_or_better=True),
    )

Usage (CLI)
-----------
    python ranking.py --jd-id 3 --top 10
    python ranking.py --jd-id 3 --min-score 60 --exclude-risk excessive_job_hopping
"""


import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import psycopg
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ats_ranking")

DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Risk flag penalty table (deducted from weighted_final_ats_score)
RISK_PENALTIES = {
    "missing_core_skills":      2.0,   
    "insufficient_experience":  8.0,   
    "excessive_job_hopping":    5.0,
    "degree_mismatch":          4.0,
    "major_mismatch":           4.0,
    "certification_gap":        3.0,
    "employment_gap":           1.0,
}

# Tier boundaries (applied to final_rank_score after risk adjustment)
TIER_BANDS: list[tuple[float, float, str]] = [
    (85.0, 100.0, "A+"),
    (70.0,  84.999, "A"),
    (50.0,  69.999, "B"),
    (30.0,  49.999, "C"),
    ( 0.0,  29.999, "D"),
]

# Score-band labels (mirrors ats_score_calculator.RECOMMENDATION_BANDS)
SCORE_BAND_LABELS: list[tuple[float, float, str]] = [
    (85.0, 100.0, "Strong Hire"),
    (70.0,  84.999, "Hire"),
    (50.0,  69.999, "Maybe"),
    ( 0.0,  49.999, "No Hire"),
]

# Columns fetched from ats_evaluations in every query
_EVAL_COLUMNS = [
    "id",
    "resume_id",
    "jd_id",
    "candidate_name",

    # Scores
    "weighted_final_ats_score",   
    "skills_score",
    "experience_score",
    "education_score",
    "certification_score",
    "project_score",
    "responsibility_score",
    "leadership_score",
    "domain_score",

    # Existing JSON columns
    "risk_flags",
    "interview_intelligence",
    "evidence",
    "confidence",
    "raw_output",

    "created_at",
    "missing_skills",
]
# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RankingFilter:
    """
    Optional filters applied before ranking.

    All filters are AND-combined — a candidate must satisfy every
    specified constraint to appear in the ranked output.
    """
    min_score: float = 0.0
    """Minimum raw weighted_final_ats_score (0–100)."""

    min_experience_score: float = 0.0
    """Minimum experience_score (0–100)."""

    exclude_risk_flags: list[str] = field(default_factory=list)
    """
    Drop any candidate that carries at least one of these risk flags.
    Example: ["excessive_job_hopping", "degree_mismatch"]
    """

    only_hire_or_better: bool = False
    """
    If True, only candidates with score_band_recommendation in
    {"Hire", "Strong Hire"} are returned.
    """


@dataclass
class RankedCandidate:
    """Single recruiter-ready ranked candidate record."""
    rank: int
    resume_id: int
    eval_id: int
    candidate_name: str
    raw_ats_score: float
    risk_penalty: float
    final_rank_score: float
    score_band: str
    tier: str
    risk_flags: list[str]
    top_strengths: list[str]
    top_weaknesses: list[str]
    ranking_summary: str
    why_ranked_here: list[str]
    category_scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank":              self.rank,
            "resume_id":         self.resume_id,
            "eval_id":           self.eval_id,
            "candidate_name":    self.candidate_name,
            "raw_ats_score":     self.raw_ats_score,
            "risk_penalty":      self.risk_penalty,
            "final_rank_score":  self.final_rank_score,
            "score_band":        self.score_band,
            "tier":              self.tier,
            "risk_flags":        self.risk_flags,
            "top_strengths":     self.top_strengths,
            "top_weaknesses":    self.top_weaknesses,
            "ranking_summary":   self.ranking_summary,
            "why_ranked_here":   self.why_ranked_here,
            "category_scores":   self.category_scores,
        }

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_connection() -> psycopg.Connection:
    """Open and return a fresh psycopg connection."""
    return psycopg.connect(
        dbname="resume_db",
        user="postgres",
        password=DB_PASSWORD,
        host="localhost",
        port="5432",
    )


def _row_to_dict(row: tuple, columns: list[str]) -> dict[str, Any]:
    """Zip a DB row tuple into a plain dict."""
    return dict(zip(columns, row))


def _parse_jsonb(value: Any) -> list:
    """
    Safely parse a JSONB column that psycopg may return as a list,
    dict, or JSON string depending on the driver version.
    """
    if value is None:
        return []
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []


def _iter_evaluated_candidates(
    conn: psycopg.Connection,
    jd_id: int,
    page_size: int = 500,
) -> Iterator[dict[str, Any]]:
    """
    Stream all ats_evaluations rows for a given JD in paginated batches.
    Uses keyset (cursor) pagination on the auto-increment `id` column to
    avoid OFFSET performance degradation on large tables.

    Yields one dict per candidate row.
    """
    col_list = ", ".join(_EVAL_COLUMNS)
    last_id = 0

    while True:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                e.id,
                e.resume_id,
                e.jd_id,
                r.name AS candidate_name,
                e.weighted_final_ats_score,
                e.skills_score,
                e.experience_score,
                e.education_score,
                e.certification_score,
                e.project_score,
                e.responsibility_score,
                e.leadership_score,
                e.domain_score,

                e.risk_flags,

                e.created_at

            FROM ats_evaluations e
            JOIN resumes r
            ON r.id = e.resume_id

            WHERE e.jd_id = %s
            AND e.id > %s

            ORDER BY e.id ASC

            LIMIT %s
                """,
                (jd_id, last_id, page_size),
            )
            rows = cur.fetchall()

        if not rows:
            break

        for row in rows:
            yield _row_to_dict(row, _EVAL_COLUMNS)

        last_id = rows[-1][0]  # `id` is the first column

        if len(rows) < page_size:
            break  # last page


def _deduplicate_by_resume(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    If a resume was evaluated more than once against the same JD
    (e.g. forced re-run), keep only the most recent evaluation
    (highest `id` = latest INSERT).
    """
    seen: dict[int, dict[str, Any]] = {}
    for row in rows:
        rid = row["resume_id"]
        if rid not in seen or row["id"] > seen[rid]["id"]:
            seen[rid] = row
    return list(seen.values())

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _compute_risk_penalty(row):
    flags = _parse_jsonb(row.get("risk_flags"))
    penalty = 0.0

    # Dynamic penalty
    if "missing_core_skills" in flags:
        missing = _parse_jsonb(row.get("missing_skills"))
        penalty += min(len(missing), 5)

    # Fixed penalties
    for flag in flags:
        if flag != "missing_core_skills":
            penalty += RISK_PENALTIES.get(flag, 0)

    return penalty


def _compute_final_rank_score(raw_score: float, penalty: float) -> float:
    """Clamp (raw_score - penalty) to [0, 100]."""
    return max(0.0, min(100.0, raw_score - penalty))


def _resolve_tier(score: float) -> str:
    for low, high, tier in TIER_BANDS:
        if low <= score <= high:
            return tier
    return "D"


def _resolve_score_band(score: float) -> str:
    """Derive score band from numeric score (deterministic, not from DB string)."""
    for low, high, label in SCORE_BAND_LABELS:
        if low <= score <= high:
            return label
    return "No Hire"


def _tiebreaker_key(row: dict[str, Any]) -> tuple:
    """
    Secondary sort key used when two candidates share the same
    final_rank_score. Order: skills → experience → project →
    leadership → communication (all descending — negate for sort asc).
    """
    def s(col: str) -> float:
        return float(row.get(col) or 0)

    return (
        s("skills_score"),
        s("experience_score"),
        s("project_score"),
        s("leadership_score"),
        s("domain_score"),
    )

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def _build_why_ranked_here(
    row: dict[str, Any],
    final_rank_score: float,
    risk_penalty: float,
    rank: int,
) -> list[str]:
    """
    Generate 2–5 plain-English bullets explaining why this candidate
    occupies their rank position. Entirely deterministic — no LLM.
    """
    reasons: list[str] = []
#     raw = (
#     float(row.get("skills_score") or 0) * 0.20 +
#     float(row.get("experience_score") or 0) * 0.20 +
#     float(row.get("education_score") or 0) * 0.10 +
#     float(row.get("certification_score") or 0) * 0.10 +
#     float(row.get("project_score") or 0) * 0.15 +
#     float(row.get("responsibility_score") or 0) * 0.10 +
#     float(row.get("leadership_score") or 0) * 0.10 +
#     float(row.get("domain_score") or 0) * 0.05
# )

    # ── Score context ───────────────────────────────────────────────────────
    tier = _resolve_tier(final_rank_score)
    reasons.append(
        f"Ranked #{rank} with a final score of {final_rank_score:.1f}/100 "
        f"(Tier {tier}, {_resolve_score_band(final_rank_score)})"
    )

    # ── Risk penalty commentary ─────────────────────────────────────────────
    if risk_penalty > 0:
        flags_str = ", ".join(row.get("risk_flags") or [])
        reasons.append(
            f"Risk penalty of -{risk_penalty:.0f} pts applied "
            f"(flags: {flags_str})"
        )
    else:
        reasons.append("No risk penalties applied — clean risk profile")

    # ── Standout category scores ────────────────────────────────────────────
    category_labels = {
    "skills_score": "Skills",
    "experience_score": "Experience",
    "education_score": "Education",
    "project_score": "Projects",
    "leadership_score": "Leadership",
    "domain_score": "Domain expertise",
    }
    strong: list[str] = []
    weak:   list[str] = []

    for col, label in category_labels.items():
        val = float(row.get(col) or 0)
        if val >= 80:
            strong.append(f"{label} ({val:.0f})")
        elif val < 50:
            weak.append(f"{label} ({val:.0f})")

    if strong:
        reasons.append(f"Strong category scores: {', '.join(strong)}")
    if weak:
        reasons.append(f"Below-average category scores: {', '.join(weak)}")

    # ── Recommendation alignment note ───────────────────────────────────────
    alignment = row.get("recommendation_alignment", "")
    if alignment == "inconsistent":
        llm_rec = row.get("llm_recommendation", "")
        band    = row.get("score_band_recommendation", "")
        reasons.append(
            f"⚠ LLM recommendation ({llm_rec}) differs from score band "
            f"({band}) — manual review advised"
        )

    return reasons

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

_HIRE_OR_BETTER: frozenset[str] = frozenset({"Hire", "Strong Hire"})


def _passes_filter(row: dict[str, Any], f: RankingFilter) -> bool:
    """Return True if the candidate row satisfies all active filter criteria."""
    raw = float(row.get("weighted_final_ats_score") or 0)
    if raw < f.min_score:
        return False

    exp = float(row.get("experience_score") or 0)
    if exp < f.min_experience_score:
        return False

    risk_flags: list[str] = _parse_jsonb(row.get("risk_flags"))
    if f.exclude_risk_flags:
        if any(flag in f.exclude_risk_flags for flag in risk_flags):
            return False

    if f.only_hire_or_better:
        band = row.get("score_band_recommendation") or ""
        if band not in _HIRE_OR_BETTER:
            return False

    return True

# ---------------------------------------------------------------------------
# Core ranking engine
# ---------------------------------------------------------------------------

def _rank_rows(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], float, float]]:
    """
    Compute (risk_penalty, final_rank_score) for each row and sort descending.

    Returns a list of (row, risk_penalty, final_rank_score) tuples,
    ordered by (final_rank_score DESC, tiebreakers DESC).
    """
    scored: list[tuple[dict[str, Any], float, float]] = []
    for row in rows:
        raw     = float(row.get("weighted_final_ats_score") or 0)
        flags   = _parse_jsonb(row.get("risk_flags"))
        penalty = _compute_risk_penalty(row)
        final   = _compute_final_rank_score(raw, penalty)
        scored.append((row, penalty, final))

    scored.sort(
        key=lambda t: (t[2], _tiebreaker_key(t[0])),
        reverse=True,
    )
    return scored


def _build_ranked_candidate(
    rank: int,
    row: dict[str, Any],
    risk_penalty: float,
    final_rank_score: float,
) -> RankedCandidate:
    """Assemble a RankedCandidate from a raw DB row + computed scores."""
    risk_flags   = _parse_jsonb(row.get("risk_flags"))
    top_strengths  = _parse_jsonb(row.get("top_strengths"))
    top_weaknesses = _parse_jsonb(row.get("top_weaknesses"))

    raw = float(row.get("weighted_final_ats_score") or 0)

    category_scores = {
        "skills_score": float(row.get("skills_score") or 0),
        "experience_score": float(row.get("experience_score") or 0),
        "education_score": float(row.get("education_score") or 0),
        "certification_score": float(row.get("certification_score") or 0),
        "project_score": float(row.get("project_score") or 0),
        "responsibility_score": float(row.get("responsibility_score") or 0),
        "leadership_score": float(row.get("leadership_score") or 0),
        "domain_score": float(row.get("domain_score") or 0),
    }

    return RankedCandidate(
        rank             = rank,
        resume_id        = int(row.get("resume_id")   or 0),
        eval_id          = int(row.get("id")          or 0),
        candidate_name   = row.get("candidate_name")  or "",
        raw_ats_score    = raw,
        risk_penalty     = risk_penalty,
        final_rank_score = final_rank_score,
        score_band       = _resolve_score_band(final_rank_score),
        tier             = _resolve_tier(final_rank_score),
        risk_flags       = risk_flags,
        top_strengths    = top_strengths,
        top_weaknesses   = top_weaknesses,
        ranking_summary  = row.get("ranking_summary") or "",
        why_ranked_here  = _build_why_ranked_here(row, final_rank_score, risk_penalty, rank),
        category_scores  = category_scores,
    )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_all_candidates(
    jd_id: int,
    filters: Optional[RankingFilter] = None,
    page_size: int = 500,
    conn: Optional[psycopg.Connection] = None,
) -> list[RankedCandidate]:
    """
    Fetch, filter, and rank ALL evaluated candidates for a given JD.

    Parameters
    ----------
    jd_id:      Job description primary key.
    filters:    Optional RankingFilter instance. Pass None to rank everyone.
    page_size:  DB pagination batch size (tune for memory vs query overhead).
    conn:       Optional existing psycopg connection (caller owns lifecycle).

    Returns
    -------
    List of RankedCandidate objects ordered rank #1 → #N.
    """
    if filters is None:
        filters = RankingFilter()

    owns_conn = conn is None
    if owns_conn:
        conn = _get_connection()

    try:
        logger.info("Fetching evaluations for JD %d …", jd_id)

        raw_rows: list[dict[str, Any]] = list(
            _iter_evaluated_candidates(conn, jd_id, page_size=page_size)
        )

        logger.info("Fetched %d rows; deduplicating …", len(raw_rows))
        raw_rows = _deduplicate_by_resume(raw_rows)
        logger.info("%d unique candidates after deduplication.", len(raw_rows))

        # Apply pre-rank filters
        filtered = [r for r in raw_rows if _passes_filter(r, filters)]
        logger.info("%d candidates pass filters.", len(filtered))

        if not filtered:
            return []

        # Sort by risk-adjusted score + tiebreakers
        scored = _rank_rows(filtered)

        ranked: list[RankedCandidate] = []
        for rank, (row, penalty, final) in enumerate(scored, start=1):
            ranked.append(_build_ranked_candidate(rank, row, penalty, final))

        logger.info("Ranking complete: %d candidates ranked for JD %d.", len(ranked), jd_id)
        return ranked

    finally:
        if owns_conn:
            conn.close()


def get_top_candidates(
    jd_id: int,
    k: int = 10,
    filters: Optional[RankingFilter] = None,
    conn: Optional[psycopg.Connection] = None,
) -> list[RankedCandidate]:
    """
    Convenience wrapper — return the top-k ranked candidates.

    Equivalent to rank_all_candidates(…)[:k].

    Parameters
    ----------
    jd_id:    Job description primary key.
    k:        Number of candidates to return (default 10).
    filters:  Optional pre-rank filter criteria.
    conn:     Optional open DB connection.
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")

    all_ranked = rank_all_candidates(jd_id=jd_id, filters=filters, conn=conn)
    return all_ranked[:k]


def get_candidate_rank(
    jd_id: int,
    resume_id: int,
    conn: Optional[psycopg.Connection] = None,
) -> Optional[RankedCandidate]:
    """
    Return the ranking record for a single candidate within a JD pool.
    Returns None if the candidate hasn't been evaluated for this JD.

    Useful for "where does this resume stand?" lookups from the UI.
    """
    all_ranked = rank_all_candidates(jd_id=jd_id, conn=conn)
    for candidate in all_ranked:
        if candidate.resume_id == resume_id:
            return candidate
    return None


def get_tier_summary(ranked: list[RankedCandidate]) -> dict[str, int]:
    """
    Return a count of candidates per tier.

    Example output: {"A+": 2, "A": 8, "B": 15, "C": 4, "D": 1}
    """
    summary: dict[str, int] = {"A+": 0, "A": 0, "B": 0, "C": 0, "D": 0}
    for c in ranked:
        summary[c.tier] = summary.get(c.tier, 0) + 1
    return summary


def get_score_band_summary(ranked: list[RankedCandidate]) -> dict[str, int]:
    """
    Return a count of candidates per score band (Strong Hire / Hire / Maybe / No Hire).
    """
    summary: dict[str, int] = {
        "Strong Hire": 0, "Hire": 0, "Maybe": 0, "No Hire": 0
    }
    for c in ranked:
        summary[c.score_band] = summary.get(c.score_band, 0) + 1
    return summary


def export_ranked_to_dicts(ranked: list[RankedCandidate]) -> list[dict[str, Any]]:
    """Serialize a ranked list to plain dicts (e.g. for JSON export or Streamlit)."""
    return [c.to_dict() for c in ranked]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="ATS Candidate Ranking Engine",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--jd-id", type=int, required=True,
        help="Job Description ID to rank candidates for.",
    )
    parser.add_argument(
        "--top", type=int, default=None, metavar="K",
        help="Return only the top-K candidates (default: all).",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Minimum raw ATS score (0–100) to include. Default: 0.",
    )
    parser.add_argument(
        "--min-exp-score", type=float, default=0.0,
        help="Minimum experience_score (0–100) to include. Default: 0.",
    )
    parser.add_argument(
        "--exclude-risk", nargs="*", default=[],
        metavar="FLAG",
        help=(
            "Exclude candidates carrying any of these risk flags.\n"
            f"Valid flags: {', '.join(sorted(RISK_PENALTIES))}"
        ),
    )
    parser.add_argument(
        "--hire-or-better", action="store_true",
        help="Only include Hire / Strong Hire candidates.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print tier + band summary counts instead of full list.",
    )
    parser.add_argument(
        "--page-size", type=int, default=500,
        help="DB pagination batch size. Default: 500.",
    )

    args = parser.parse_args()

    filters = RankingFilter(
        min_score            = args.min_score,
        min_experience_score = args.min_exp_score,
        exclude_risk_flags   = args.exclude_risk or [],
        only_hire_or_better  = args.hire_or_better,
    )

    logger.info(
        "Ranking candidates for JD %d  |  top=%s  filters=%s",
        args.jd_id, args.top or "all", filters,
    )

    if args.top:
        ranked = get_top_candidates(
            jd_id=args.jd_id, k=args.top, filters=filters,
        )
    else:
        ranked = rank_all_candidates(
            jd_id=args.jd_id, filters=filters, page_size=args.page_size,
        )

    if args.summary:
        print("\n=== Tier Summary ===")
        for tier, count in get_tier_summary(ranked).items():
            print(f"  Tier {tier}: {count}")
        print("\n=== Score Band Summary ===")
        for band, count in get_score_band_summary(ranked).items():
            print(f"  {band}: {count}")
        print(f"\nTotal ranked: {len(ranked)}")
        return

    output = export_ranked_to_dicts(ranked)
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    _cli()
