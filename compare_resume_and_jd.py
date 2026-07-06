

from __future__ import annotations

import os
import json
import time
import random
import hashlib
import logging
from typing import Optional

import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

# file3 — deterministic scoring layer (single source of truth for final score)
from ats_score_calculator import evaluate_candidate

# =========================
# SETUP
# =========================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("ats_comparator")

API_KEY    = os.getenv("GEMINI_API_KEY")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY")
if not DB_PASSWORD:
    raise ValueError("Missing DB_PASSWORD")

gemini_client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"


# =========================
# SYNONYM NORMALIZATION
# =========================
SYNONYM_MAP: dict[str, str] = {
    "postgres":   "postgresql",
    "mongo":      "mongodb",
    "k8s":        "kubernetes",
    "tf":         "terraform",
    "gpt":        "openai",
    "react.js":   "react",
    "vue.js":     "vue",
    "node.js":    "nodejs",
    "js":         "javascript",
    "ts":         "typescript",
    "ml":         "machine learning",
    "dl":         "deep learning",
    "nlp":        "natural language processing",
    "cv":         "computer vision",
    "aws":        "amazon web services",
    "gcp":        "google cloud platform",
    "az":         "azure",
    "ci/cd":      "devops",
    "rest":       "rest api",
    "restful":    "rest api",
}

def normalize_term(term: str) -> str:
    t = term.lower().strip()
    return SYNONYM_MAP.get(t, t)

def normalize_list(items: list) -> list[str]:
    seen:   set[str]  = set()
    result: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        n = normalize_term(item)
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    return result


# =========================
# DB HELPERS
# =========================
def get_db_connection() -> psycopg.Connection:
    conn = psycopg.connect(
        dbname="resume_db",
        user="postgres",
        password=DB_PASSWORD,
        host="localhost",
        port="5432"
    )
    logger.debug("DB connected: %s", conn.info.dbname)
    return conn


def ensure_ats_table(conn: psycopg.Connection) -> None:
    """
    Create (or migrate) ats_evaluations so its columns match the flat
    postgres_row dict that ats_score_calculator.build_postgres_row() returns,
    plus the pipeline's own resume_id / jd_id / comparison_hash / created_at.
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ats_evaluations (
                id                          SERIAL PRIMARY KEY,
                resume_id                   INTEGER NOT NULL,
                jd_id                       INTEGER NOT NULL,
                comparison_hash             TEXT UNIQUE NOT NULL,

                -- identity (from orchestration layer, not LLM)
                candidate_name              TEXT,
                job_title                   TEXT,
                department                  TEXT,

                -- skill evidence
                matched_skills              JSONB,
                missing_skills              JSONB,

                -- category scores (v2 naming — singular, matching schema.json)
                skills_score                NUMERIC(5,2),
                experience_score            NUMERIC(5,2),
                education_score             NUMERIC(5,2),
                certification_score         NUMERIC(5,2),
                project_score               NUMERIC(5,2),
                responsibility_score        NUMERIC(5,2),
                keyword_score               NUMERIC(5,2),
                leadership_score            NUMERIC(5,2),
                communication_score         NUMERIC(5,2),
                domain_score                NUMERIC(5,2),

                -- final computed score (Python only, never LLM)
                weighted_final_ats_score    NUMERIC(5,2),

                -- recommendation
                llm_recommendation          TEXT,
                score_band_recommendation   TEXT,
                recommendation_alignment    TEXT,

                -- qualitative
                risk_flags                  JSONB,
                top_strengths               JSONB,
                top_weaknesses              JSONB,
                ranking_summary             TEXT,

                -- full payload for audit / debugging
                full_evaluation_json        JSONB,

                created_at                  TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_ats_resume_jd
                ON ats_evaluations (resume_id, jd_id);
        """)
    conn.commit()


def fetch_resume(conn: psycopg.Connection, resume_id: int) -> Optional[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                id, name, degree, major, cgpa, university, graduation_year,
                experience_years, skills, certifications, leadership, languages,
                consultancy_companies, mega_projects, raw_data,
                communication_score
            FROM resumes
            WHERE id = %s
        """, (resume_id,))
        row = cur.fetchone()
    if not row:
        return None
    cols = [
        "id", "name", "degree", "major", "cgpa", "university",
        "graduation_year", "experience_years", "skills", "certifications",
        "leadership", "languages", "consultancy_companies", "mega_projects",
        "raw_data", "communication_score",
    ]
    return dict(zip(cols, row))

# FROM job_descriptions
#             WHERE id = %s
def fetch_job_description(conn: psycopg.Connection, jd_id: int) -> Optional[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                id, job_title, department, seniority, required_degree,
                required_major, required_experience_years, required_skills,
                preferred_skills, required_certifications, preferred_certifications,
                required_languages, required_projects, required_keywords,
                responsibilities, weights,
                consultancy_experience_required,
                mega_project_experience_required,
                donor_project_experience_required
            FROM job_descriptions
            WHERE id = %s
            LIMIT 1
        """, (jd_id,))

        row = cur.fetchone()
    if not row:
        return None
    cols = [
        "id", "job_title", "department", "seniority", "required_degree",
        "required_major", "required_experience_years", "required_skills",
        "preferred_skills", "required_certifications", "preferred_certifications",
        "required_languages", "required_projects", "required_keywords",
        "responsibilities", "weights",
        "consultancy_experience_required",
        "mega_project_experience_required",
        "donor_project_experience_required",
    ]
    return dict(zip(cols, row))


def fetch_all_resume_ids(conn: psycopg.Connection, page: int, page_size: int) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM resumes ORDER BY id LIMIT %s OFFSET %s",
            (page_size, page * page_size)
        )
        return [row[0] for row in cur.fetchall()]


def get_cached_evaluation(conn: psycopg.Connection, comparison_hash: str) -> Optional[dict]:
    """Return the stored full_evaluation_json if this pair was already evaluated."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT full_evaluation_json
            FROM ats_evaluations
            WHERE comparison_hash = %s
        """, (comparison_hash,))
        row = cur.fetchone()
    return row[0] if row else None


def save_evaluation(
    conn:             psycopg.Connection,
    resume_id:        int,
    jd_id:            int,
    comparison_hash:  str,
    postgres_row:     dict,
) -> None:
    """
    Insert the flat postgres_row produced by ats_score_calculator plus the
    pipeline's own FK columns.  Uses ON CONFLICT DO NOTHING so a cached
    re-run is a no-op.
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ats_evaluations (
                resume_id, jd_id, comparison_hash,
                candidate_name, job_title, department,
                matched_skills, missing_skills,
                skills_score, experience_score, education_score,
                certification_score, project_score, responsibility_score,
                keyword_score, leadership_score, communication_score, domain_score,
                weighted_final_ats_score,
                llm_recommendation, score_band_recommendation, recommendation_alignment,
                risk_flags, top_strengths, top_weaknesses,
                ranking_summary, full_evaluation_json
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s
            )
            ON CONFLICT (comparison_hash) DO NOTHING;
        """, (
            resume_id,
            jd_id,
            comparison_hash,
            postgres_row.get("candidate_name"),
            postgres_row.get("job_title"),
            postgres_row.get("department"),
            postgres_row.get("matched_skills"),         # already JSON string from file3
            postgres_row.get("missing_skills"),
            postgres_row.get("skills_score"),
            postgres_row.get("experience_score"),
            postgres_row.get("education_score"),
            postgres_row.get("certification_score"),
            postgres_row.get("project_score"),
            postgres_row.get("responsibility_score"),
            postgres_row.get("keyword_score"),
            postgres_row.get("leadership_score"),
            postgres_row.get("communication_score"),
            postgres_row.get("domain_score"),
            postgres_row.get("weighted_final_ats_score"),
            postgres_row.get("llm_recommendation"),
            postgres_row.get("score_band_recommendation"),
            postgres_row.get("recommendation_alignment"),
            postgres_row.get("risk_flags"),             # already JSON string from file3
            postgres_row.get("top_strengths"),
            postgres_row.get("top_weaknesses"),
            postgres_row.get("ranking_summary"),
            postgres_row.get("full_evaluation_json"),
        ))
    conn.commit()
    logger.info("Saved evaluation: resume=%s jd=%s  score=%.1f",
                resume_id, jd_id,
                postgres_row.get("weighted_final_ats_score") or 0)


# =========================
# PREPROCESSING
# =========================
def safe_json_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def truncate_text(text: str, max_chars: int = 800) -> str:
    return text[:max_chars] if text else ""


def preprocess_resume(raw: dict) -> dict:
    raw_data = raw.get("raw_data") or {}
    if isinstance(raw_data, str):
        try:
            raw_data = json.loads(raw_data)
        except (json.JSONDecodeError, TypeError):
            raw_data = {}

    experience_entries = []
    for exp in (raw_data.get("experience") or [])[:6]:
        if isinstance(exp, dict):
            experience_entries.append({
                "company":     truncate_text(str(exp.get("company", "")),     100),
                "role":        truncate_text(str(exp.get("role", "")),        100),
                "duration":    exp.get("duration", ""),
                "description": truncate_text(str(exp.get("description", "")), 200),
            })

    projects = []
    for p in (raw_data.get("projects") or [])[:8]:
        if isinstance(p, dict):
            projects.append({
                "name":        truncate_text(str(p.get("name", "")),        100),
                "description": truncate_text(str(p.get("description", "")), 200),
            })

    return {
        "id":                   raw.get("id"),
        "name":                 raw.get("name") or "",
        "degree":               raw.get("degree") or "",
        "major":                raw.get("major") or "",
        "cgpa":                 raw.get("cgpa") or "",
        "university":           raw.get("university") or "",
        "graduation_year":      raw.get("graduation_year") or "",
        "experience_years":     raw.get("experience_years") or 0,
        "communication_score":  raw.get("communication_score"),   # passed to file3
        "skills":               normalize_list(safe_json_list(raw.get("skills")))[:40],
        "certifications":       normalize_list(safe_json_list(raw.get("certifications")))[:20],
        "leadership":           safe_json_list(raw.get("leadership"))[:10],
        "languages":            normalize_list(safe_json_list(raw.get("languages")))[:10],
        "consultancy_companies":safe_json_list(raw.get("consultancy_companies"))[:10],
        "mega_projects":        safe_json_list(raw.get("mega_projects"))[:8],
        "experience":           experience_entries,
        "projects":             projects,
    }


def preprocess_jd(raw: dict) -> dict:
    return {
        "id":                       raw.get("id"),
        "job_title":                raw.get("job_title") or "",
        "department":               raw.get("department") or "",
        "seniority":                raw.get("seniority") or "",
        "required_degree":          raw.get("required_degree") or "",
        "required_major":           normalize_list(safe_json_list(raw.get("required_major")))[:5],
        "required_experience_years":raw.get("required_experience_years") or 0,
        "required_skills":          normalize_list(safe_json_list(raw.get("required_skills")))[:40],
        "preferred_skills":         normalize_list(safe_json_list(raw.get("preferred_skills")))[:20],
        "required_certifications":  normalize_list(safe_json_list(raw.get("required_certifications")))[:15],
        "preferred_certifications": normalize_list(safe_json_list(raw.get("preferred_certifications")))[:10],
        "required_languages":       normalize_list(safe_json_list(raw.get("required_languages")))[:10],
        "required_projects":        safe_json_list(raw.get("required_projects"))[:8],
        "required_keywords":        normalize_list(safe_json_list(raw.get("required_keywords")))[:20],
        "responsibilities":         safe_json_list(raw.get("responsibilities"))[:15],
        # weights key used by ats_score_calculator.normalize_weights()
        "weights":                  raw.get("weights") or {},
        "consultancy_required":     raw.get("consultancy_experience_required") or False,
        "mega_project_required":    raw.get("mega_project_experience_required") or False,
        "donor_project_required":   raw.get("donor_project_experience_required") or False,
    }


# =========================
# COMPARISON HASH
# =========================
def make_comparison_hash(resume_id: int, jd_id: int) -> str:
    return hashlib.sha256(f"resume:{resume_id}::jd:{jd_id}".encode()).hexdigest()


# =========================
# GEMINI PROMPT  (v2 schema)
# =========================
# This is the exact structure the LLM must return — it mirrors the v2 JSON schema
# so validate_llm_output() (from ats_score_calculator) accepts it without error.
# category_scores_for_ats contains ONLY sub-scores; final ATS score is never
# computed by the LLM — that is done deterministically in ats_score_calculator.
V2_OUTPUT_SCHEMA = {
    "skill_match": {
        "matched_skills":  [],
        "missing_skills":  [],
        "partial_skills":  [],
        "keyword_matches": []
    },
    "experience_analysis": {
        "required_experience_years":    0,
        "candidate_experience_years":   0,
        "experience_gap_years":         0,
        "experience_score":             0,
        "experience_relevance_score":   0,
        "job_role_similarity_score":    0,
        "industry_similarity_score":    0
    },
    "education_analysis": {
        "degree_match":         False,
        "major_match":          False,
        "education_score":      0,
        "education_reasoning":  ""
    },
    "certification_analysis": {
        "matched_certifications": [],
        "missing_certifications": [],
        "certification_score":    0
    },
    "project_analysis": {
        "project_relevance_score":  0,
        "project_complexity_score": 0,
        "project_scale_score":      0,
        "most_relevant_projects":   []
    },
    "responsibility_alignment": {
        "responsibility_match_percentage": 0,
        "matched_responsibilities":        [],
        "missing_responsibilities":        []
    },
    "category_scores_for_ats": {
        "skills_score":          0,
        "experience_score":      0,
        "education_score":       0,
        "certification_score":   0,   # singular — matches schema.json
        "project_score":         0,   # singular — matches schema.json
        "responsibility_score":  0,   # singular — matches schema.json
        "keyword_score":         0,
        "leadership_score":      0,
        "communication_score":   0,   # LLM provides a guess; Python may override from DB
        "domain_score":          0
    },
    "risk_analysis": {
        "risk_flags": []
    },
    "strength_analysis": {
        "top_strengths": []
    },
    "weakness_analysis": {
        "top_weaknesses": []
    },
    "interview_intelligence": {
        "interview_focus_points": []
    },
    "hiring_recommendation": {
        "recommendation": "Maybe",   # Strong Hire | Hire | Maybe | No Hire
        "reasoning":      ""
    },
    "ranking_summary": ""
}

SYSTEM_PROMPT = """
You are an expert ATS (Applicant Tracking System) evaluator.

Compare the RESUME against the JOB DESCRIPTION and return structured analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULE — DO NOT COMPUTE FINAL SCORE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY category_scores_for_ats sub-scores (0–100 each).
The final weighted ATS score is computed deterministically by Python — never by you.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RULES (category_scores_for_ats)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0   = absolutely no evidence in resume
1–39  = poor match
40–59 = weak match
60–69 = acceptable
70–79 = good match
80–89 = strong match
90–99 = excellent match
100   = exceeds expectations

SEMANTIC MATCHING — use conceptual equivalence:
  REST API ≈ FastAPI / Flask
  Django   ≈ Backend Development
  TensorFlow ≈ Machine Learning
  SQL      ≈ PostgreSQL / MySQL
  AWS ≈ Azure ≈ GCP
  CI/CD    ≈ DevOps
  PMP      ≈ Project Management

CATEGORY RULES:
  skills_score         — match required_skills semantically; reward relevant extras;
                         if JD has no required skills → 100
  experience_score     — years + seniority + industry + tech stack;
                         exceeding years boosts; different industry reduces;
                         if none required → 100
  education_score      — exact/equivalent degree = high; related field = medium;
                         unrelated = low; if none required → 100
  certification_score  — ONLY relevant certs count; unrelated certs MUST NOT
                         affect score; if none required → 100
  project_score        — enterprise/production > academic; scale, complexity,
                         ownership; if none required → 100
  responsibility_score — semantic match; partial coverage allowed; if none → 100
  keyword_score        — match required_keywords in resume text / skills / experience
  leadership_score     — team lead, architect, mentoring = high; no evidence = low
  communication_score  — provide your best estimate (Python may override from DB)
  domain_score         — same domain = high; transferable = medium; unrelated = low

top_weaknesses — MUST contain EXACTLY 3 items maximum (not more, not less if possible)
Return ONLY the 3 most important weaknesses ranked by severi

RISK FLAGS (risk_analysis.risk_flags) — include ONLY if strongly supported:
  insufficient_experience, missing_core_skills, degree_mismatch, major_mismatch,
  certification_gap, employment_gap, excessive_job_hopping,
  overqualification, underqualification

HIRING RECOMMENDATION BANDS (for hiring_recommendation.recommendation):
  Strong Hire: score would be ~85–100
  Hire:        score would be ~70–84
  Maybe:       score would be ~50–69
  No Hire:     score would be ~0–49
  (Your recommendation is a qualitative judgment; Python confirms against the
  actual weighted score and flags any disagreement.)

ABSOLUTE RULES:
  - NEVER invent missing information
  - Missing data = 0 (never positive)
  - Output ONLY valid JSON matching the schema exactly
  - temperature=0 — deterministic output
STRICT LIMIT RULES:
- top_strengths: max 3 items
- top_weaknesses: max 3 items
- interview_focus_points: max 5 items
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JSON OUTPUT REQUIREMENTS (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY a single valid JSON object.

DO NOT:
- Add markdown.
- Add ```json fences.
- Add explanations.
- Add notes.
- Add comments.
- Add trailing commas.
- Add text before or after the JSON.

The response MUST be parseable by Python's json.loads() without any modification.

Every object and array must be properly closed.

Every property name must be enclosed in double quotes.

Every string value must use double quotes.

Do not omit required fields.

If information is unavailable, use:
- "" for strings
- [] for arrays
- false for booleans
- 0 for numeric values

Never return null unless the schema explicitly requires it.

Your entire response must be one complete JSON object matching the provided schema exactly.

If you cannot determine a value, return the default value defined in the schema instead of inventing information.

INVALID OUTPUT EXAMPLES:
- Markdown
- ```json
- Partial JSON
- Missing commas
- Missing quotes
- Extra text after the closing }

VALID OUTPUT:
{
  ...
}
"""


def build_prompt(resume: dict, jd: dict) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"OUTPUT SCHEMA (return JSON matching this structure exactly):\n"
        f"{json.dumps(V2_OUTPUT_SCHEMA, indent=2)}\n\n"
        f"RESUME:\n{json.dumps(resume, indent=2)}\n\n"
        f"JOB DESCRIPTION:\n{json.dumps(jd, indent=2)}"
    )


# =========================
# GEMINI CLIENT
# =========================
def call_gemini(prompt: str, max_retries: int = 5) -> dict:
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.5, 1.5))

            response = gemini_client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0,
                }
            )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            parsed = json.loads(raw)
            logger.debug("Gemini call successful on attempt %d", attempt + 1)
            return parsed

        except ServerError as e:
            last_error = e
            wait = 2 ** (attempt + 1) + random.uniform(0, 1)
            logger.warning(
                "Gemini ServerError (attempt %d): %s — retrying in %.1fs",
                attempt + 1, e, wait
            )
            time.sleep(wait)

        except json.JSONDecodeError as e:
            logger.error("Gemini returned invalid JSON: %s", e)
            raise ValueError(f"Gemini JSON parse error: {e}") from e

    raise RuntimeError(f"Gemini API failed after {max_retries} attempts: {last_error}")


# =========================
# CORE PIPELINE
# =========================
def _run_pipeline(
    resume_id:    int,
    jd_id:        int,
    raw_resume:   dict,
    raw_jd:       dict,
    conn:         psycopg.Connection,
    comparison_hash: str,
) -> dict:
    """
    Inner pipeline — preprocessing → Gemini → schema validation → scoring → DB.
    Returns the full enriched evaluation dict (same shape as evaluate_candidate).
    """
    resume = preprocess_resume(raw_resume)
    jd     = preprocess_jd(raw_jd)

    # ── Step 1: LLM analysis (no final score) ──────────────────────────────
    prompt    = build_prompt(resume, jd)
    llm_output = call_gemini(prompt)

    # ── Step 2: schema validation + deterministic scoring ──────────────────
    # evaluate_candidate() calls validate_llm_output() internally (jsonschema
    # if available, fallback key-check otherwise), then computes the weighted
    # final ATS score and enriches the payload.
    # enriched = evaluate_candidate(
    #     job_description=raw_jd,       # needs 'weights' key for normalization
    #     llm_output=llm_output,
    #     candidate_json=raw_resume,    # supplies communication_score override
    #     skip_validation=False,
    #     injected_weights=llm_output["normalized_weights_used"],
    # )
    enriched = evaluate_candidate(
    job_description=raw_jd,
    llm_output=llm_output,
    candidate_json=raw_resume,
    skip_validation=False,
    injected_weights=raw_jd.get("weights", {}),
)

    # ── Step 3: persist ────────────────────────────────────────────────────
    save_evaluation(conn, resume_id, jd_id, comparison_hash, enriched["postgres_row"])

    return enriched


# =========================
# PUBLIC API
# =========================
def compare_resume_and_jd(
    resume_id: int,
    jd_id:     int,
    conn:      Optional[psycopg.Connection] = None,
    force:     bool = False,
) -> dict:
    """
    Compare a single resume against a single job description.

    Returns the full enriched evaluation dict which includes:
      - all LLM analysis sections (skill_match, experience_analysis, …)
      - category_scores_for_ats  (LLM sub-scores)
      - weighted_final_ats_score (Python computed)
      - score_band_recommendation + score_recommendation_alignment
      - postgres_row (flat DB-ready dict)
    """
    owns_conn = conn is None
    if owns_conn:
        conn = get_db_connection()

    try:
        ensure_ats_table(conn)
        comparison_hash = make_comparison_hash(resume_id, jd_id)

        if not force:
            cached = get_cached_evaluation(conn, comparison_hash)
            if cached:
                logger.info("Cache hit: resume=%s jd=%s", resume_id, jd_id)
                return cached

        raw_resume = fetch_resume(conn, resume_id)
        if not raw_resume:
            raise ValueError(f"Resume ID {resume_id} not found")

        raw_jd = fetch_job_description(conn, jd_id)
        if not raw_jd:
            raise ValueError(f"JD ID {jd_id} not found")

        return _run_pipeline(resume_id, jd_id, raw_resume, raw_jd, conn, comparison_hash)

    finally:
        if owns_conn:
            conn.close()


def batch_compare(
    jd_id:                int,
    page_size:            int   = 50,
    delay_between_calls:  float = 1.0,
    max_resumes:          Optional[int] = None,
) -> list[dict]:
    """
    Compare all resumes against a single JD.
    Pre-fetches the JD once; processes resumes in paginated batches.

    Returns list of enriched evaluation dicts (each has 'resume_id' prepended).
    """
    conn = get_db_connection()
    ensure_ats_table(conn)

    raw_jd = fetch_job_description(conn, jd_id)
    if not raw_jd:
        conn.close()
        raise ValueError(f"JD ID {jd_id} not found")

    results:         list[dict] = []
    page             = 0
    total_processed  = 0

    logger.info("Starting batch compare for JD %s", jd_id)

    try:
        while True:
            resume_ids = fetch_all_resume_ids(conn, page, page_size)
            if not resume_ids:
                logger.info("No more resumes at page %d — done", page)
                break

            for resume_id in resume_ids:
                if max_resumes and total_processed >= max_resumes:
                    logger.info("Reached max_resumes cap: %d", max_resumes)
                    return results

                comparison_hash = make_comparison_hash(resume_id, jd_id)

                cached = get_cached_evaluation(conn, comparison_hash)
                if cached:
                    logger.info("Cache hit: resume=%s jd=%s", resume_id, jd_id)
                    results.append({"resume_id": resume_id, **cached})
                    total_processed += 1
                    continue

                raw_resume = fetch_resume(conn, resume_id)
                if not raw_resume:
                    logger.warning("Resume %s not found — skipping", resume_id)
                    continue

                try:
                    enriched = _run_pipeline(
                        resume_id, jd_id, raw_resume, raw_jd, conn, comparison_hash
                    )
                    results.append({"resume_id": resume_id, **enriched})
                    logger.info(
                        "Evaluated resume=%s jd=%s  final=%.1f  rec=%s",
                        resume_id, jd_id,
                        enriched.get("weighted_final_ats_score", 0),
                        enriched.get("score_band_recommendation", "?"),
                    )
                except Exception as e:
                    logger.error("Failed resume=%s: %s", resume_id, e)

                total_processed += 1
                time.sleep(delay_between_calls)

            page += 1

    finally:
        conn.close()

    logger.info(
        "Batch complete: %d resumes evaluated for JD %s", total_processed, jd_id
    )
    return results


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="ATS Resume vs JD Comparator v2")
    sub    = parser.add_subparsers(dest="command")

    sp = sub.add_parser("single", help="Compare one resume to one JD")
    sp.add_argument("--resume-id", type=int, required=True)
    sp.add_argument("--jd-id",     type=int, required=True)
    sp.add_argument("--force",     action="store_true", help="Skip cache")

    bp = sub.add_parser("batch", help="Compare all resumes to one JD")
    bp.add_argument("--jd-id",       type=int,  required=True)
    bp.add_argument("--page-size",   type=int,  default=50)
    bp.add_argument("--delay",       type=float, default=1.0)
    bp.add_argument("--max-resumes", type=int,  default=None)

    args = parser.parse_args()

    if args.command == "single":
        result = compare_resume_and_jd(args.resume_id, args.jd_id, force=args.force)
        # drop postgres_row from stdout (it's in the DB already)
        printable = {k: v for k, v in result.items() if k != "postgres_row"}
        print(json.dumps(printable, indent=2))

    elif args.command == "batch":
        results = batch_compare(
            jd_id=args.jd_id,
            page_size=args.page_size,
            delay_between_calls=args.delay,
            max_resumes=args.max_resumes,
        )
        printable = [
            {k: v for k, v in r.items() if k != "postgres_row"}
            for r in results
        ]
        print(json.dumps(printable, indent=2))

    else:
        parser.print_help()
        sys.exit(1)
