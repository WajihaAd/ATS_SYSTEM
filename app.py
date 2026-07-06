"""
app.py — AI ATS Resume Ranking System (Streamlit)
==================================================

Wires together the existing backend pipeline:

    job_description_parser.parse_job_description
    resume_extractor_gemini.parse_resume / get_resume_hash / extract_text_from_file
    compare_resume_and_jd.compare_resume_and_jd
    ranking.rank_all_candidates / get_tier_summary / get_score_band_summary /
             export_ranked_to_dicts / RankingFilter
    exporter.generate_excel / generate_csv_all / generate_csv_top10 / generate_csv_top20

This file does NOT modify any backend module. It only:
  - creates the `job_descriptions` / `resumes` tables those modules INSERT into
    (they assume the tables already exist)
  - de-duplicates JD / resume uploads by content hash before calling the
    (expensive, Gemini-backed) parse functions
  - resolves the integer IDs the pipeline needs (jd_id / resume_id / eval_id)
  - supplements the ranking output with a couple of fields ranking.py's query
    does not select (matched/missing skills, strengths/weaknesses, summary)
    by reading them straight back from `ats_evaluations`
  - renders everything in a dark, recruiter-dashboard style UI

NOTE ON KNOWN BACKEND ISSUES
-----------------------------
Two issues exist in the backend files as supplied, outside this file's control:

1. `compare_resume_and_jd._run_pipeline` calls
   `evaluate_candidate(..., injected_weights=llm_output["normalized_weights_used"])`
   — but `llm_output` is the raw Gemini response and never contains that key
   (it's a key on `evaluate_candidate`'s *return value*, not its input).
   Every single evaluation will raise a KeyError until that line is changed
   to read the JD's own weights, e.g. `injected_weights=raw_jd.get("weights")`.
2. `ranking.py`'s `fetch_job_description`-style query in `compare_resume_and_jd.py`
   ignores the `jd_id` parameter (no WHERE clause), and `ranking._EVAL_COLUMNS`
   doesn't line up with the columns actually selected in
   `_iter_evaluated_candidates`, so a couple of fields shift by one position.

This app guards every backend call with try/except and surfaces failures as
`st.error(...)` (never a silent crash), and re-derives the fields affected by
issue #2 directly from `ats_evaluations` so the UI stays correct regardless.
"""

from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import psycopg
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Backend imports (per spec — used exactly as provided, never rewritten)
# ---------------------------------------------------------------------------
try:
    from job_description_parser import parse_job_description
    from resume_extractor_gemini import parse_resume, get_resume_hash, extract_text_from_file
    from compare_resume_and_jd import compare_resume_and_jd
    from ranking import (
        rank_all_candidates,
        get_tier_summary,
        get_score_band_summary,
        export_ranked_to_dicts,
        RankingFilter,
    )
    from exporter import generate_excel, generate_csv_all, generate_csv_top10, generate_csv_top20
except Exception as e:  # missing env vars / deps will raise at import time
    st.set_page_config(page_title="ATS Ranking System", layout="wide")
    st.error(
        "Failed to load the backend pipeline. This usually means "
        "`GEMINI_API_KEY` / `DB_PASSWORD` are missing from your `.env`, or a "
        f"dependency isn't installed.\n\n**Details:** {e}"
    )
    st.stop()

DB_PASSWORD = os.getenv("DB_PASSWORD")

# ---------------------------------------------------------------------------
# Page config + dark theme styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI ATS Resume Ranking System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f17; color: #e7eaf0; }
    section[data-testid="stSidebar"] { background-color: #0f1420; }
    h1, h2, h3, h4 { color: #f5f7fb; font-family: 'Segoe UI', sans-serif; }
    .candidate-card {
        background: linear-gradient(145deg, #131a29, #0f1420);
        border: 1px solid #232c3d;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.35);
    }
    .rank-badge {
        display: inline-block; font-weight: 700; font-size: 0.85rem;
        padding: 3px 10px; border-radius: 20px; color: #0b0f17;
        background: #5ec8f8; margin-right: 8px;
    }
    .tier-A\\+ { background:#4ee08a !important; }
    .tier-A   { background:#7ad77a !important; }
    .tier-B   { background:#f6d860 !important; }
    .tier-C   { background:#f6a360 !important; }
    .tier-D   { background:#f15c5c !important; }
    .band-pill {
        font-size: 0.75rem; font-weight: 600; padding: 2px 9px; border-radius: 12px;
        background:#1d2536; color:#aab4c8; margin-right:6px; display:inline-block;
    }
    .score-big { font-size: 1.9rem; font-weight: 800; color:#5ec8f8; }
    .skill-pill-match { background:#163d2c; color:#7ee0a8; padding:2px 8px; border-radius:10px; font-size:0.78rem; margin:2px; display:inline-block; }
    .skill-pill-missing { background:#3d1616; color:#f08a8a; padding:2px 8px; border-radius:10px; font-size:0.78rem; margin:2px; display:inline-block; }
    .why-line { color:#b8c0d0; font-size:0.85rem; margin:2px 0; }
    hr { border-color: #232c3d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# DB connection + schema bootstrap
# ---------------------------------------------------------------------------
def get_db_connection() -> psycopg.Connection:
    return psycopg.connect(
        dbname="resume_db",
        user="postgres",
        password=DB_PASSWORD,
        host="localhost",
        port="5432",
    )


@st.cache_resource(show_spinner=False)
def ensure_schema() -> bool:
    """Create job_descriptions / resumes tables if they don't exist yet.
    (ats_evaluations is created lazily by compare_resume_and_jd.ensure_ats_table)."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS job_descriptions (
                    id SERIAL PRIMARY KEY,
                    job_title TEXT,
                    department TEXT,
                    seniority TEXT,
                    location TEXT,
                    employment_type TEXT,
                    required_degree TEXT,
                    required_major JSONB,
                    required_cgpa NUMERIC,
                    required_experience_years NUMERIC,
                    required_skills JSONB,
                    preferred_skills JSONB,
                    required_certifications JSONB,
                    preferred_certifications JSONB,
                    required_languages JSONB,
                    required_coursework JSONB,
                    required_projects JSONB,
                    required_keywords JSONB,
                    responsibilities JSONB,
                    consultancy_experience_required BOOLEAN,
                    mega_project_experience_required BOOLEAN,
                    donor_project_experience_required BOOLEAN,
                    weights JSONB,
                    raw_data JSONB,
                    jd_hash TEXT UNIQUE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS resumes (
                    id SERIAL PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    phone TEXT,
                    location TEXT,
                    degree TEXT,
                    major TEXT,
                    cgpa TEXT,
                    university TEXT,
                    graduation_year TEXT,
                    experience_years NUMERIC,
                    skills JSONB,
                    certifications JSONB,
                    leadership JSONB,
                    languages JSONB,
                    consultancy_companies JSONB,
                    mega_projects JSONB,
                    raw_data JSONB,
                    filename TEXT,
                    file_hash TEXT UNIQUE,
                    communication_score NUMERIC,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
        conn.commit()
        return True
    finally:
        conn.close()


def _parse_maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


# ---------------------------------------------------------------------------
# DB helper queries (read-only wiring around the backend modules)
# ---------------------------------------------------------------------------
def get_jd_id_by_hash(jd_hash: str) -> Optional[int]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM job_descriptions WHERE jd_hash = %s", (jd_hash,))
            row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_resume_id_by_hash(file_hash: str) -> Optional[int]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM resumes WHERE file_hash = %s", (file_hash,))
            row = cur.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def fetch_jd_record(jd_id: int) -> Optional[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, job_title, department, seniority, location, employment_type,
                          required_degree, required_major, required_cgpa,
                          required_experience_years, required_skills, preferred_skills,
                          required_certifications, preferred_certifications,
                          required_languages, required_projects, required_keywords,
                          responsibilities, weights,
                          consultancy_experience_required, mega_project_experience_required,
                          donor_project_experience_required, created_at
                   FROM job_descriptions WHERE id = %s""",
                (jd_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cols = [
                "id", "job_title", "department", "seniority", "location", "employment_type",
                "required_degree", "required_major", "required_cgpa",
                "required_experience_years", "required_skills", "preferred_skills",
                "required_certifications", "preferred_certifications",
                "required_languages", "required_projects", "required_keywords",
                "responsibilities", "weights",
                "consultancy_experience_required", "mega_project_experience_required",
                "donor_project_experience_required", "created_at",
            ]
            rec = dict(zip(cols, row))
            for k in ("required_major", "required_skills", "preferred_skills",
                      "required_certifications", "preferred_certifications",
                      "required_languages", "required_projects", "required_keywords",
                      "responsibilities", "weights"):
                rec[k] = _parse_maybe_json(rec.get(k))
            return rec
    finally:
        conn.close()


def fetch_all_jds(limit: int = 50) -> list[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, job_title, department, created_at FROM job_descriptions "
                "ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {"id": r[0], "job_title": r[1] or "Untitled", "department": r[2] or "", "created_at": r[3]}
            for r in rows
        ]
    finally:
        conn.close()


def batch_fetch_resume_profiles(resume_ids: list[int]) -> dict[int, dict]:
    if not resume_ids:
        return {}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, name, email, phone, location, degree, major, university,
                          experience_years, skills, certifications
                   FROM resumes WHERE id = ANY(%s)""",
                (resume_ids,),
            )
            rows = cur.fetchall()
        cols = ["id", "name", "email", "phone", "location", "degree", "major",
                "university", "experience_years", "skills", "certifications"]
        out = {}
        for r in rows:
            rec = dict(zip(cols, r))
            rec["skills"] = _parse_maybe_json(rec.get("skills")) or []
            rec["certifications"] = _parse_maybe_json(rec.get("certifications")) or []
            out[rec["id"]] = rec
        return out
    finally:
        conn.close()


def batch_fetch_eval_extras(eval_ids: list[int]) -> dict[int, dict]:
    """Pulls fields ranking.py's own query doesn't select (matched/missing
    skills, strengths/weaknesses, ranking summary, true weighted score, etc.)
    straight from ats_evaluations, keyed by evaluation id."""
    if not eval_ids:
        return {}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, matched_skills, missing_skills, top_strengths, top_weaknesses,
                          ranking_summary, llm_recommendation, weighted_final_ats_score,
                          score_band_recommendation, recommendation_alignment,
                          job_title, department
                   FROM ats_evaluations WHERE id = ANY(%s)""",
                (eval_ids,),
            )
            rows = cur.fetchall()
        cols = ["id", "matched_skills", "missing_skills", "top_strengths", "top_weaknesses",
                "ranking_summary", "llm_recommendation", "weighted_final_ats_score",
                "score_band_recommendation", "recommendation_alignment", "job_title", "department"]
        out = {}
        for r in rows:
            rec = dict(zip(cols, r))
            for k in ("matched_skills", "missing_skills", "top_strengths", "top_weaknesses"):
                rec[k] = _parse_maybe_json(rec.get(k)) or []
            out[rec["id"]] = rec
        return out
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        return {}
    finally:
        conn.close()


def make_jd_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
DEFAULTS = {
    "jd": None,                # raw JD dict (from DB or parser)
    "jd_id": None,
    "jd_text": "",
    "resumes": [],              # list[{filename, file_hash, resume_id, name, status}]
    "ranked": [],                # list[dict] from export_ranked_to_dicts
    "resume_profiles": {},       # resume_id -> dict
    "eval_extras": {},           # eval_id   -> dict
    "tier_summary": {},
    "band_summary": {},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

ensure_schema()

# ---------------------------------------------------------------------------
# Pipeline operations
# ---------------------------------------------------------------------------
def handle_parse_jd(jd_text: str):
    jd_text = (jd_text or "").strip()
    if len(jd_text) < 30:
        st.sidebar.error("Job description text is too short to parse.")
        return

    jd_hash = make_jd_hash(jd_text)
    existing_id = get_jd_id_by_hash(jd_hash)

    if existing_id:
        st.session_state.jd_id = existing_id
        st.session_state.jd = fetch_jd_record(existing_id)
        st.sidebar.success(f"JD already in database — reusing JD #{existing_id}.")
        return

    with st.spinner("Parsing job description with Gemini and saving to PostgreSQL..."):
        try:
            jd = parse_job_description(jd_text)
        except Exception as e:
            st.sidebar.error(f"Failed to parse job description: {e}")
            return

    new_id = get_jd_id_by_hash(jd.get("jd_hash") or jd_hash)
    if not new_id:
        st.sidebar.error("JD parsed but could not be located in the database afterward.")
        return

    st.session_state.jd_id = new_id
    st.session_state.jd = fetch_jd_record(new_id) or jd
    # reset downstream state — new JD means old ranking is no longer relevant
    st.session_state.ranked = []
    st.session_state.eval_extras = {}
    st.sidebar.success(f"JD parsed and stored as JD #{new_id}.")


def handle_process_resumes(uploaded_files: list):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one resume file first.")
        return

    progress = st.sidebar.progress(0.0)
    status = st.sidebar.empty()
    total = len(uploaded_files)
    succeeded, reused, failed = 0, 0, 0

    for i, f in enumerate(uploaded_files):
        status.text(f"Processing resumes… {i + 1}/{total}")
        try:
            file_bytes = f.getvalue()
            file_hash = get_resume_hash(file_bytes)
            existing_id = get_resume_id_by_hash(file_hash)

            if existing_id:
                profile = batch_fetch_resume_profiles([existing_id]).get(existing_id, {})
                entry = {
                    "filename": f.name, "file_hash": file_hash,
                    "resume_id": existing_id, "name": profile.get("name", f.name),
                    "status": "reused",
                    "file_bytes": file_bytes,
                }
                reused += 1
            else:
                result = parse_resume(file_bytes, f.name)
                new_id = get_resume_id_by_hash(file_hash)
                if not new_id:
                    raise RuntimeError("Resume parsed but not found in DB afterward.")
                entry = {
                    "filename": f.name, "file_hash": file_hash,
                    "resume_id": new_id, "name": result.get("name", f.name),
                    "status": "parsed",
                    "file_bytes": file_bytes,
                }
                succeeded += 1

            existing_ids = {r["file_hash"] for r in st.session_state.resumes}
            if file_hash not in existing_ids:
                st.session_state.resumes.append(entry)

        except Exception as e:
            failed += 1
            st.sidebar.error(f"❌ {f.name}: {e}")

        progress.progress((i + 1) / total)

    status.empty()
    progress.empty()
    st.sidebar.success(f"Resumes ready — {succeeded} parsed, {reused} reused, {failed} failed.")


def handle_evaluate_and_rank(filters: "RankingFilter"):
    jd_id = st.session_state.jd_id
    resumes = st.session_state.resumes

    if not jd_id:
        st.sidebar.warning("Parse or load a job description first.")
        return
    if not resumes:
        st.sidebar.warning("Process at least one resume first.")
        return

    progress = st.sidebar.progress(0.0)
    status = st.sidebar.empty()
    total = len(resumes)
    ok, failed = 0, 0

    for i, r in enumerate(resumes):
        status.text(f"Evaluating… {i + 1}/{total}")
        try:
            compare_resume_and_jd(r["resume_id"], jd_id)
            ok += 1
        except Exception as e:
            failed += 1
            st.sidebar.error(f"❌ Evaluation failed for {r.get('name') or r['filename']}: {e}")
        progress.progress((i + 1) / total)

    status.empty()
    progress.empty()

    if ok == 0:
        st.sidebar.error(
            "No resumes could be evaluated — see errors above. The ranking step was skipped."
        )
        return

    try:
        ranked_objs = rank_all_candidates(jd_id=jd_id, filters=filters)
    except Exception as e:
        st.sidebar.error(f"Ranking failed: {e}")
        return

    ranked = export_ranked_to_dicts(ranked_objs)
    st.session_state.ranked = ranked
    st.session_state.tier_summary = get_tier_summary(ranked_objs)
    st.session_state.band_summary = get_score_band_summary(ranked_objs)

    resume_ids = [c["resume_id"] for c in ranked]
    eval_ids = [c["eval_id"] for c in ranked]
    st.session_state.resume_profiles = batch_fetch_resume_profiles(resume_ids)
    st.session_state.eval_extras = batch_fetch_eval_extras(eval_ids)

    st.sidebar.success(f"Evaluated {ok}/{total} resumes. Ranked {len(ranked)} candidates.")


def handle_load_existing_jd(jd_id: int, filters: "RankingFilter"):
    rec = fetch_jd_record(jd_id)
    if not rec:
        st.sidebar.error("That JD could not be loaded.")
        return
    st.session_state.jd_id = jd_id
    st.session_state.jd = rec

    try:
        ranked_objs = rank_all_candidates(jd_id=jd_id, filters=filters)
    except Exception as e:
        st.sidebar.error(f"Loaded JD #{jd_id}, but ranking failed: {e}")
        return

    ranked = export_ranked_to_dicts(ranked_objs)
    st.session_state.ranked = ranked
    st.session_state.tier_summary = get_tier_summary(ranked_objs)
    st.session_state.band_summary = get_score_band_summary(ranked_objs)

    resume_ids = [c["resume_id"] for c in ranked]
    eval_ids = [c["eval_id"] for c in ranked]
    st.session_state.resume_profiles = batch_fetch_resume_profiles(resume_ids)
    st.session_state.eval_extras = batch_fetch_eval_extras(eval_ids)
    st.session_state.resumes = [
        {"filename": "", "file_hash": "", "resume_id": rid,
         "name": st.session_state.resume_profiles.get(rid, {}).get("name", ""), "status": "loaded"}
        for rid in resume_ids
    ]
    st.sidebar.success(f"Loaded JD #{jd_id} with {len(ranked)} ranked candidates.")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 ATS Control Panel")

    st.markdown("### 1️⃣ Job Description")
    jd_input_mode = st.radio("JD input", ["Paste text", "Upload file"], horizontal=True, label_visibility="collapsed")

    jd_text_value = ""
    if jd_input_mode == "Paste text":
        jd_text_value = st.text_area("Paste job description", height=160, key="jd_text_area")
    else:
        jd_file = st.file_uploader("Upload JD (pdf, docx, txt)", type=["pdf", "docx", "txt", "md"])
        if jd_file is not None:
            try:
                jd_text_value = extract_text_from_file(jd_file.getvalue(), jd_file.name)
                with st.expander("Preview extracted text"):
                    st.text(jd_text_value[:1500])
            except Exception as e:
                st.error(f"Could not extract JD text: {e}")

    if st.button("🚀 Parse JD", use_container_width=True):
        handle_parse_jd(jd_text_value)

    if st.session_state.jd_id:
        st.caption(f"Active JD: **{(st.session_state.jd or {}).get('job_title', 'Untitled')}**  (#{st.session_state.jd_id})")

    st.markdown("---")
    st.markdown("### 2️⃣ Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload resumes (pdf, docx, txt)", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
    )
    if st.button("📄 Process Resumes", use_container_width=True):
        handle_process_resumes(uploaded_resumes or [])

    if st.session_state.resumes:
        st.caption(f"{len(st.session_state.resumes)} resume(s) ready for evaluation")

    st.markdown("---")
    st.markdown("### 3️⃣ Evaluate & Rank")
    with st.expander("Ranking filters", expanded=False):
        min_score = st.slider("Minimum ATS score", 0, 100, 0)
        min_exp_score = st.slider("Minimum experience score", 0, 100, 0)
        hire_or_better = st.checkbox("Only show Hire / Strong Hire", value=False)
        exclude_risks = st.multiselect(
            "Exclude candidates with risk flags",
            ["missing_core_skills", "insufficient_experience",
             "degree_mismatch", "major_mismatch", "certification_gap",
              "underqualification"],
        )

    active_filters = RankingFilter(
        min_score=float(min_score),
        min_experience_score=float(min_exp_score),
        exclude_risk_flags=exclude_risks,
        only_hire_or_better=hire_or_better,
    )

    if st.button("⚡ Evaluate & Rank", use_container_width=True, type="primary"):
        handle_evaluate_and_rank(active_filters)

    st.markdown("---")
    with st.expander("📂 Load existing JD from database"):
        jds = fetch_all_jds()
        if jds:
            options = {f"#{j['id']} — {j['job_title']} ({j['department']})": j["id"] for j in jds}
            choice = st.selectbox("Select a previously parsed JD", list(options.keys()))
            if st.button("Load & Re-rank", use_container_width=True):
                handle_load_existing_jd(options[choice], active_filters)
        else:
            st.caption("No job descriptions stored yet.")

# ---------------------------------------------------------------------------
# MAIN AREA
# ---------------------------------------------------------------------------

st.markdown("""
<div class="ticker-wrap">
    <div class="ticker">
        🤖 AI-Assisted Recruitment System • Candidate rankings and ATS scores are generated using Artificial Intelligence and should be reviewed and verified by recruiters before making hiring decisions.
    </div>
</div>

<style>
.ticker-wrap {
    width: 100%;
    overflow: hidden;
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 8px 0;
    margin-bottom: 15px;
}

.ticker {
    display: inline-block;
    white-space: nowrap;
    color: #FFD166;
    font-weight: 600;
    padding-left: 100%;
    animation: ticker 22s linear infinite;
}

@keyframes ticker {
    0% {
        transform: translateX(0%);
    }
    100% {
        transform: translateX(-100%);
    }
}
</style>
""", unsafe_allow_html=True)
st.title("AI ATS Resume Ranking System")
jd = st.session_state.jd or {}
ranked = st.session_state.ranked
top_cols = st.columns(4)
top_cols[0].metric("Active JD", jd.get("job_title", "—") if jd else "—")
top_cols[1].metric("Resumes loaded", len(st.session_state.resumes))
top_cols[2].metric("Ranked candidates", len(ranked))
top_cols[3].metric(
    "Top score",
    f"{max((c.get('final_rank_score', 0) for c in ranked), default=0):.1f}" if ranked else "—",
)

st.markdown("---")

tab_cards, tab_table, tab_jd, tab_export = st.tabs(
    ["🏆 Ranked Candidates", "📊 Data Table", "📋 JD Analysis", "⬇️ Export"]
)


def get_display_score(candidate: dict) -> float:
    """Prefer the true score read back from ats_evaluations (eval_extras);
    falls back to ranking.py's own value if extras aren't available."""
    extra = st.session_state.eval_extras.get(candidate["eval_id"], {})
    val = extra.get("weighted_final_ats_score")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    return float(candidate.get("raw_ats_score") or 0)


# ---------------- Tab 1: Ranked Candidate Cards ----------------
with tab_cards:
    if not ranked:
        st.info("No ranked candidates yet. Parse a JD, process resumes, then click **Evaluate & Rank**.")
    else:
        if st.session_state.tier_summary:
            tcols = st.columns(len(st.session_state.tier_summary))
            for col, (tier, count) in zip(tcols, st.session_state.tier_summary.items()):
                col.metric(f"Tier {tier}", count)
        st.markdown("")

        for c in ranked:
            extra = st.session_state.eval_extras.get(c["eval_id"], {})
            profile = st.session_state.resume_profiles.get(c["resume_id"], {})
            resume_entry = next(
            (
                r for r in st.session_state.resumes
                if r["resume_id"] == c["resume_id"]
            ),
            None,
        )
            display_score = get_display_score(c)

            matched_skills = extra.get("matched_skills") or []
            missing_skills = extra.get("missing_skills") or []
            top_strengths = extra.get("top_strengths") or c.get("top_strengths") or []
            top_weaknesses = extra.get("top_weaknesses") or c.get("top_weaknesses") or []
            ranking_summary = extra.get("ranking_summary") or c.get("ranking_summary") or ""
            llm_rec = extra.get("llm_recommendation") or c.get("score_band", "")

            name = c.get("candidate_name") or profile.get("name") or f"Resume #{c['resume_id']}"
            email = profile.get("email") or "—"

            tier = c.get("tier", "D")
            tier_class = "tier-" + tier.replace("+", "plus")

            with st.container():
                st.markdown('<div class="candidate-card">', unsafe_allow_html=True)
                head_l, head_r = st.columns([3, 1])
                with head_l:
                    st.markdown(
                        f'<span class="rank-badge">#{c["rank"]}</span>'
                        f'<span class="rank-badge {tier_class}">Tier {tier}</span>'
                        f'<span class="band-pill">{c.get("score_band", "")}</span>'
                        f'<span class="band-pill">LLM: {llm_rec}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"### {name}")
                    st.caption(f"✉️ {email}   ·   Resume ID #{c['resume_id']}   ·   Eval ID #{c['eval_id']}")
                with head_r:
                    st.markdown(f'<div class="score-big">{display_score:.1f}</div>', unsafe_allow_html=True)
                    st.caption("ATS score")
                    st.progress(min(1.0, max(0.0, display_score / 100.0)))

                if c.get("risk_penalty", 0):
                    st.caption(f"⚠ Risk-adjusted final rank score: {c['final_rank_score']:.1f} "
                               f"(−{c['risk_penalty']:.0f} pts for: {', '.join(c.get('risk_flags') or [])})")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Matched skills**")
                    if matched_skills:
                        st.markdown(
                            "".join(f'<span class="skill-pill-match">{s}</span>' for s in matched_skills[:25]),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No matched skills on record.")
                with col_b:
                    st.markdown("**Missing skills**")
                    if missing_skills:
                        st.markdown(
                            "".join(f'<span class="skill-pill-missing">{s}</span>' for s in missing_skills[:25]),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("No notable gaps.")

                with st.expander("Strengths, weaknesses & explanation"):
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.markdown("**Top strengths**")
                        for s in top_strengths:
                            st.markdown(f"- ✅ {s}")
                    with sc2:
                        st.markdown("**Top weaknesses**")
                        for w in top_weaknesses:
                            st.markdown(f"- ⚠️ {w}")

                    if ranking_summary:
                        st.markdown(f"**Summary:** {ranking_summary}")

                    if c.get("why_ranked_here"):
                        st.markdown("**Why ranked here:**")
                        for line in c["why_ranked_here"]:
                            st.markdown(f'<div class="why-line">• {line}</div>', unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
                if resume_entry and "file_bytes" in resume_entry:
                    st.download_button(
                        "📄 Download Resume",
                        data=resume_entry["file_bytes"],
                        file_name=resume_entry["filename"],
                        mime="application/octet-stream",
                        key=f"resume_{c['resume_id']}",
                    )

# ---------------- Tab 2: Data Table ----------------
with tab_table:
    if not ranked:
        st.info("No ranked candidates yet.")
    else:
        rows = []
        for c in ranked:
            extra = st.session_state.eval_extras.get(c["eval_id"], {})
            profile = st.session_state.resume_profiles.get(c["resume_id"], {})
            rows.append({
                "Rank": c["rank"],
                "Candidate": c.get("candidate_name") or profile.get("name") or f"Resume #{c['resume_id']}",
                "Email": profile.get("email") or "",
                "ATS Score": round(get_display_score(c), 2),
                "Final Rank Score": round(c.get("final_rank_score", 0), 2),
                "Tier": c.get("tier", ""),
                "Score Band": c.get("score_band", ""),
                "Risk Flags": ", ".join(c.get("risk_flags") or []),
                "Skills Score": c.get("category_scores", {}).get("skills_score", ""),
                "Experience Score": c.get("category_scores", {}).get("experience_score", ""),
                "Education Score": c.get("category_scores", {}).get("education_score", ""),
                "Project Score": c.get("category_scores", {}).get("project_score", ""),
                "Leadership Score": c.get("category_scores", {}).get("leadership_score", ""),
                "Domain Score": c.get("category_scores", {}).get("domain_score", ""),
            })
        df_display = pd.DataFrame(rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        st.session_state["_last_df"] = df_display

# ---------------- Tab 3: JD Analysis ----------------
with tab_jd:
    if not jd:
        st.info("No job description loaded yet. Parse a JD from the sidebar.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Job Title", jd.get("job_title") or "—")
        c2.metric("Department", jd.get("department") or "—")
        c3.metric("Seniority", jd.get("seniority") or "—")

        c4, c5 = st.columns(2)
        c4.metric("Required Degree", jd.get("required_degree") or "—")
        c5.metric("Required Experience (yrs)", jd.get("required_experience_years") or 0)

        st.markdown("#### Required Skills")
        req_skills = jd.get("required_skills") or []
        if req_skills:
            st.markdown(
                "".join(f'<span class="skill-pill-match">{s}</span>' for s in req_skills),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No required skills extracted.")

        pref_skills = jd.get("preferred_skills") or []
        if pref_skills:
            st.markdown("#### Preferred Skills")
            st.markdown(
                "".join(f'<span class="band-pill">{s}</span>' for s in pref_skills),
                unsafe_allow_html=True,
            )

        st.markdown("#### Responsibilities")
        responsibilities = jd.get("responsibilities") or []
        if responsibilities:
            for r in responsibilities:
                st.markdown(f"- {r}")
        else:
            st.caption("No responsibilities extracted.")

        st.markdown("#### ATS Weights")
        weights = jd.get("weights") or {}
        if weights:
            weights_df = pd.DataFrame({"Weight": weights}).sort_values("Weight", ascending=False)
            st.bar_chart(weights_df)
        else:
            st.caption("No custom weights set — defaults will be applied during scoring.")

        with st.expander("Raw JD record"):
            st.json(jd, expanded=False)

# ---------------- Tab 4: Export ----------------
with tab_export:
    if not ranked:
        st.info("No ranked candidates to export yet.")
    else:
        rows = []
        for c in ranked:
            extra = st.session_state.eval_extras.get(c["eval_id"], {})
            profile = st.session_state.resume_profiles.get(c["resume_id"], {})
            rows.append({
                "Rank": c["rank"],
                "Candidate Name": c.get("candidate_name") or profile.get("name") or f"Resume #{c['resume_id']}",
                "Email": profile.get("email") or "",
                "ATS Score": round(get_display_score(c), 2),
                "Final Rank Score": round(c.get("final_rank_score", 0), 2),
                "Tier": c.get("tier", ""),
                "Score Band": c.get("score_band", ""),
                "Risk Flags": ", ".join(c.get("risk_flags") or []),
                "Top Strengths": "; ".join(extra.get("top_strengths") or c.get("top_strengths") or []),
                "Top Weaknesses": "; ".join(extra.get("top_weaknesses") or c.get("top_weaknesses") or []),
                "Ranking Summary": extra.get("ranking_summary") or c.get("ranking_summary") or "",
            })
        df_all = pd.DataFrame(rows)
        df_top10 = df_all.head(10)
        df_top20 = df_all.head(20)

        pipeline_output = {
            "jd": jd,
            "df_all": df_all,
            "df_top10": df_top10,
            "df_top20": df_top20,
        }

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Excel report")
            try:
                excel_bytes = generate_excel(pipeline_output)
                st.download_button(
                    "⬇️ Download Excel (.xlsx)",
                    data=excel_bytes,
                    file_name=f"ats_ranking_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Excel export failed: {e}")

        with col2:
            st.markdown("#### CSV exports")
            try:
                st.download_button(
                    "⬇️ All candidates (CSV)", data=generate_csv_all(pipeline_output),
                    file_name="ats_all_candidates.csv", mime="text/csv", use_container_width=True,
                )
                st.download_button(
                    "⬇️ Top 10 (CSV)", data=generate_csv_top10(pipeline_output),
                    file_name="ats_top10.csv", mime="text/csv", use_container_width=True,
                )
                st.download_button(
                    "⬇️ Top 20 (CSV)", data=generate_csv_top20(pipeline_output),
                    file_name="ats_top20.csv", mime="text/csv", use_container_width=True,
                )
            except Exception as e:
                st.error(f"CSV export failed: {e}")

        st.markdown("#### Preview")
        st.dataframe(df_all, use_container_width=True, hide_index=True)
