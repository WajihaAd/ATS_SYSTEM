import os
import json
import psycopg
from dotenv import load_dotenv
from google import genai
import hashlib
# =========================
# LOAD ENV
# =========================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
DB_PASSWORD = os.getenv("DB_PASSWORD")

client = genai.Client(api_key=API_KEY)
MODEL = "gemini-2.5-flash"

def make_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =========================
# DB CONNECTION
# =========================
def get_db_connection():
    return psycopg.connect(
        dbname="resume_db",
        user="postgres",
        password=DB_PASSWORD,
        host="localhost",
        port="5432"
    )


# =========================
# SCHEMA
# =========================
JD_SCHEMA = {
    "job_title": "",
    "department": "",
    "seniority": "",
    "location": "",
    "employment_type": "",
    "required_degree": "",
    "required_major": [],
    "required_cgpa": 0,
    "required_experience_years": 0,
    "required_skills": [],
    "preferred_skills": [],
    "required_certifications": [],
    "preferred_certifications": [],
    "required_languages": [],
    "required_coursework": [],
    "required_projects": [],
    "required_keywords": [],
    "responsibilities": [],
    "consultancy_experience_required": False,
    "mega_project_experience_required": False,
    "donor_project_experience_required": False,
    "weights": {}
}


# =========================
# GEMINI
# =========================
def call_gemini(text: str):
    prompt = f"""
Return ONLY valid JSON following this schema:

{json.dumps(JD_SCHEMA, indent=2)}

JOB DESCRIPTION:
{text}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    return json.loads(raw)


# =========================
# SAVE TO DB (FIXED)
# =========================
def save_to_postgres(jd: dict):
    conn = get_db_connection()
    cur = conn.cursor()
    
    sql = """
    INSERT INTO job_descriptions (
        job_title,
        department,
        seniority,
        location,
        employment_type,
        required_degree,
        required_major,
        required_cgpa,
        required_experience_years,
        required_skills,
        preferred_skills,
        required_certifications,
        preferred_certifications,
        required_languages,
        required_coursework,
        required_projects,
        required_keywords,
        responsibilities,
        consultancy_experience_required,
        mega_project_experience_required,
        donor_project_experience_required,
        weights,
        raw_data,
        jd_hash
    )
    VALUES (
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s,
        %s
    )
    """

    values = (
        
        jd.get("job_title"),
        jd.get("department"),
        jd.get("seniority"),
        jd.get("location"),
        jd.get("employment_type"),

        jd.get("required_degree"),
        json.dumps(jd.get("required_major", [])),
        jd.get("required_cgpa"),
        jd.get("required_experience_years"),

        json.dumps(jd.get("required_skills", [])),
        json.dumps(jd.get("preferred_skills", [])),

        json.dumps(jd.get("required_certifications", [])),
        json.dumps(jd.get("preferred_certifications", [])),

        json.dumps(jd.get("required_languages", [])),
        json.dumps(jd.get("required_coursework", [])),

        json.dumps(jd.get("required_projects", [])),
        json.dumps(jd.get("required_keywords", [])),

        json.dumps(jd.get("responsibilities", [])),

        jd.get("consultancy_experience_required"),
        jd.get("mega_project_experience_required"),
        jd.get("donor_project_experience_required"),

        json.dumps(jd.get("weights", {})),
        json.dumps(jd),
        jd.get("jd_hash")
    )

    try:
        cur.execute(sql, values)
        conn.commit()
        print("\n💾 INSERT SUCCESS")

    except Exception as e:
        print("\n❌ DB ERROR:", e)

    finally:
        cur.close()
        conn.close()


# =========================
# MAIN
# =========================
def parse_job_description(text: str):
    jd = call_gemini(text)
    jd["jd_hash"] = make_hash(text)
    save_to_postgres(jd)
    return jd


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import sys

    print("\n📌 Paste Job Description:\n")
    jd_text = sys.stdin.read()

    print("\n🤖 Processing...\n")

    result = parse_job_description(jd_text)

    print("\n========== RESULT ==========\n")
    print(json.dumps(result, indent=2))