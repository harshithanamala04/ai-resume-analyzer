import streamlit as st
import pandas as pd
import pypdf
import google.generativeai as genai
import json
import os
import re
from dotenv import load_dotenv
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(
    page_title="AI Resume & ATS Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD API KEY (OPTIONAL) ---
load_dotenv()
env_api_key = os.getenv("GEMINI_API_KEY", "")

# Safely support Streamlit Secrets for cloud deployment without crashing locally
if not env_api_key:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            env_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

# --- MINIMAL, CLEAN & MODERN CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App Canvas */
.stApp {
    background-color: #0f172a;
    color: #f8fafc;
}

/* Clean Header */
.app-header {
    padding: 10px 0 24px 0;
    border-bottom: 1px solid #334155;
    margin-bottom: 24px;
}
.app-title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px 0;
}
.app-subtitle {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.5;
}

/* Engine Badge */
.engine-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59, 130, 246, 0.12);
    border: 1px solid rgba(59, 130, 246, 0.3);
    color: #60a5fa;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 14px;
}

/* Metric / Stat Cards */
.stat-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.stat-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.stat-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.stat-sub {
    font-size: 0.78rem;
    margin-top: 6px;
    font-weight: 500;
}

/* Skill Container & Badges */
.skill-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    min-height: 200px;
}
.skill-card-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 0.86rem;
    font-weight: 500;
    letter-spacing: 0.2px;
}
.badge-matched {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.badge-missing {
    background: rgba(244, 63, 94, 0.12);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.3);
}

/* Suggestion Card */
.suggestion-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.suggestion-badge {
    background: #3b82f6;
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 700;
    min-width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 2px;
}
.suggestion-text {
    font-size: 0.95rem;
    color: #e2e8f0;
    line-height: 1.5;
    margin: 0;
}

/* Primary Button */
div.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
}

/* Cleaner text area and uploader borders */
.stTextArea textarea, .stFileUploader {
    border-radius: 8px !important;
}

/* Completely hide sidebar and collapse toggle */
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

#MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- ENGINE CONFIGURATION ---
# Uses Gemini API Key if defined in .env / secrets; otherwise defaults to Built-in ATS Engine
api_key_to_use = env_api_key.strip() if env_api_key else ""
model_choice = "gemini-1.5-flash" if api_key_to_use else "Built-in ATS"

# --- HEADER ---
st.markdown("""
<div class="app-header">
    <h1 class="app-title">🎯 AI Resume & ATS Analyzer</h1>
    <p class="app-subtitle">
        Accurately evaluate your resume against target job requirements. Detect missing skills, calculate objective ATS compatibility, and receive precise recruiter recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

# --- PDF TEXT EXTRACTION ---
def extract_text_from_pdf(uploaded_file):
    """Accurately extracts text from all pages of an uploaded PDF."""
    try:
        reader = pypdf.PdfReader(uploaded_file)
        extracted = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted.append(text)
        return "\n\n".join(extracted).strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# --- COMPREHENSIVE SKILL TAXONOMY FOR LOCAL ENGINE ---
COMMON_SKILLS_TAXONOMY = [
    # Languages
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Golang", "Rust", "Ruby", "PHP", 
    "Swift", "Kotlin", "SQL", "HTML", "CSS", "Bash", "Shell", "R", "Scala", "Dart",
    # AI / ML / LLMs / Agentic Systems
    "AI Agent", "AI Agents", "Agentic AI", "Agentic Workflows", "LLM", "LLMs", "Machine Learning", 
    "Deep Learning", "NLP", "Natural Language Processing", "Computer Vision", "LangChain", "LlamaIndex", 
    "RAG", "Retrieval-Augmented Generation", "PyTorch", "TensorFlow", "Keras", "Scikit-Learn", "Pandas", 
    "NumPy", "Hugging Face", "OpenAI", "Prompt Engineering", "Fine-tuning", "Vector Database", 
    "Pinecone", "ChromaDB", "Weaviate", "Milvus", "pgvector", "Embedding",
    # Web & Full Stack
    "React", "React.js", "Next.js", "Vue", "Vue.js", "Angular", "Svelte", "Node.js", "Express", 
    "Express.js", "FastAPI", "Django", "Flask", "Spring Boot", "ASP.NET", "GraphQL", "REST API", 
    "RESTful APIs", "REST APIs", "WebSockets", "TailwindCSS", "Bootstrap", "Redux", "API Development",
    # Cloud & DevOps
    "AWS", "Amazon Web Services", "GCP", "Google Cloud", "Azure", "Docker", "Kubernetes", "K8s", 
    "CI/CD", "CI/CD Pipeline", "Dev Pipeline", "DevOps", "GitHub Actions", "GitLab CI", "Terraform", 
    "Ansible", "Linux", "Nginx", "Microservices", "Serverless", "AWS Lambda", "EC2", "S3",
    # Databases & Caching
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Cassandra", "DynamoDB", "SQLite", "Elasticsearch", 
    "Firebase", "Supabase", "Kafka", "RabbitMQ", "Message Queues",
    # Practices, Testing & Domain Workflows
    "Git", "GitHub", "GitLab", "Testing", "Unit Testing", "PyTest", "Jest", "Selenium", "Cypress", 
    "System Design", "Agile", "Scrum", "Automation", "Workflow Automation", "CRM", "Lead Followup",
    "Inbound Leads", "WhatsApp", "IndiaMART", "API Integration", "Asynchronous Programming"
]

# Common words to exclude when extracting custom domain phrases
STOP_WORDS = {
    'role', 'overview', 'we', 'our', 'you', 'the', 'as', 'an', 'in', 'on', 'for', 'and', 'with', 'about', 
    'requirements', 'responsibilities', 'qualifications', 'looking', 'seeking', 'work', 'team', 'must', 
    'plus', 'preferred', 'required', 'ideal', 'candidate', 'join', 'help', 'build', 'ship', 'own', 'run', 
    'days', 'every', 'day', 'years', 'level', 'senior', 'junior', 'lead', 'mid', 'they', 'them', 'their',
    'this', 'that', 'from', 'have', 'has', 'will', 'who', 'what', 'where', 'when', 'why', 'how', 'same',
    'puts', 'deal', 'slow', 'demo', 'harness', 'proves', 'hold'
}

# --- ENGINE 1: LOCAL ALGORITHMIC ATS ENGINE (NO API KEY REQUIRED) ---
def analyze_resume_locally(resume_text, job_description):
    """
    Evaluates candidate resume against job description using accurate algorithmic keyword
    and domain competency matching. Works 100% offline without any API key.
    """
    # 1. Detect standard skills in Job Description
    jd_skills = []
    for skill in COMMON_SKILLS_TAXONOMY:
        pattern = r'\b' + re.escape(skill) + r'(?:s|es)?\b'
        if re.search(pattern, job_description, re.IGNORECASE):
            if skill not in jd_skills:
                jd_skills.append(skill)

    # 2. Extract capitalized custom domain terms / phrases from Job Description
    custom_candidates = re.findall(r'\b[A-Z][a-zA-Z0-9\.\-_]{2,}(?:\s+[A-Z][a-zA-Z0-9\.\-_]+)*\b', job_description)
    for phrase in custom_candidates:
        phrase_clean = phrase.strip()
        if phrase_clean.lower() not in STOP_WORDS and len(phrase_clean) > 2:
            if not any(phrase_clean.lower() == s.lower() for s in jd_skills):
                jd_skills.append(phrase_clean)

    # Keep the most relevant skills (up to 16)
    if not jd_skills:
        # Fallback to key technical words found in JD
        words = re.findall(r'\b[A-Za-z]{4,}\b', job_description)
        jd_skills = [w.capitalize() for w in words if w.lower() not in STOP_WORDS][:10]

    jd_skills = jd_skills[:16]

    # 3. Match detected skills against candidate resume
    matched_skills = []
    missing_skills = []

    for skill in jd_skills:
        pattern = r'\b' + re.escape(skill) + r'(?:s|es)?\b'
        if re.search(pattern, resume_text, re.IGNORECASE):
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    # 4. Calculate realistic ATS score
    if jd_skills:
        skill_match_ratio = len(matched_skills) / len(jd_skills)
    else:
        skill_match_ratio = 0.5

    # Overall vocabulary overlap (Jaccard similarity on content tokens)
    def extract_tokens(text):
        tokens = set(re.findall(r'[a-zA-Z]{3,}', text.lower()))
        return {t for t in tokens if t not in STOP_WORDS}

    jd_tokens = extract_tokens(job_description)
    resume_tokens = extract_tokens(resume_text)
    overlap_ratio = len(jd_tokens & resume_tokens) / max(len(jd_tokens), 1)

    # Weighted calculation: 75% skill match + 25% broader vocabulary alignment
    raw_score = int(round((0.75 * skill_match_ratio + 0.25 * overlap_ratio) * 100))
    calculated_score = max(min(raw_score, 96), 15)

    if calculated_score >= 75:
        verdict = "Strong Match"
    elif calculated_score >= 50:
        verdict = "Moderate Alignment"
    else:
        verdict = "Needs Keyword Optimization"

    # 5. Build tailored, actionable suggestions
    suggestions = []
    if missing_skills:
        top_missing = missing_skills[:3]
        suggestions.append(
            f"Add target keywords: The job posting explicitly highlights **{', '.join(top_missing)}**, which were not detected in your resume. Integrate these in your skills section or project bullets."
        )
        if len(missing_skills) > 3:
            suggestions.append(
                f"Demonstrate experience with **{missing_skills[3]}**: Reframe a recent project bullet point to detail how you applied or built workflows with this technology."
            )

    if matched_skills:
        suggestions.append(
            f"Quantify matched skills: For verified proficiencies like **{', '.join(matched_skills[:2])}**, include concrete metrics (e.g. latency reduction %, user scale, or testing coverage) to strengthen impact."
        )

    suggestions.append(
        "Mirror job description terminology: Use the exact technical phrases and action verbs from the posting (e.g. 'ship agents', 'dev pipeline') rather than generic synonyms to pass ATS automated filters."
    )
    suggestions.append(
        "Standardize resume headings: Ensure clear, standard section titles (*Technical Skills*, *Professional Experience*, *Projects*, *Education*) to ensure 100% parser readability."
    )

    return {
        "ats_score": calculated_score,
        "verdict": verdict,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "improvement_suggestions": suggestions[:4]
    }

# --- ENGINE 2: GEMINI AI ENGINE (WHEN API KEY IS PROVIDED) ---
def analyze_resume_with_gemini(resume_text, job_description, api_key, model_name):
    """
    Calls Gemini API with strict evaluation rules to ensure fact-based, objective results.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    prompt = f"""
    You are a meticulous, expert Applicant Tracking System (ATS) auditor and senior technical recruiter.
    Evaluate the candidate's resume strictly and objectively against the target job description.

    ### CANDIDATE RESUME:
    \"\"\"{resume_text}\"\"\"

    ### TARGET JOB DESCRIPTION:
    \"\"\"{job_description}\"\"\"

    ### STRICT ACCURACY & EVALUATION RULES:
    1. FACTUAL GROUNDING: Do NOT assume or invent skills. 
       - "matched_skills": Include ONLY skills, technologies, qualifications, or tools that are EXPLICITLY present in the candidate's resume AND requested/preferred by the job description.
       - "missing_skills": Include ONLY skills, technologies, certifications, or qualifications that are EXPLICITLY required or desired in the job description but are NOT clearly found in the resume.
    2. OBJECTIVE ATS MATCH SCORE (0 to 100):
       - Calculate the score mathematically:
         Score = (Number of matched essential & preferred qualifications / Total required & preferred qualifications) * 100, weighted by required seniority depth.
       - 80-100: Exceptional fit; candidate meets almost all required and preferred criteria.
       - 60-79: Solid match; meets major core criteria, but lacks a few specific tools or domain depth.
       - 40-59: Partial match; foundational skills present, but misses multiple critical prerequisites.
       - 0-39: Poor match; significant qualification and technology misalignment.
       - Provide a calibrated, realistic score. Avoid inflating scores.
    3. ACTIONABLE RECOMMENDATIONS:
       - Provide 3 to 5 clear, concrete, and non-generic bullet points.
       - Explain exactly what terms to incorporate, where to emphasize experience, and how to quantify bullet points for maximum ATS ranking.

    Output your response STRICTLY as a raw, valid JSON object matching this schema without any markdown formatting, code fences, or additional text:
    {{
      "ats_score": <integer between 0 and 100>,
      "verdict": "<Short 2-3 word assessment: e.g. Strong Match / Moderate Alignment / Needs Work / Low Match>",
      "matched_skills": ["<matched skill 1>", "<matched skill 2>"],
      "missing_skills": ["<missing skill 1>", "<missing skill 2>"],
      "improvement_suggestions": [
        "<specific, actionable suggestion 1>",
        "<specific, actionable suggestion 2>",
        "<specific, actionable suggestion 3>"
      ]
    }}
    """
    
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
        
    return json.loads(raw)

# --- INPUT WORKBENCH (2 CLEAN COLUMNS) ---
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("📄 1. Candidate Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF format)",
        type=["pdf"],
        help="Upload the PDF resume to evaluate."
    )
    
    # Optional text fallback
    with st.expander("Or enter / view resume text directly"):
        pasted_resume = st.text_area(
            "Paste resume text",
            height=200,
            placeholder="Alternatively, paste plain resume text here...",
            label_visibility="collapsed"
        )

with col_right:
    st.subheader("💼 2. Job Description")
    job_desc = st.text_area(
        "Target Job Description",
        height=260,
        placeholder="Paste the target job description, responsibilities, and required qualifications here...",
        label_visibility="collapsed"
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# --- ACTION BUTTON ---
analyze_clicked = st.button("🔍 Analyze ATS Match", type="primary", use_container_width=True)

# --- PROCESSING & RESULTS ---
if analyze_clicked:
    # 1. Resolve Resume Text
    resume_text = ""
    if uploaded_file is not None:
        with st.spinner("Extracting text from uploaded PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text or len(resume_text) < 30:
                st.warning("⚠️ Little or no text could be extracted from this PDF. If it is a scanned image, please paste the text directly.")
    elif pasted_resume.strip():
        resume_text = pasted_resume.strip()

    # 2. Input Validations
    if not resume_text:
        st.warning("Please upload a PDF resume or paste resume text before analyzing.")
    elif not job_desc.strip():
        st.warning("Please paste the target job description before analyzing.")
    else:
        # 3. Perform Analysis (Gemini if key is available, Local Engine otherwise)
        results = None
        engine_label = ""

        if api_key_to_use:
            with st.spinner(f"Analyzing with Gemini AI ({model_choice})..."):
                try:
                    results = analyze_resume_with_gemini(resume_text, job_desc.strip(), api_key_to_use, model_choice)
                    engine_label = f"✨ Evaluated with Gemini AI ({model_choice})"
                except Exception as e:
                    st.warning(f"⚠️ Gemini API issue: {e}. Falling back to Built-in ATS Engine...")
                    results = analyze_resume_locally(resume_text, job_desc.strip())
                    engine_label = "⚡ Evaluated with Built-in Algorithmic ATS Engine"
        else:
            with st.spinner("Analyzing resume against job requirements with Built-in ATS Engine..."):
                results = analyze_resume_locally(resume_text, job_desc.strip())
                engine_label = "⚡ Evaluated with Built-in Algorithmic ATS Engine (No API Key Required)"

        if results:
            st.session_state["results"] = results
            st.session_state["extracted_text"] = resume_text
            st.session_state["engine_label"] = engine_label
            st.success("Analysis complete!")

# --- DISPLAY RESULTS ---
if "results" in st.session_state:
    data = st.session_state["results"]
    score = int(data.get("ats_score", 0))
    verdict = data.get("verdict", "Analysis Complete")
    matched = data.get("matched_skills", [])
    missing = data.get("missing_skills", [])
    suggestions = data.get("improvement_suggestions", [])
    engine_label = st.session_state.get("engine_label", "ATS Engine")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.divider()
    
    # Engine & Title Row
    header_col1, header_col2 = st.columns([2, 1])
    with header_col1:
        st.subheader("📊 Evaluation Results")
    with header_col2:
        st.markdown(f"""
            <div style="text-align: right;">
                <span class="engine-badge">{engine_label}</span>
            </div>
        """, unsafe_allow_html=True)

    # Row 1: KPI Stat Cards & Donut Chart
    score_col, stat1_col, stat2_col, chart_col = st.columns([1.2, 1, 1, 1.3], gap="medium")

    # Score color logic
    if score >= 75:
        score_color = "#34d399" # Emerald
    elif score >= 50:
        score_color = "#fbbf24" # Amber
    else:
        score_color = "#f87171" # Rose

    with score_col:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">ATS Match Score</div>
                <div class="stat-value" style="color: {score_color};">{score}%</div>
                <div class="stat-sub" style="color: {score_color};">{verdict}</div>
            </div>
        """, unsafe_allow_html=True)

    with stat1_col:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Matched Skills</div>
                <div class="stat-value" style="color: #34d399;">{len(matched)}</div>
                <div class="stat-sub" style="color: #94a3b8;">Found in resume</div>
            </div>
        """, unsafe_allow_html=True)

    with stat2_col:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Missing Skills</div>
                <div class="stat-value" style="color: #fb7185;">{len(missing)}</div>
                <div class="stat-sub" style="color: #94a3b8;">Gaps to address</div>
            </div>
        """, unsafe_allow_html=True)

    with chart_col:
        total = len(matched) + len(missing)
        if total > 0:
            fig = go.Figure(data=[go.Pie(
                labels=['Matched', 'Missing'],
                values=[len(matched), len(missing)],
                hole=0.6,
                marker_colors=['#10b981', '#f43f5e'],
                textinfo='percent',
                hoverinfo='label+value'
            )])
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=130,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5,
                    font=dict(color="#94a3b8", size=11)
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Row 2: Skills Breakdown (Clean Side-by-Side Cards)
    match_col, miss_col = st.columns(2, gap="large")

    with match_col:
        matched_badges = "".join([f'<span class="badge badge-matched">✓ {s}</span>' for s in matched]) if matched else '<p style="color:#94a3b8; font-size:0.9rem;">No matching skills identified.</p>'
        st.markdown(f"""
            <div class="skill-card">
                <div class="skill-card-title">
                    <span style="color:#34d399;">✅ Matched Qualifications ({len(matched)})</span>
                </div>
                <div class="badge-container">
                    {matched_badges}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with miss_col:
        missing_badges = "".join([f'<span class="badge badge-missing">✕ {s}</span>' for s in missing]) if missing else '<p style="color:#34d399; font-size:0.9rem;">No critical skill gaps identified!</p>'
        st.markdown(f"""
            <div class="skill-card">
                <div class="skill-card-title">
                    <span style="color:#fb7185;">⚠️ Missing or Weak Skills ({len(missing)})</span>
                </div>
                <div class="badge-container">
                    {missing_badges}
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Row 3: Actionable Recommendations
    st.subheader("💡 Actionable Improvement Suggestions")
    if suggestions:
        for idx, item in enumerate(suggestions, 1):
            st.markdown(f"""
                <div class="suggestion-box">
                    <div class="suggestion-badge">{idx}</div>
                    <p class="suggestion-text">{item}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.write("No specific suggestions provided.")

    # Optional: Extracted Text Verification
    if "extracted_text" in st.session_state and st.session_state["extracted_text"]:
        with st.expander("🔍 View Raw Extracted Resume Text (Transparency Check)"):
            st.text(st.session_state["extracted_text"])

    # Row 4: Clean Export
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    json_data = json.dumps(data, indent=2)
    st.download_button(
        label="📥 Download Analysis Report (JSON)",
        data=json_data,
        file_name="ats_evaluation_report.json",
        mime="application/json"
    )