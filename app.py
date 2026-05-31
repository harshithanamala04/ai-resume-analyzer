import streamlit as st
import pandas as pd
import pypdf
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import plotly.graph_objects as go

# Load API Key from the .env file
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to extract text from the uploaded PDF resume
def extract_text_from_pdf(uploaded_file):
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to analyze resume text against job description using Gemini AI
def analyze_resume(resume_text, job_description):
    # Initialize the fast, multimodal model optimized for text extraction
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Carefully engineered prompt with strict architectural output constraints
    prompt = f"""
    You are an expert ATS (Applicant Tracking System) optimizer and technical recruiter.
    Analyze the following Resume against the Job Description.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Provide your analysis STRICTLY in the following JSON format. Do not include any markdown formatting like ```json or extra text outside the raw JSON block.
    {{
        "ats_score": <an integer between 0 and 100>,
        "matched_skills": [<list of strings of matching skills found>],
        "missing_skills": [<list of critical skills missing or weak>],
        "improvement_suggestions": [<list of specific, actionable suggestions>]
    }}
    """
    
    # Request generation with an enforced application/json mime-type configuration
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    
    # Parse the raw text string directly into an accessible Python dictionary object
    return json.loads(response.text)

# --- USER INTERFACE DESIGN ---
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("🤖 AI-Powered Resume Analyzer")
st.subheader("Optimize your resume against target Job Descriptions")

# Split screen into two columns for input
col1, col2 = st.columns(2)

with col1:
    st.header("📄 Upload Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])

with col2:
    st.header("💼 Job Description")
    job_desc = st.text_area("Paste the job description here", height=200)

# Main execution processing block
if st.button("🚀 Analyze Resume"):
    if uploaded_file is not None and job_desc:
        with st.spinner("Analyzing your resume against the job requirements..."):
            try:
                # 1. Process local PDF into string data
                resume_text = extract_text_from_pdf(uploaded_file)
                
                # 2. Invoke remote inference call to Gemini AI
                analysis_results = analyze_resume(resume_text, job_desc)
                
                st.success("Analysis Complete!")
                st.divider()
                
                # --- DASHBOARD METRIC DISPLAY ---
                st.header("📊 Evaluation Dashboard")

                # Split the dashboard header row into two parts: Metric and Chart
                metric_col, chart_col = st.columns([1, 2]) # 1:2 ratio layout

                # Extract parsed data components early to prevent scoping issues
                score = analysis_results.get("ats_score", 0)
                matched = analysis_results.get("matched_skills", [])
                missing = analysis_results.get("missing_skills", [])

                with metric_col:
                    if score >= 75:
                        st.metric(label="Estimated ATS Match Score", value=f"{score}%", delta="Strong Match")
                    elif score >= 50:
                        st.metric(label="Estimated ATS Match Score", value=f"{score}%", delta="Needs Improvement", delta_color="off")
                    else:
                        st.metric(label="Estimated ATS Match Score", value=f"{score}%", delta="- Critical Gaps", delta_color="inverse")

                with chart_col:
                    # Generate a dynamic donut chart using the lengths of our skill arrays
                    total_matched = len(matched)
                    total_missing = len(missing)

                    if total_matched > 0 or total_missing > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=['Matched Skills', 'Missing Skills'],
                            values=[total_matched, total_missing],
                            hole=.5, # This creates the middle "donut" hole
                            marker_colors=['#2ecc71', '#e74c3c'] # Green and Red styling
                        )])

                        fig.update_layout(
                            margin=dict(t=0, b=0, l=0, r=0),
                            height=180,
                            showlegend=True
                        )

                        # Display the Plotly chart inside our Streamlit interface
                        st.plotly_chart(fig, use_container_width=True)
                
                # --- PANDAS DATA VISUALIZATION COLUMNS ---
                dash_col1, dash_col2 = st.columns(2)
                
                with dash_col1:
                    st.subheader("✅ Matched Skills")
                    if matched:
                        # Feed array directly into tabular Pandas Dataframe formats
                        df_matched = pd.DataFrame(matched, columns=["Skill Name"])
                        st.dataframe(df_matched, use_container_width=True)
                    else:
                        st.write("No matching skills found.")
                        
                with dash_col2:
                    st.subheader("⚠️ Missing Skills / Skill Gaps")
                    if missing:
                        df_missing = pd.DataFrame(missing, columns=["Skill to Add"])
                        st.dataframe(df_missing, use_container_width=True)
                    else:
                        st.write("No major skill gaps identified.")
                
                # --- RECRUITER INSIGHTS SECTION ---
                st.subheader("💡 Actionable Improvement Suggestions")
                suggestions = analysis_results.get("improvement_suggestions", [])
                for idx, suggestion in enumerate(suggestions, 1):
                    st.markdown(f"**{idx}.** {suggestion}")
                    
            except Exception as e:
                st.error(f"A processing anomaly occurred: {e}")
    else:
        st.warning("Please ensure both a resume PDF is uploaded and a job description is provided.")