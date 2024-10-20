import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import PyPDF2 as pdf
import json

load_dotenv()

groq_api_key = os.getenv("Groq_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-70b-8192")

template = """
Act as a highly skilled ATS (Applicant Tracking System) with expertise in tech fields. Evaluate the resume based on the given job description, considering technical skills, experience, education, soft skills, and project relevance.

Resume: {resume_text}
Job Description: {job_description}

Provide a detailed analysis in the following JSON format:
{{
  "JD Match": "percentage",
  "MissingKeywords": ["keyword1", "keyword2", ...],
  "Profile Summary": "brief summary",
  "Strengths": ["strength1", "strength2", ...],
  "Areas for Improvement": ["area1", "area2", ...],
  "Recommendations": ["recommendation1", "recommendation2", ...]
}}
"""

prompt = PromptTemplate(template=template, input_variables=["resume_text", "job_description"])
chain = LLMChain(llm=llm, prompt=prompt)

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

st.title("ATS Resume Evaluator")
st.text("Improve Your Resume ATS")

job_description = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the pdf")


def extract_json_from_response(response):
    try:
        
        start = response.find('{')
        
        end = response.rfind('}')
        
        if start != -1 and end != -1:
            json_str = response[start:end+1]
            return json.loads(json_str)
        else:
            return {"error": "No valid JSON found in the response"}
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON from the response"}

if st.button("Submit"):
    if uploaded_file is not None and job_description:
        resume_text = input_pdf_text(uploaded_file)
        response = chain.run(resume_text=resume_text, job_description=job_description)
        parsed_response = extract_json_from_response(response)
        st.json(parsed_response)
    else:
        st.warning("Please upload a resume and provide a job description.")
