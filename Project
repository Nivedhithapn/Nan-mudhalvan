import os
import re
import docx2txt
import spacy
import pandas as pd
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# === TEXT EXTRACTION ===
def extract_text_from_file(file_path):
    if file_path.endswith('.pdf'):
        return extract_text(file_path)
    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path)
    else:
        raise ValueError("Unsupported file format")

# === DATA EXTRACTION ===
def extract_email(text):
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)

def extract_phone(text):
    return re.findall(r"\b\d{10}\b", text)

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Name Not Found"

def extract_skills(text, skills_list):
    text = text.lower()
    return list({skill for skill in skills_list if skill.lower() in text})

# === JOB MATCHING ===
def compute_similarity(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# === MAIN SCREENING FUNCTION ===
def screen_resumes(resume_folder, job_description, skills_list):
    candidates = []

    for file in os.listdir(resume_folder):
        if file.endswith(('.pdf', '.docx')):
            file_path = os.path.join(resume_folder, file)
            try:
                text = extract_text_from_file(file_path)

                name = extract_name(text)
                email = extract_email(text)
                phone = extract_phone(text)
                skills = extract_skills(text, skills_list)
                score = compute_similarity(text, job_description)

                candidates.append({
                    'Name': name,
                    'Email': email[0] if email else '',
                    'Phone': phone[0] if phone else '',
                    'Skills': ', '.join(skills),
                    'Score': round(score, 2),
                    'File': file
                })

            except Exception as e:
                print(f"Error processing {file}: {e}")

    df = pd.DataFrame(candidates)
    df = df.sort_values(by='Score', ascending=False)
    return df

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    job_description = """
    We are looking for a Python developer with experience in machine learning, 
    data analysis, and RESTful APIs. Knowledge of Django or Flask is a plus.
    """

    skills_list = ["Python", "Machine Learning", "Data Analysis", "Flask", "Django", "REST API"]
    resume_folder = "resumes/"  # Place your PDF/DOCX files in this folder

    result_df = screen_resumes(resume_folder, job_description, skills_list)
    print(result_df)

    # Optional: Export results
    result_df.to_csv("screened_candidates.csv", index=False)
