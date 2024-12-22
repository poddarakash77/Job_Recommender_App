import spacy
from spacy.matcher import Matcher
import PyPDF2
import os
import csv
from spacy.cli import download

def load_spacy_model(model_name="en_core_web_sm"):
    try:
        # Try loading the model
        nlp = spacy.load(model_name)
    except OSError:
        # If the model is not found, download it
        print(f"Model '{model_name}' not found. Downloading now...")
        download(model_name)
        nlp = spacy.load(model_name)  # Load after downloading
    return nlp

# Load the model using the function
nlp = load_spacy_model()

# Read skills from CSV file
file_path=r'C:\Users\podda\PycharmProjects\FINAL_PROJECT_NLU\skills.csv'
with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    skills = [row for row in csv_reader]

# Create pattern dictionaries from skills
skill_patterns = [[{'LOWER': skill}] for skill in skills[0]]

# Create a Matcher object
matcher = Matcher(nlp.vocab)

# Add skill patterns to the matcher
for pattern in skill_patterns:
    matcher.add('Skills', [pattern])

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    matches = matcher(doc)
    skills = set()
    for match_id, start, end in matches:
        skill = doc[start:end].text
        skills.add(skill)
    return skills

# Function to extract text from PDF
def extract_text_from_pdf(file_path:str):
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def skills_extractor(file_path):
        # Extract text from PDF
        # path=r'Job-Recommendation-System\src\components'
        # full_file_path = os.path.join(path, file_path)
        resume_text = extract_text_from_pdf(file_path)

        # Extract skills from resume text
        skills = list(extract_skills(resume_text))
        return skills


