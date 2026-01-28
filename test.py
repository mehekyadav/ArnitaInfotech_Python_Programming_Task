import os
import nltk
import string
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords (run once)
nltk.download('stopwords')
from nltk.corpus import stopwords

# -------------------------------
# Function to extract text from PDF
# -------------------------------
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# -------------------------------
# Text preprocessing function
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# -------------------------------
# Load Job Description
# -------------------------------
with open("job_description.txt", "r") as jd_file:
    job_description = jd_file.read()

job_description = preprocess_text(job_description)

# -------------------------------
# Load and process resumes
# -------------------------------
resume_folder = "resumes"
resumes = []
resume_names = []

for file in os.listdir(resume_folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(resume_folder, file)
        text = extract_text_from_pdf(file_path)
        text = preprocess_text(text)
        resumes.append(text)
        resume_names.append(file)

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
documents = [job_description] + resumes
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# -------------------------------
# Cosine Similarity Calculation
# -------------------------------
similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

# -------------------------------
# Ranking Resumes
# -------------------------------
ranked_resumes = sorted(
    zip(resume_names, similarity_scores),
    key=lambda x: x[1],
    reverse=True
)

# -------------------------------
# Display Results
# -------------------------------
print("\n📊 Resume Screening Results:\n")
for rank, (name, score) in enumerate(ranked_resumes, start=1):
    print(f"{rank}. {name} - Match Score: {round(score*100, 2)}%")
