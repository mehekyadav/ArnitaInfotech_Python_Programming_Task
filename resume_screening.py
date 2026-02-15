import os
import string
import nltk
import PyPDF2

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords (only first time)
nltk.download("stopwords")


# ---------------------------
# Extract text from PDF resume
# ---------------------------
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    return text


# ---------------------------
# Preprocess text (clean text)
# ---------------------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]

    return " ".join(words)


# ---------------------------
# Resume Screening Function
# ---------------------------
def resume_screening():
    # Read Job Description
    with open("job_description.txt", "r") as file:
        job_description = file.read()

    job_description = preprocess_text(job_description)

    # Read Resumes
    resume_folder = "resumes"
    resumes = []
    resume_names = []

    for file in os.listdir(resume_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(resume_folder, file)

            resume_text = extract_text_from_pdf(file_path)
            resume_text = preprocess_text(resume_text)

            resumes.append(resume_text)
            resume_names.append(file)

    # Convert text into numbers using TF-IDF
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Cosine Similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # Rank resumes
    ranked_resumes = sorted(
        zip(resume_names, similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Display results
    print("\n📊 Resume Screening Results:\n")
    for rank, (name, score) in enumerate(ranked_resumes, start=1):
        print(f"{rank}. {name} - Match Score: {round(score * 100, 2)}%")


# Run main program
if __name__ == "__main__":
    resume_screening()