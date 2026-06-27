import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# -----------------------------
# Get available subjects
# -----------------------------
def get_subjects():
    subjects = [
        folder for folder in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, folder))
    ]
    return subjects


# -----------------------------
# Get PDFs inside a subject
# -----------------------------
def get_subject_files(subject):
    subject_path = os.path.join(DATA_DIR, subject)

    if not os.path.exists(subject_path):
        return []

    files = [f for f in os.listdir(subject_path) if f.endswith(".pdf")]
    return files