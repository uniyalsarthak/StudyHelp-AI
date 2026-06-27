import os
from dotenv import load_dotenv

# from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
DEVICE = os.getenv("DEVICE", "cpu")

# -----------------------------
# Path setup (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # project root

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")

# Ensure vector_db folder exists
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# -----------------------------
# Embedding model
# -----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": DEVICE}
)

# -----------------------------
# Text splitter
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

# -----------------------------
# Vectorize ONE subject
# -----------------------------
def vectorize_subject(subject_name: str):
    subject_path = os.path.join(DATA_DIR, subject_name)

    if not os.path.exists(subject_path):
        print(f"❌ Subject folder not found: {subject_name}")
        return

    print(f"\n📚 Processing subject: {subject_name}")

    loader = DirectoryLoader(
        path=subject_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader

    )

    documents = loader.load()

    if not documents:
        print(f"⚠️ No PDFs found in {subject_name}")
        return

    print(f"📄 Loaded {len(documents)} documents")

    # Split into chunks
    chunks = text_splitter.split_documents(documents)

    # Add metadata (chapter + subject)
    for doc in chunks:
        file_path = doc.metadata.get("source", "")
        file_name = os.path.basename(file_path)
        chapter_name = file_name.replace(".pdf", "").lower()

        doc.metadata["subject"] = subject_name
        doc.metadata["chapter"] = chapter_name

    print(f"✂️ Created {len(chunks)} chunks with metadata")

    # Store in subject-specific DB
    persist_path = os.path.join(VECTOR_DB_DIR, subject_name)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_path
    )

    print(f"✅ {subject_name} stored at: {persist_path}")


# -----------------------------
# Vectorize ALL subjects
# -----------------------------
def vectorize_all_subjects():
    print("🚀 Starting vectorization for all subjects...\n")

    for subject in os.listdir(DATA_DIR):
        subject_path = os.path.join(DATA_DIR, subject)

        if os.path.isdir(subject_path):
            vectorize_subject(subject)

    print("\n🎉 All subjects vectorized successfully!")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":

    # 👉 OPTION 1: Vectorize ALL subjects
    vectorize_all_subjects()

    # 👉 OPTION 2: Only one subject (for testing)
    # vectorize_subject("physics")