import os
from dotenv import load_dotenv
import streamlit as st

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory

from chatbot_utility import get_subjects
from get_ytvid import get_yt_video_link

# -----------------------------
# Load env
# -----------------------------
load_dotenv()
DEVICE = os.getenv("DEVICE", "cpu")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "vector_db")

#ADDED THIS ON MY OWN
from langchain.prompts import PromptTemplate

prompt_template = """
You are a Class 12 expert tutor.

Answer the question ONLY using the provided context.

Guidelines:
- Use professional and clear language
- Explain step-by-step when needed
- For theory → explain in structured form
- If the context is partially relevant, answer related to it if not exact
- When told for set of question and answers or asked to tell most import questions that can be asked, write question and then the answer in detail
- dont answer in too short or too long untill asked. The length of the answer should be according to how much lenght answer user specifies.
- answer in different paragraphs if answer is long.
- suggest some follow up question or related to the user query after answering the question .
- add some formulas in the answer wherever required.
- answer in the style of chatgpt.
- answer structure should be attractive to the user like modern design.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)



# -----------------------------
# Setup chain
# -----------------------------
def setup_chain(subject):
    vector_db_path = os.path.join(VECTOR_DB_DIR, subject)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": DEVICE}
    )

    vectorstore = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    #ADDED ON MY OWN
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 14
        }
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},  # 🔥 THIS LINE
        return_source_documents=True
    )

    return chain


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="StudyHelp", page_icon="📚")

st.title("StudyHelp")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

# if "video_history" not in st.session_state:
#     st.session_state.video_history = {}

if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None

# -----------------------------
# Subject selection
# -----------------------------
subjects = get_subjects()

selected_subject = st.selectbox(
    "Select Subject",
    subjects,
    index=None
)

# Setup chain when subject changes
if selected_subject:
    # Initialize subject chat if not exists
    if selected_subject not in st.session_state.chat_history:
        st.session_state.chat_history[selected_subject] = []

    # Only reset chain (NOT chat history)
    if st.session_state.get("selected_subject") != selected_subject:
        st.session_state.chat_chain = setup_chain(selected_subject)

    st.session_state.selected_subject = selected_subject

# -----------------------------
# Display chat history
# -----------------------------
if selected_subject:
    current_chat = st.session_state.chat_history[selected_subject]

    for idx, message in enumerate(current_chat):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        # if message["role"] == "assistant" and idx < len(st.session_state.video_history):
        #     video_refs = st.session_state.video_history[idx]
        #     if video_refs:
        #         st.subheader("🎥 Video References")
        #         for title, link in video_refs:
        #             st.info(f"{title}\n\n{link}")

# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask your question...")

if user_input and st.session_state.chat_chain and selected_subject:

    # Show user message
    if selected_subject not in st.session_state.chat_history:
        st.session_state.chat_history[selected_subject] = []
    st.session_state.chat_history[selected_subject].append({"role": "user", "content": user_input})
    # st.session_state.video_history.append(None)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        response = st.session_state.chat_chain({"question": user_input})

        answer = response["answer"]
        st.markdown(answer)

        # YouTube suggestions
        # video_titles, video_links = get_yt_video_link(user_input)

        # st.subheader("🎥 Video References")
        # video_refs = []

        # for i in range(len(video_titles)):
        #     st.info(f"{video_titles[i]}\n\n{video_links[i]}")
        #     video_refs.append((video_titles[i], video_links[i]))

        # Save history
        st.session_state.chat_history[selected_subject].append({"role": "assistant", "content": answer})
        # st.session_state.video_history.append(video_refs)