from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Resolve project root folder (ChatBot/) so .env and vectorstore_careflow are always found
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name.lower() == "frontend" else CURRENT_DIR

load_dotenv(ROOT_DIR / ".env")

# 2. Page configuration
st.set_page_config(
    page_title="CAREFLOW AI Policy Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 CAREFLOW AI Policy Assistant")
st.caption("Ask questions about CAREFLOW AI platform rules, user roles, subscriptions, and policies.")

# 3. Load and cache the embedding model, FAISS database, and Gemini LLM
@st.cache_resource
def load_rag_pipeline():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        str(ROOT_DIR / "vectorstore_careflow"),
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    return retriever, llm

retriever, llm = load_rag_pipeline()

# 4. Retrieve policy chunks and generate a grounded answer
def ask_policy_bot(question: str):
    documents = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in documents])

    final_prompt = f"""
    You are a helpful CAREFLOW AI platform and operational policy assistant.

    Answer the user's question using ONLY the provided context from the CAREFLOW AI handbook.
    Be clear, accurate, and cite exact plan limits, roles, or rules when present.

    If the answer is not available in the context, say:
    "I don't know based on the provided CAREFLOW AI policy document."

    Context:
    {context}

    Question: {question}
    """

    response = llm.invoke(final_prompt)
    answer = response.text if getattr(response, "text", None) else response.content
    return answer, documents

# 5. Sidebar controls and sample questions
with st.sidebar:
    st.header("Chat Controls")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today using the CAREFLOW AI policy document?"}
        ]
        st.rerun()

    st.divider()
    st.subheader("Sample Questions")
    sample_questions = [
        "What are the five user roles in CAREFLOW AI?",
        "How many monthly AI credits does each plan include?",
        "What is the 6-month clinic reverification rule?",
        "What are the three RAG knowledge domains?"
    ]
    selected_sample = None
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            selected_sample = q

# 6. Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today using the CAREFLOW AI policy document?"}
    ]

# 7. Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 8. Handle new user input (typed in input box OR clicked in sidebar)
user_question = st.chat_input("Ask a question about CAREFLOW AI...") or selected_sample

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy handbook..."):
            answer, docs = ask_policy_bot(user_question)
            st.markdown(answer)

            with st.expander("📄 View Retrieved Policy Sources"):
                for i, doc in enumerate(docs, start=1):
                    page_num = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Chunk {i} — Page {page_num}:**")
                    st.caption(doc.page_content[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})