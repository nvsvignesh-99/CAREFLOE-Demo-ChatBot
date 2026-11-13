# 🏥 CAREFLOW AI — Policy & Operations RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **FAISS**, **Hugging Face Embeddings**, **Google Gemini**, and **Streamlit**. The assistant answers platform, operational, subscription, and clinical governance questions strictly grounded in the **CAREFLOW AI Platform & Operational Policy Handbook**.

---

## 📌 Overview

Traditional LLMs can hallucinate policy details or lack knowledge of private product documentation. This application implements a complete end-to-end RAG pipeline that:
1. Extracts and chunks the 6-page **CAREFLOW AI Platform & Operational Policy Handbook** PDF.
2. Generates dense vector embeddings locally using `sentence-transformers/all-MiniLM-L6-v2` and stores them in a persistent **FAISS** vector index.
3. Retrieves the top-4 most relevant policy sections for any user query and synthesizes accurate, citation-backed answers using **Google Gemini (`gemini-3.5-flash-lite`)**.

---

## ✨ Key Features

- **Grounded Policy Q&A:** Strictly answers from the provided CAREFLOW AI handbook and refuses out-of-scope queries (*"I don't know based on the provided CAREFLOW AI policy document"*).
- **Local Vector Indexing (FAISS):** Fast semantic similarity search (`k=4`) with zero embedding API quota cost.
- **Interactive Streamlit Web UI (`Frontend/app.py`):**
  - Clean conversational chat interface with persistent session history.
  - One-click sample policy questions and chat reset controls in the sidebar.
  - Expandable **Retrieved Policy Sources** viewer displaying exact PDF page numbers and matched text chunks.
  - Resource caching (`@st.cache_resource`) for instant response times after initial model load.
- **Terminal CLI Mode (`chatbot.py`):** Lightweight command-line chatbot loop for rapid local testing.

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                       1. OFFLINE INGESTION (ingest.py)                  │
│                                                                         │
│  Policy PDF ──▶ PyPDFLoader ──▶ RecursiveCharacterTextSplitter         │
│                                  (chunk_size=1000, overlap=200)         │
│                                          │                              │
│                                          ▼                              │
│  FAISS Index (vectorstore_careflow/) ◀── HuggingFaceEmbeddings          │
│                                          (all-MiniLM-L6-v2)             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   2. ONLINE RAG QUERY (Frontend/app.py)                 │
│                                                                         │
│  User Question ──▶ FAISS Retriever (Top-4 Chunks) ──▶ Grounded Prompt  │
│                                                              │          │
│                                                              ▼          │
│  Streamlit UI + Page Citations ◀── Google Gemini (gemini-3.5-flash-lite)│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | [LangChain](https://www.langchain.com/) (`langchain`, `langchain-community`) | Document loading, text splitting, retrieval, and LLM invocation |
| **LLM Engine** | Google Gemini (`gemini-3.5-flash-lite` via `langchain-google-genai`) | Context-grounded answer synthesis |
| **Embeddings** | Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) | Local dense vector generation for chunks and queries |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) | Local similarity search and persistent vector storage |
| **PDF Parser** | `pypdf` (`PyPDFLoader`) | Multi-page PDF text and page metadata extraction |
| **Web Frontend** | [Streamlit](https://streamlit.io/) | Interactive browser chat UI, sidebar shortcuts, and source expanders |
| **Configuration** | `python-dotenv` | Environment variable management for API keys |

---

## 📂 Project Structure

```text
ChatBot/
├── data/
│   └── CAREFLOW AI Platform & Operational Policy Handbook.pdf   # Source knowledge base (6 pages)
├── Frontend/
│   └── app.py                  # Streamlit web chat interface with source citations
├── vectorstore_careflow/       # Saved FAISS vector index (index.faiss & index.pkl)
├── ingest.py                   # PDF loader, chunker, embedder, and FAISS builder
├── chatbot.py                  # Terminal-based CLI RAG chatbot
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Excludes .env and Python cache files from Git
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python **3.10+** installed
- A free **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)

### 2. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/careflow-ai-policy-chatbot.git](https://github.com/YOUR_USERNAME/careflow-ai-policy-chatbot.git)
cd careflow-ai-policy-chatbot
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Gemini API key:
```env
GOOGLE_API_KEY=your_google_ai_studio_api_key_here
```

---

## 💻 Usage

### Step 1: Build the Vector Database (Run Once or When PDF Changes)
Loads the policy PDF, splits it into overlapping 1,000-character chunks, generates embeddings, and saves the FAISS index to `vectorstore_careflow/`:
```bash
python ingest.py
```

### Step 2: Launch the Streamlit Web Interface
```bash
python -m streamlit run Frontend/app.py
```
Open your browser at **`http://localhost:8501`**.

### Step 3: Run the Terminal Chatbot (Optional)
```bash
python chatbot.py
```
Type `exit` to quit the CLI session.

---

## 🧪 Sample Questions to Test

1. **AI Credit Quotas:** *"How many monthly AI credits do the Single Clinic AI, Multi-Clinic AI, and Max AI plans include, and what happens when Max AI reaches its limit?"*
2. **Role-Based Access Control (RBAC):** *"What are the five user roles in CAREFLOW AI and their workspace capabilities?"*
3. **Clinic Verification:** *"What details are required for clinic onboarding, and how often is reverification required?"*
4. **Vector Architecture:** *"What are the three RAG knowledge domains and who can access each one?"*
5. **Clinical Safety:** *"How does the Human-in-the-Loop (HITL) policy restrict the Doctor AI Assistant?"*
6. **Out-of-Scope Guardrail Check:** *"How many clinics are available in Hyderabad?"* *(Expected: `"I don't know based on the provided CAREFLOW AI policy document."`)*

---

## 🔒 Security & Grounding Practices

- **Secret Protection:** API keys are loaded via `python-dotenv` and excluded from version control via `.gitignore`.
- **Anti-Hallucination Prompting:** System instructions enforce strict reliance on retrieved FAISS context chunks and surface exact source page numbers for verification.
