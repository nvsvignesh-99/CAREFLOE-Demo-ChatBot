from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load your CAREFLOW AI Policy PDF
loader = PyPDFLoader("data/CAREFLOW AI Platform & Operational Policy Handbook.pdf")
documents = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print("Total chunks created:", len(chunks))

# Step 3: Create local Embedding Model (zero API quota used)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Create & Save the FAISS Vector Database
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore_careflow")
print("CAREFLOW AI Vector Database Created Successfully")