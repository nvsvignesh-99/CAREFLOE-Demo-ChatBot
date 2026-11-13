from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Step 1: Load Environment Variables (reads GOOGLE_API_KEY from .env)
load_dotenv()

# Step 2: Load the Embedding Model (must match ingest.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Load the saved FAISS Vector Database
vectorstore = FAISS.load_local(
    "vectorstore_careflow",
    embeddings,
    allow_dangerous_deserialization=True
)

# Step 4: Configure Retriever (top 4 matching policy chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Step 5: Initialize Free Google Gemini Model
# Use "gemini-3.5-flash-lite" for high daily request limits, or "gemini-3.8-flash" for deeper reasoning
# Step 5: Initialize Free Google Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# Step 6: Chatbot Loop
while True:
    question = input("You: ")

    if question.strip().lower() == "exit":
        break

    # Retrieve relevant policy chunks
    documents = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in documents])

    # Grounded System Prompt
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

    # Generate and print the response safely across all langchain-google-genai versions
    response = llm.invoke(final_prompt)
    answer = response.text if getattr(response, "text", None) else response.content
    print("\nAI:", answer, "\n")