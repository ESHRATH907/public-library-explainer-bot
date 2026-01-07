import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from library_docs import library_documents

# Configure Gemini API
genai.configure(api_key="AIzaSyANxSIh4yAWuZi9-Xp85GE-D6_0WaXae-U")

SYSTEM_PROMPT = """
You are a Public Library Services Explainer Bot.
Answer ONLY using the provided library context.
Do NOT assume or invent policies.
Do NOT issue books or manage accounts.
Politely refuse restricted actions.
"""

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
doc_embeddings = embedder.encode(library_documents)
dimension = doc_embeddings.shape[1]

# FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Gemini model
model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    system_instruction=SYSTEM_PROMPT
)

st.title("📚 Public Library Services Explainer Bot (RAG Enabled)")
st.write("Ask about library rules, borrowing, and digital services.")

user_question = st.text_input("Enter your question")

if user_question:
    # Retrieve relevant docs
    query_embedding = embedder.encode([user_question])
    _, indices = index.search(np.array(query_embedding), k=3)

    retrieved_docs = "\n".join([library_documents[i] for i in indices[0]])

    prompt = f"""
    Library Context:
    {retrieved_docs}

    User Question:
    {user_question}

    Answer clearly and concisely using ONLY the context above.
    """

    response = model.generate_content(prompt)

    st.write("### 📖 Explanation")
    st.write(response.text)
