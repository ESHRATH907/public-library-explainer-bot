# 📚 Public Library Explainer Bot

An AI-powered **Retrieval-Augmented Generation (RAG)** application that helps users understand public library services through natural language conversations. The application uses **Google Gemini**, **Sentence Transformers**, and **FAISS** to retrieve relevant library information and generate accurate, context-aware responses.

---

## 🚀 Features

* 🤖 AI-powered chatbot using Google Gemini
* 🔍 Semantic document retrieval with FAISS
* 🧠 Sentence embeddings using Sentence Transformers
* 📖 Context-aware responses based on library information
* 💬 Interactive Streamlit web interface
* ⚡ Fast and lightweight Retrieval-Augmented Generation (RAG) pipeline

---

## 🏗️ Project Architecture

```
User Query
     │
     ▼
Sentence Transformer
     │
     ▼
Generate Query Embedding
     │
     ▼
FAISS Vector Search
     │
     ▼
Retrieve Top Relevant Library Documents
     │
     ▼
Google Gemini Flash
     │
     ▼
AI-Generated Response
     │
     ▼
Streamlit Interface
```

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries & Frameworks

* Streamlit
* Google Generative AI (Gemini)
* Sentence Transformers
* FAISS
* NumPy

### AI Concepts

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Embeddings
* Prompt Engineering
* Large Language Models (LLMs)

---

## 📂 Project Structure

```
public-library-explainer-bot/
│
├── app.py                  # Main Streamlit application
├── library_docs.py         # Library knowledge base
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/ESHRATH907/public-library-explainer-bot.git
```

```bash
cd public-library-explainer-bot
```

### Create a Virtual Environment (Optional)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Configure Gemini API Key

Create a `.env` file:

```text
GEMINI_API_KEY=YOUR_API_KEY
```

Load it inside your application before configuring Gemini.

> **Note:** Never upload API keys to GitHub.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 💡 How It Works

1. The user enters a question about public library services.
2. The question is converted into a vector embedding using Sentence Transformers.
3. FAISS searches for the most relevant library documents.
4. The retrieved information is provided as context to Google Gemini.
5. Gemini generates an accurate, context-aware response.
6. The response is displayed through the Streamlit interface.

---

## 📖 Example Questions

* What services does the library provide?
* How can I borrow books?
* Are digital resources available?
* Can I reserve a book online?
* What are the library opening hours?
* Does the library provide study spaces?

---

## 🎯 Skills Demonstrated

* Python Programming
* Retrieval-Augmented Generation (RAG)
* Google Gemini API Integration
* Semantic Search
* FAISS Vector Database
* Sentence Embeddings
* Prompt Engineering
* Streamlit Application Development
* Git & GitHub
* Software Development

---

## 🔮 Future Improvements

* Load documents from PDFs or databases
* Chat history support
* Persistent FAISS index
* Document source citations
* User authentication
* Admin panel for updating library documents
* Deployment on Streamlit Community Cloud
* Docker support
* Unit testing
* Logging and monitoring

---

## 👩‍💻 Author

**Eshrath Jahan**

* GitHub: https://github.com/ESHRATH907

---

## 📄 License

This project is licensed under the MIT License.
