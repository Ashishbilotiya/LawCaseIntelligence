# ⚖️ LawCaseIntelligence

An AI-powered legal case analysis platform that leverages **Retrieval-Augmented Generation (RAG)** and a **multi-agent orchestration system** to process legal cases, extract insights, and answer queries.

---

## 🚀 Features

- **Multi-Agent AI Pipeline:** Specialized agents (`IssueAgent`, `ArgumentAgent`, `PrecedentAgent`, `LawStatuteAgent`, `ReasoningVerdictAgent`, `MasterAgent`) collaborate to analyze legal judgments.
- **Hybrid RAG System:** Ingests PDF judgments, chunks them semantically, generates embeddings using `BGE`, and stores them in a vector database for intelligent retrieval.
- **Interactive Chat:** Ask natural language questions about specific cases or the entire project.
- **Project Management:** Organize judgments into separate projects with isolated data and vector collections.
- **Real-Time Progress:** SocketIO-powered live pipeline updates during document processing.
- **Robust LLM Load Balancing:** `TokenScheduler` rotates across multiple `GROQ_API_KEY`s to mitigate TPM/RPM limits.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Flask-SocketIO
- **AI / LLM:** LangChain, Groq (Llama 3.3 70B Versatile)
- **Vector DB:** ChromaDB (Persistent SQLite-backed)
- **Database:** SQLAlchemy (SQLite)
- **PDF Processing:** PyMuPDF
- **Embeddings:** HuggingFace `BAAI/bge-large-en-v1.5`

---

## 💻 Local Development

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/LawCaseIntelligence.git
cd LawCaseIntelligence
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy the example environment file and add your Groq API keys:
```bash
cp .env.example .env
```
Edit `.env` and provide at least one `GROQ_API_KEY`.

### 5. Run the Application
```bash
cd LawCaseIntelligence
python app.py
```
The app will be available at `http://localhost:5001`.

---

## ☁️ Deployment on Render

This project is pre-configured for deployment on [Render](https://render.com).

### Option A: Using `render.yaml` (Infrastructure as Code)
1. Push your code to a GitHub repository.
2. In Render Dashboard, click **New** → **Blueprint**.
3. Connect your GitHub repository. Render will automatically detect `render.yaml` and provision the service.

### Option B: Manual Setup
1. Push your code to a GitHub repository.
2. In Render Dashboard, click **New** → **Web Service**.
3. Connect your GitHub repository.
4. Configure as follows:
   - **Environment:** `Docker`
   - **Region:** Choose closest to you
   - **Plan:** `Free` (or higher for better performance)
5. Add a **Persistent Disk** to retain data across deploys:
   - **Mount Path:** `/app/LawCaseIntelligence/data`
   - **Size:** `1 GB` (minimum)
6. Add **Environment Variables** in the Render Dashboard:
   - `GROQ_API_KEY_1`, `GROQ_API_KEY_2`, etc.
   - `FLASK_SECRET` (any long random string)
   - `ASYNC_MODE=gevent`
   - `PORT=10000`
7. Click **Deploy**.

---

## 📂 Project Structure

```
LawCaseIntelligence/
├── agents/              # Multi-agent orchestration (LangGraph)
├── backend/             # Core business logic, LLM services, chat router
├── database/           # SQLAlchemy models and repositories
├── data/                # Uploads, ChromaDB, SQLite DB (gitignored)
├── frontend/            # Flask app, templates, static files
├── rag/                 # PDF ingestion, chunking, embeddings, vector DB
├── app.py               # Application entry point
├── gunicorn.conf.py     # Production WSGI configuration
├── Dockerfile           # Docker image for Render
├── render.yaml          # Render infrastructure definition
└── requirements.txt     # Python dependencies
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY_1` | ✅ Yes | Your primary Groq API key |
| `GROQ_API_KEY_2` | ❌ No | Secondary key for load balancing |
| `GROQ_API_KEY_3` | ❌ No | Tertiary key |
| `GROQ_API_KEY_4` | ❌ No | Quaternary key |
| `FLASK_SECRET` | ✅ Yes | Secret key for Flask sessions |
| `ASYNC_MODE` | ✅ Yes (Prod) | Set to `gevent` for production |
| `PORT` | ❌ No | Defaults to `5001` (dev) / `10000` (Render) |
| `LOG_LEVEL` | ❌ No | `INFO`, `DEBUG`, `WARNING`, `ERROR` |

---

## 📜 License

This project is for educational and research purposes.

---

## ⚠️ Disclaimer

This tool provides AI-generated legal analysis for informational purposes only. It is **not** a substitute for professional legal advice. Always consult a qualified attorney for legal matters.