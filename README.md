# DocMind AI 🧠

**DocMind AI** is a production-grade, grounded **Retrieval-Augmented Generation (RAG)** platform designed for enterprise PDF document analysis, multi-hop synthesis, and interactive chat.

Built with a fast **FastAPI** Python backend, zero-dependency **BM25 + FAISS** hybrid retrieval engine, and a modern **Next.js 16 / React 19** frontend dashboard.

---

## ✨ Key Features & RAG V2 Highlights

* 🔍 **Hybrid Retrieval Engine (BM25 + Vector Search + RRF):** Combines lexical term density (BM25 Okapi) with dense vector embeddings (`BAAI/bge-small-en-v1.5`) using Reciprocal Rank Fusion ($K=60$).
* 📍 **1:1 Grounded Page Citations:** PyMuPDF text cleaning ensures precise page-level grounding and zero character offset drift across document pages.
* ⚡ **Selective Query Expansion:** Smart heuristic filtering skips unnecessary LLM query expansion calls on short queries or technical codes, saving tokens and eliminating Groq rate-limit spikes (`HTTP 429`).
* 💬 **Multi-Turn Session Memory:** Supports conversational context tracking across multi-turn follow-up questions.
* 🛡️ **Failure-Path Robustness:** Built-in safeguards reject corrupt PDFs, empty documents, missing index files, and provider timeouts (`HTTP 504`) with clean, descriptive error responses.
* ⚡ **Lazy Auto-Upgrade:** Automatically upgrades legacy V1 document indexes to V2 on-the-fly during user queries.

---

## 🏗️ Architecture

```
                          +-------------------------+
                          |   User Question Input   |
                          +------------+------------+
                                       |
                                       v
                     +-----------------------------------+
                     | Selective Query Rewriting Filter  |
                     |  - Skip if <= 4 words             |
                     |  - Skip if technical code/ID      |
                     +-----------------+-----------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
     +---------------------------+           +---------------------------+
     |   BM25 Okapi Keyword      |           | Sentence-Transformers     |
     |   Retriever (Term Density)|           | BAAI/bge-small-en-v1.5    |
     +-------------+-------------+           +-------------+-------------+
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  Reciprocal Rank Fusion (RRF)     |
                     |  Score = 1/(60 + r_vec) + ...     |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  Neighbor-Chunk Merging & Cap     |
                     |  - Max 2 adjacent chunks          |
                     |  - Max 1,500 characters cap       |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  Grounded Prompt Builder & LLM    |
                     |  - Groq llama-3.1-8b-instant      |
                     |  - Factual synonym matching rules |
                     +-----------------+-----------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  Response & Grounded Citations    |
                     +-----------------------------------+
```

---

## 📁 Project Structure

```text
DocMind AI/
├── backend/                            # FastAPI Python Backend Application
│   ├── app/
│   │   ├── main.py                     # Application entrypoint & CORS middleware
│   │   ├── api/                        # REST API routes & router registry
│   │   ├── core/                       # Settings, health checks, prompts, & rate limiter
│   │   ├── schemas/                    # Pydantic request/response data models
│   │   ├── services/                   # Ingestion, chunking, embedding, RAG, & management
│   │   └── utils/                      # Formatting helpers & utilities
│   ├── scripts/                        # Operational & Admin Scripts
│   │   └── migrate_v2_documents.py     # Batch admin migration script for V2 upgrades
│   ├── .env.example                    # Environment variable configuration template
│   └── requirements.txt                # Python backend dependencies
├── frontend/                           # Next.js 16 / React 19 Frontend Application
│   ├── app/                            # App Router pages & layouts
│   ├── components/                     # Modern UI components (dropzone, chat, citations)
│   ├── services/                       # API integration services
│   ├── package.json                    # Frontend dependencies
│   └── .env.local.example              # Frontend environment template
├── LICENSE                             # MIT License
└── README.md                           # Documentation & quickstart guide
```

---

## 🛠️ Tech Stack

### Backend
* **Language & Framework:** Python 3.12+, FastAPI, Uvicorn
* **PDF Inspection & Processing:** PyMuPDF (`fitz`)
* **Embeddings & Vector Store:** `BAAI/bge-small-en-v1.5`, FAISS (`faiss-cpu`)
* **Retrieval Engine:** Zero-dependency BM25 Okapi + FAISS Reciprocal Rank Fusion
* **LLM Provider:** Groq SDK (`llama-3.1-8b-instant`)

### Frontend
* **Framework:** Next.js 16 (App Router), React 19, TypeScript
* **Styling:** Tailwind CSS, Framer Motion, Lucide Icons
* **Data Fetching & State:** React Query (`@tanstack/react-query`), Axios

---

## 🚀 Installation & Setup

### 1. Prerequisites
* Python 3.12+ installed
* Node.js 18+ and npm installed
* A free [Groq API Key](https://console.groq.com/)

---

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
# Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Configure environment variables
# Copy template to .env
cp .env.example .env
```

Open `backend/.env` and insert your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

Start the backend dev server:
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Verify backend health by navigating to `http://127.0.0.1:8000/docs`.

---

### 3. Frontend Setup

Open a new terminal window:
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Configure environment variables
cp .env.local.example .env.local

# Start frontend development server
npm run dev
```
Open `http://localhost:3000` in your browser.

---

### 4. Running Tests

Execute the automated pytest regression suite:
```bash
cd backend
python -m pytest tests/ -v
```

---


## 🌐 API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Comprehensive system health & diagnostics check |
| `POST` | `/api/v1/upload` | Upload PDF file with magic bytes validation |
| `POST` | `/api/v1/process` | Extract cleaned page text & metadata |
| `POST` | `/api/v1/chunk` | Smart text chunking with page bounds |
| `POST` | `/api/v1/embed` | Generate dense vector embeddings |
| `POST` | `/api/v1/index` | Build local FAISS vector search index |
| `POST` | `/api/v1/retrieve` | Execute BM25 + Vector hybrid RRF retrieval |
| `POST` | `/api/v1/chat` | Send question & receive grounded RAG answer with citations |
| `GET` | `/api/v1/management/documents` | List all processed documents & pipeline status |

---

## 📊 Evaluation & Robustness Benchmarks

DocMind AI RAG V2 was benchmarked across multi-page technical documents:

* **Unanswerable Refusal Precision:** **100% (4/4)** strict refusal rate on unanswerable questions without context leakage.
* **Multi-Hop Synthesis Rate:** **100% (4/4)** accurate synthesis across facts separated by up to 11 pages.
* **Cross-Document Generalization:** **100% (2/2)** accurate grounding across distinct HR and Financial policy PDFs.
* **Error Resilience:** Clean `HTTPException` responses for corrupt PDF files (`400`), empty text documents (`400`), missing indexes (`400`), and provider timeouts (`504`).

---

## 📜 License

This project is licensed under the MIT License.
