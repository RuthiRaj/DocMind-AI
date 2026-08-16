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
├── .gitignore
├── LICENSE
├── README.md
├── backend/
│   ├── .env.example
│   ├── .gitignore
│   ├── pytest.ini
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── utils/
│   ├── scripts/
│   │   ├── migrate_v2_documents.py
│   │   ├── run_adversarial_audit.py
│   │   └── verify_live_api_execution.py
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_preflight_trimming.py
│   │   └── test_rag_architecture_integration.py
│   └── vector_store/
│       └── .gitkeep
└── frontend/
    ├── .env.local.example
    ├── .gitignore
    ├── next.config.ts
    ├── package.json
    ├── tsconfig.json
    ├── app/
    ├── components/
    ├── constants/
    ├── hooks/
    ├── lib/
    ├── providers/
    ├── services/
    └── types/
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

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health check |
| `POST` | `/upload` | Upload a PDF document |
| `POST` | `/process/{document_id}` | Extract/process document text |
| `POST` | `/chunk/{document_id}` | Create document chunks |
| `POST` | `/embed/{document_id}` | Generate embeddings |
| `POST` | `/index/{document_id}` | Create/update FAISS index |
| `POST` | `/retrieve/{document_id}` | Retrieve relevant chunks |
| `GET` | `/retrieve/{document_id}/debug` | Retrieval/debug telemetry |
| `POST` | `/chat/{document_id}` | Grounded RAG chat |
| `GET` | `/documents` | List documents |
| `GET` | `/documents/statistics` | Document statistics |
| `GET` | `/documents/{document_id}` | Get document details |
| `GET` | `/documents/{document_id}/status` | Get document pipeline status |
| `DELETE` | `/documents/{document_id}` | Delete a document |
| `POST` | `/maintenance/cleanup` | Clean disposable runtime data |

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
