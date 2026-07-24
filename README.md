# DocMind AI 🧠

DocMind AI is a production-grade AI-powered Retrieval-Augmented Generation (RAG) platform designed to interact intelligently with PDF documents. This repository contains the core foundation for both backend services and frontend interfaces.

> [!NOTE]
> **Milestone Status:** Currently at **Milestone 1: Project Foundation**. Core FastAPI architecture, configuration settings, environment structure, and health routes are fully operational. RAG, database, and AI integrations are planned for subsequent milestones.

---

## 📁 Project Structure

```text
docmind-ai/
│
├── backend/
│   ├── app/
│   │   ├── api/          # API controllers, sub-routers, and request schemas
│   │   ├── core/         # Core application settings and environment parsing
│   │   ├── services/     # Business logic layer (reserved for document & RAG services)
│   │   ├── utils/        # Utility modules and helper functions
│   │   ├── __init__.py   # Application package initialization
│   │   └── main.py       # FastAPI application entrypoint & middleware setup
│   │
│   ├── uploads/          # Local directory for stored PDF uploads (Git ignored)
│   ├── vector_store/     # Local directory for vector index storage (Git ignored)
│   ├── requirements.txt  # Backend Python dependencies
│   ├── .env.example      # Environment variables configuration template
│   └── .gitignore        # Backend-specific Git ignore rules
│
├── frontend/             # Frontend application workspace (reserved for future UI)
│
├── README.md             # Project documentation and setup guide
└── .gitignore            # Top-level workspace Git ignore rules
```

---

## 🛠️ Tech Stack & Requirements

- **Language:** Python 3.12+
- **Framework:** FastAPI
- **ASGI Server:** Uvicorn
- **Environment Management:** `python-dotenv` & `pydantic-settings`

---

## 🚀 Installation & Setup

Follow these steps to set up and run the DocMind AI backend locally.

### 1. Clone & Navigate to Workspace
```bash
cd docmind-ai/backend
```

### 2. Create & Activate Virtual Environment
- **On Linux/macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

# Environment Setup

To run the application, configure your environment variables:

1. **Configure Backend Environment:**
   * Copy `backend/.env.example` to `backend/.env`:
     ```powershell
     # Windows (PowerShell)
     Copy-Item backend/.env.example backend/.env
     ```
     ```bash
     # macOS / Linux
     cp backend/.env.example backend/.env
     ```
   * Open `backend/.env` and add your `GROQ_API_KEY`:
     ```env
     GROQ_API_KEY=gsk_your_groq_api_key_goes_here
     ```

2. **Configure Frontend Environment:**
   * Copy `frontend/.env.local.example` to `frontend/.env.local`:
     ```powershell
     # Windows (PowerShell)
     Copy-Item frontend/.env.local.example frontend/.env.local
     ```
     ```bash
     # macOS / Linux
     cp frontend/.env.local.example frontend/.env.local
     ```

3. **Start Backend Server:**
   * Run uvicorn from the `backend/` directory:
     ```bash
     uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
     ```

4. **Start Frontend Dev Server:**
   * Run npm dev server from the `frontend/` directory:
     ```bash
     npm run dev
     ```

5. **Verify Setup:**
   * Send a `GET` request to `http://127.0.0.1:8000/health` or view `http://localhost:3000/health`.
   * Under diagnostics, verify the `groq_service` reports `"status": "healthy"` and `"details": "Groq API key configured."`.


## 🏃 Running the Server

Start the FastAPI development server with Uvicorn auto-reload:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

---

## 🌐 API Endpoints & URLs

Once the server is running, access the following endpoints:

| Endpoint | Method | Description | Expected Output |
| :--- | :--- | :--- | :--- |
| `http://127.0.0.1:8000/` | `GET` | Root API metadata | `{"project": "DocMind AI", "status": "running", "version": "1.0.0"}` |
| `http://127.0.0.1:8000/health` | `GET` | Health status check | `{"status": "healthy"}` |
| `http://127.0.0.1:8000/docs` | `GET` | Interactive Swagger API Docs | OpenAPI Documentation UI |
| `http://127.0.0.1:8000/redoc` | `GET` | ReDoc API Documentation | ReDoc UI |

---

## 🗺️ Roadmap & Future Milestones

- [x] **Milestone 1: Project Foundation & Clean Architecture Base**
- [ ] **Milestone 2: PDF Upload & Ingestion Pipeline**
- [ ] **Milestone 3: Document Chunking & Vector Embedding Engine**
- [ ] **Milestone 4: Vector Store & Retrieval Pipeline**
- [ ] **Milestone 5: LLM Integration & RAG Execution Engine**
- [ ] **Milestone 6: Modern React / Next.js Frontend Dashboard**
- [ ] **Milestone 7: Authentication, History & Production Deployment**
