# 📖 WikiRAG — Wikipedia, Distilled

A fully local RAG (Retrieval-Augmented Generation) chatbot that answers questions using **live Wikipedia content**, runs entirely on your machine, and cites its sources for every answer.

Ask a question → it searches Wikipedia → pulls the relevant page → retrieves the most relevant chunks → generates a concise, cited answer using a local LLM. No OpenAI key, no cloud API, no data leaving your machine.

---

## ✨ Features

- **Live Wikipedia retrieval** — every answer is grounded in an actual Wikipedia page, fetched in real time
- **Smart topic extraction** — understands natural questions like *"who is sir lewis hammilton"* and resolves typos/phrasing to the correct page
- **Typo-tolerant search** — falls back to Wikipedia's fuzzy `opensearch` API when an exact match isn't found
- **Local caching with ChromaDB** — once a page is fetched and embedded, repeat questions about the same topic skip the network round-trip entirely
- **Multi-turn conversation memory** — follow-up questions using "he," "she," "it," etc. are resolved using prior chat context
- **Greeting detection** — small talk ("hi," "thanks," "who are you") is answered instantly without touching Wikipedia
- **Cited answers** — every response links back to the exact Wikipedia page it was generated from
- **100% local inference** — powered by [Ollama](https://ollama.com) running Llama 3.2, no external LLM API required
- **Custom dark-themed frontend** — a single self-contained HTML file, no build step, no framework

---

## 🏗️ Architecture & Flow

┌──────────────┐      ┌──────────────────┐      ┌─────────────────────┐

│   Frontend   │ ───▶ │   FastAPI Backend │ ───▶ │   WikipediaRAG       │

│ (index.html) │      │   (app/main.py)   │      │   (app/rag.py)       │

└──────────────┘      └──────────────────┘      └─────────┬───────────┘

│

┌────────────────────────────────────┼────────────────────────────┐

▼                                    ▼                            ▼

1. Greeting check                  2. Topic extraction          3. Wikipedia search

(regex match, instant            (Ollama resolves pronouns      (REST API + fuzzy

canned reply, no LLM call)        & extracts clean topic)       opensearch fallback)

│

▼

4. Cache check (ChromaDB)

│              │

cache hit        cache miss

│              │

│              ▼

│    5. Fetch page content,

│       chunk, embed, store

│◀─────────────┘

▼

6. Retrieve top-k relevant chunks

▼

7. Generate concise answer (Ollama)

using retrieved context + history

▼

8. Return answer + cited source URL

### Step-by-step

1. **Greeting check** — incoming message is matched against common small-talk patterns (hi, hello, thanks, bye, who are you). If matched, a canned reply is returned instantly with zero Wikipedia or LLM calls.
2. **Topic extraction** — the raw question (plus recent conversation history) is sent to the local LLM with a few-shot prompt that extracts the actual subject of the question, resolving typos and pronouns ("he," "that") using prior context.
3. **Wikipedia search** — the clean topic is searched against Wikipedia's `action=query&list=search` API. If no result is found, it falls back to the typo-tolerant `action=opensearch` endpoint (the same one that powers Wikipedia's own search-as-you-type box).
4. **Cache check** — before fetching anything, the resolved page title is checked against the local ChromaDB collection. If chunks for that page already exist, the network fetch is skipped entirely.
5. **Fetch, chunk, embed** *(cache miss only)* — the full page is fetched via Wikipedia's REST API as plain text, split into ~800-character overlapping chunks, and stored in ChromaDB with metadata (page title, source URL).
6. **Retrieval** — the original question is used to semantically query ChromaDB for the most relevant chunks from that specific page.
7. **Generation** — the retrieved chunks, the original question, and recent conversation history are passed to the local LLM with a prompt enforcing concise, direct, non-repetitive answers.
8. **Response** — the answer is returned along with the canonical Wikipedia URL it was generated from, displayed in the UI as an expandable citation.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| LLM inference | Ollama (Llama 3.2), fully local |
| Vector store / cache | ChromaDB (persistent, on-disk) |
| Chunking | LangChain text splitters |
| Wikipedia access | Wikipedia REST API via `requests` (direct, no flaky third-party wrapper) |
| Frontend | Single self-contained HTML/CSS/JS file, no framework, no build step |

---

## 📂 Project Structure

WikiRAG-Chatbot/

├── app/

│   ├── init.py

│   ├── main.py          # FastAPI app, /chat and /health endpoints, CORS config

│   └── rag.py            # Core RAG pipeline: search, cache, retrieve, generate

├── frontend/

│   └── index.html        # Self-contained UI — open directly in a browser

├── data/

│   └── chroma_db/        # Auto-generated vector store cache (gitignored)

├── requirements.txt

├── .env.example

├── .gitignore

└── README.md

---

## 🚀 Setup & Running Locally

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- The `llama3.2` model pulled in Ollama:
```bash
  ollama pull llama3.2
```

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/WikiRAG-Chatbot.git
cd WikiRAG-Chatbot
```

### 2. Set up a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the backend

```bash
uvicorn app.main:app --reload --workers 1
```

The API will be running at `http://127.0.0.1:8000`. Verify it's healthy:

```bash
curl http://127.0.0.1:8000/health
```

### 5. Open the frontend

Just open the file directly in a browser — no server needed:

```bash
open frontend/index.html
```

(or double-click it in Finder/Explorer)

Make sure the backend (step 4) is running in the background — the frontend calls it directly at `http://127.0.0.1:8000/chat`.

---

## 💬 Example Usage

| You ask | What happens |
|---|---|
| `hi` | Instant greeting, no Wikipedia call |
| `who is lewis hamilton` | Searches Wikipedia → caches the page → answers with citation |
| `how many championships has he won?` | Resolves "he" → Lewis Hamilton using conversation history → answers from cached page (no re-fetch) |
| `capital of japan` | Searches, caches, answers, cites |

---

## 🧠 Why these design choices?

- **Direct Wikipedia REST calls instead of the `wikipedia` pip package** — the popular package is unmaintained and silently fails on some requests without a proper `User-Agent` header. Calling the API directly with `requests` is more reliable and fully under our control.
- **Topic extraction as a separate LLM step** — free-form questions ("who is sir lewis hammilton") are poor direct inputs for Wikipedia's search index. A lightweight extraction step dramatically improves match accuracy, especially for typos and conversational phrasing.
- **ChromaDB as a cache, not just a vector store** — since this is single-user and locally hosted, ChromaDB doubles as a simple "have we already fetched this page" cache, avoiding redundant network calls and re-embedding.
- **No frontend framework** — a single static HTML file keeps the project trivially easy to run, fork, and modify with zero build tooling.

---

## 📌 Known Limitations

- Answer quality is bounded by the local model (Llama 3.2) — it's good, not GPT-4-level.
- Only the single best-matching Wikipedia page is used per question; it doesn't synthesize across multiple pages.
- English Wikipedia only, by default.
- No authentication/rate limiting — intended for local/personal use, not public deployment as-is.

---

## 🤝 Contributing

Issues and PRs welcome. This started as a learning project and is open for anyone to extend — multi-language support, better disambiguation handling, streaming responses, and persistent multi-session history are all good next steps.

---

## 📄 License

MIT — do whatever you want with it.