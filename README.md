# AI Enhanced Research Tool

A Streamlit app that searches arXiv papers and lets you **chat with the full text** of any paper — powered by a RAG pipeline built with LangChain and free hosted LLMs.

---

## Features

| Feature | Description |
|---|---|
| **Paper Search** | Search arXiv by topic/title, retrieve up to 10 results |
| **Paper Cards** | Title, authors, venue, year, abstract, and direct arXiv link |
| **AI Explanation** | One-shot explanation in your chosen style and length |
| **RAG Chatbot** | Chat with the full text of a paper using a persistent Chroma vector store |
| **Full-Text Retrieval** | Downloads and parses the paper PDF via PyMuPDF |
| **Text Chunking** | Splits paper text with `RecursiveCharacterTextSplitter` (1000 chars / 200 overlap) |
| **LLM Settings** | Choose model, temperature, explanation style, and length |

---

## Tech Stack

| Component | Technology |
|---|---|
| **UI** | [Streamlit](https://streamlit.io/) |
| **Paper Search** | [`arxiv`](https://python-arxiv.readthedocs.io/) |
| **PDF Parsing** | [`pymupdf`](https://pymupdf.readthedocs.io/) |
| **LLM Framework** | [LangChain](https://www.langchain.com/) (LCEL) |
| **Vector Store** | [Chroma](https://www.trychroma.com/) (persistent, via `langchain-chroma`) |
| **Embeddings** | `langchain-google-genai` |
| **LLM Provider** | [OpenRouter](https://openrouter.ai/) — `openai/gpt-oss-20b:free` |
| **Env Management** | `python-dotenv` |

---

## Project Structure

```
AI Research Tool/
│
├── main.py
│
├── views/
│   ├── search_view.py          # Search form + paper cards
│   └── ai_view.py              # AI explanation + RAG chatbot UI
│
├── src/
│   ├── fetch_papers.py         # arXiv search (async)
│   ├── paper_loader.py         # PDF download + text extraction (PyMuPDF)
│   ├── text_processing.py      # Text chunking → LangChain Documents
│   ├── format_llm_output.py    # Markdown formatting helpers
│   └── papers.py               # Paper data model
│
├── llm/
│   ├── prompt_generator.py     # LangChain PromptTemplate builder
│   ├── llm_model.py            # LLM instantiation (OpenRouter / HuggingFace)
│   └── rag_chain.py            # LCEL RAG chain (retriever → prompt → LLM)
│
├── chroma_db/                  # Persistent vector store (git-ignored)
├── requirements.txt
├── .env
└── README.md
```

---

## How It Works

### 1. Paper Fetching
`PaperFetcher` queries arXiv and returns paper metadata (title, authors, abstract, link). The blocking call is wrapped in `asyncio.to_thread()` to avoid freezing Streamlit.

### 2. AI Explanation (one-shot)
`PromptGenerator` builds a `LangChain PromptTemplate` from paper metadata + user-chosen style/length, then calls the LLM via OpenRouter.

### 3. RAG Pipeline (full-text chat)

```
arXiv PDF
  │
  ▼
PaperLoader          — downloads PDF, extracts text with PyMuPDF
  │
  ▼
TextProcessor        — splits into 1000-char chunks (200 overlap)
  │
  ▼
Chroma (persistent)  — embeds and stores chunks on first load; reloads on revisit
  │
  ▼
build_rag_chain()    — LCEL chain: retriever (top-4) → RAG prompt → LLM → answer
  │
  ▼
Chatbot UI           — multi-turn Q&A rendered in st.chat_message
```

The vector store is keyed by `arxiv_id`, so each paper is only embedded once. Subsequent visits reuse the existing store.

---

## Getting Started

### Prerequisites
- Python 3.10+
- Free [OpenRouter](https://openrouter.ai/) account
- Google API key (for Gemini embeddings)

### Steps

```bash
# 1. Clone
git clone https://github.com/Amritkandel49/AI-Enhanced-Research-Tool.git
cd AI-Enhanced-Research-Tool

# 2. Virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Environment
cp .env.example .env   # fill in your keys

# 5. Run
streamlit run main.py
```

App opens at `http://localhost:8501`.

### Usage
1. Search for a topic → browse paper cards.
2. Click **ASK AI** → configure style/length → get a one-shot explanation.
3. Click **Chat with Paper** → the PDF is fetched, chunked, and embedded automatically.
4. Ask any question in the chat box — the RAG chain retrieves relevant chunks and answers from the paper's full text.

---

## Limitations

- **Embedding cost:** First-time load of a paper calls the Google embedding API for every chunk.
- **PDF availability:** A small number of arXiv papers are only available as scanned images; PyMuPDF will extract little or no text from these.
- **No result caching:** Search results are held in Streamlit session state and lost on page refresh.

---

## Author

**Amrit Kandel** — [@Amritkandel49](https://github.com/Amritkandel49)

---

## License

MIT License