# AI Enhanced Research Tool

A Streamlit web application that lets you **search for academic papers** from Google Scholar and **ask an AI** to explain them in any style you choose — all powered by LangChain, ScraperAPI, and free hosted LLMs.

---

## Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
  - [Paper Fetching Pipeline](#1-paper-fetching-pipeline)
  - [Dynamic Prompt Generation with LangChain](#2-dynamic-prompt-generation-with-langchain)
  - [LLM Integration](#3-llm-integration)
- [Environment Variables](#-environment-variables)
- [Getting Started (Replication Guide)](#-getting-started-replication-guide)
- [Limitations & Honest Notes](#%EF%B8%8F-limitations--honest-notes)
- [Future Improvements](#-future-improvements)

---

## Overview

The **AI Enhanced Research Tool** is a personal learning project that bridges academic paper discovery with AI-powered explanations. A user types a research topic or paper title, the app fetches the top relevant papers from Google Scholar, and then — for any paper the user selects — it uses a large language model to generate a tailored explanation based on the user's preferred style and length.

This project was built as a hands-on exploration of:

- The **`scholarly`** Python library for Google Scholar scraping
- **ScraperAPI** as a proxy layer to bypass bot detection on Google Scholar
- **LangChain's** `PromptTemplate` and LLM abstraction components
- **OpenRouter** as a free gateway to hosted LLMs
- **Streamlit** for building a fast, clean UI

---

## Features

| Feature | Description |
|---|---|
| **Paper Search** | Search Google Scholar by topic or title and retrieve up to 10 results |
| **Paper Cards** | Each paper is displayed with its title, authors, venue, year, and abstract |
| **Ask AI** | Select any paper and ask the AI to explain it in your preferred style |
| **LLM Settings** | Choose model, temperature, explanation type, and explanation length |
| **Explanation Styles** | *Mathematics Oriented*, *Beginner Friendly*, or *Summary* |
| **Explanation Lengths** | *Short* (~60 words), *Medium* (~175 words), or *Long* (~400 words) |
| **Paper Links** | Direct link to the original paper source is always provided |

---

## Tech Stack

| Component | Technology |
|---|---|
| **Frontend / UI** | [Streamlit](https://streamlit.io/) |
| **Google Scholar Scraping** | [`scholarly`](https://scholarly.readthedocs.io/) |
| **Proxy / Bot Bypass** | [ScraperAPI](https://www.scraperapi.com/) |
| **LLM Prompt Framework** | [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-openai`, `langchain-huggingface`) |
| **LLM Provider (Primary)** | [OpenRouter](https://openrouter.ai/) — free tier (`openai/gpt-oss-20b:free`) |
| **LLM Provider (Experimental)** | [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints) (`langchain-huggingface`) |
| **Environment Management** | `python-dotenv` |
| **HTTP / Async** | `httpx`, `asyncio` |

---

## Project Structure

```
AI Research Tool/
│
├── main.py
│
├── views/
│   ├── search_view.py         
│   └── ai_view.py             
│
├── src/
│   ├── __init__.py 
│   ├── fetch_papers.py        
│   └── papers.py             
│
├── llm/
│   ├── __init__.py           
│   ├── prompt_generator.py
│   └── llm_model.py
│
├── prompt_template.json
├── requirements.txt           
├── .env                      
├── .gitignore                
└── README.md
```

### Module Responsibilities

```
main.py
  └─► Loads environment variables
  └─► Initialises Streamlit session state
  └─► Routes to search_view or ai_view based on selected paper

views/search_view.py
  └─► Renders the search form
  └─► Calls PaperFetcher asynchronously via asyncio.run()
  └─► Displays paper cards with "ASK AI" button per paper

views/ai_view.py
  └─► Renders selected paper's details
  └─► Provides LLM settings sidebar (model, temperature, style, length)
  └─► Calls PromptGenerator → LangChain PromptTemplate → LLM → renders markdown

src/fetch_papers.py (PaperFetcher)
  └─► Sets up ScraperAPI proxy via scholarly.ProxyGenerator
  └─► Fetches papers using scholarly.search_pubs()
  └─► Offloads blocking I/O to a thread with asyncio.to_thread()

llm/prompt_generator.py (PromptGenerator)
  └─► Defines a LangChain PromptTemplate with 6 input variables
  └─► Maps user-chosen style/length to detailed instruction strings
  └─► Calls .format() on the template to produce the final prompt string
  └─► Saves the last prompt context to prompt_template.json

llm/llm_model.py (ModelSelection)
  └─► Accepts llm_settings dict (model name, temperature)
  └─► Instantiates ChatOpenAI (via OpenRouter base URL) or ChatHuggingFace
  └─► Returns the ready-to-use LangChain chat model
```

---

## How It Works

### 1. Paper Fetching Pipeline

`scholarly` is a Python library that scrapes Google Scholar. Google Scholar is heavily protected against bots, so direct requests get blocked quickly. To solve this, the project uses **ScraperAPI** as a rotating proxy layer.

```python
# src/fetch_papers.py
pg = ProxyGenerator()
pg.ScraperAPI(api_key)           # Register ScraperAPI as the proxy
scholarly.use_proxy(pg)          # Tell scholarly to route through it

search_query = scholarly.search_pubs(query)  # Now safe to scrape
```

Each result is normalised into a dictionary with keys: `title`, `authors`, `link`, `venue`, `year`, `abstract`. The fetch is wrapped in `asyncio.to_thread()` so the blocking I/O doesn't freeze the Streamlit event loop.

---

### 2. Dynamic Prompt Generation with LangChain

The core of the AI feature is a **LangChain `PromptTemplate`** defined in `llm/prompt_generator.py`. The template takes **6 variables** — paper metadata plus two instruction strings that are dynamically selected based on user preferences.

```python
# llm/prompt_generator.py
from langchain_core.prompts import PromptTemplate

paper_explanation_prompt = PromptTemplate(
    input_variables=[
        "title", "authors", "abstract", "paper_link",
        "explanation_type_instruction",   # e.g. "Focus on math formulations..."
        "explanation_length_instruction", # e.g. "2-3 sentences (~60 words)"
    ],
    template="..."  # Full system + user instruction prompt
)
```

The `explanation_type_instruction` and `explanation_length_instruction` are looked up from predefined dictionaries:

| User Selection | Mapped Instruction |
|---|---|
| Mathematics Oriented | Focus on equations, algorithms, LaTeX notation |
| Beginner Friendly | Plain language, analogies, no assumed background |
| Summary | Concise academic summary of purpose, method, findings |
| Short | ~50–70 words |
| Medium | ~150–200 words |
| Long | ~350–450 words across 3–5 paragraphs |

This design means **no hardcoded prompt strings** are scattered around the codebase — all variation is handled by swapping instruction values into the single template.

---

### 3. LLM Integration

LangChain's model abstraction (`ChatOpenAI` from `langchain-openai`) is used to call the LLM. **OpenRouter** exposes an OpenAI-compatible API endpoint, so `ChatOpenAI` works with it by simply overriding the `base_url`:

```python
# llm/llm_model.py
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url=os.getenv('OPENAI_BASE_URL'),  # Points to OpenRouter
    model="openai/gpt-oss-20b:free",        # Free OpenRouter model
    temperature=0.7,
)
```

The full pipeline in `ai_view.py` is just three lines:

```python
pg   = PromptGenerator(paper, llm_settings)
prompt_str = pg.build_explanation_prompt()  # LangChain .format() call
result     = llm.invoke(prompt_str)          # LangChain model call
st.markdown(result.content)                  # Render markdown output
```

---

## Environment Variables

Create a `.env` file in the project root with the following keys:

```env
# ScraperAPI — for proxying Google Scholar requests
SCRAPERAPI_KEY=your_scraperapi_key_here

# OpenRouter — OpenAI-compatible base URL for free LLM access
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=your_openrouter_api_key_here

# Hugging Face — only needed if you enable HuggingFace models
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

> **Never commit your `.env` file.** It is already listed in `.gitignore`.

---

## Getting Started (Replication Guide)

Follow these steps to run this project locally after cloning from GitHub.

### Prerequisites

- Python 3.10 or higher
- A free [ScraperAPI](https://www.scraperapi.com/) account (free tier: 5,000 requests/month)
- A free [OpenRouter](https://openrouter.ai/) account (access to free models like `openai/gpt-oss-20b:free`)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Amritkandel49/AI-Enhanced-Research-Tool.git
cd AI-Enhanced-Research-Tool
```

### Step 2 — Create and Activate a Virtual Environment

```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note on `requirements.txt`:** The file uses bare package names. For a reproducible install with pinned versions, consider running `pip freeze > requirements.txt` after your initial install.

### Step 4 — Set Up Environment Variables

```bash
cp .env.example .env   # or create .env manually
```

Edit `.env` and fill in your actual API keys:

```env
SCRAPERAPI_KEY=your_scraperapi_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=your_openrouter_api_key_here
```

### Step 5 — Run the Application

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

### Step 6 — Use the App

1. Type a research paper title or topic in the search box and click **Fetch Papers**.
2. Browse the list of returned papers from Google Scholar.
3. Click **ASK AI** on any paper you want to understand.
4. On the AI page, configure your preferred model, temperature, explanation style, and length.
5. Click **Ask AI** to generate an explanation.
6. Use the **⬅ Back to Search Results** button to return to your results.

---

## Limitations & Honest Notes

This project was built while **actively learning** LangChain and AI-powered application development. Here are the real constraints to be aware of:

### API & Rate Limits

| Service | Constraint |
|---|---|
| **ScraperAPI (Free Tier)** | 5,000 API calls/month — each Scholar page fetch consumes calls. Heavy usage will exhaust this quickly. |
| **OpenRouter (Free Models)** | Free model endpoints (`openai/gpt-oss-20b:free`) are rate-limited and may respond slowly or return errors during peak demand. |
| **Hugging Face Endpoints** | Free HuggingFace Inference API has cold-start delays and strict rate limits. The HuggingFace model integration is present in the codebase but kept commented out as a work-in-progress. |

### Architectural Limitations

- **Abstract-only context:** `scholarly` returns only the abstract and metadata for each paper, not the full text. The AI explanation is therefore limited to what can be inferred from the abstract. The prompt instructs the model to visit the paper link — but whether it can do so depends on the model's capabilities.
- **No chat history / memory:** The current AI view generates a fresh explanation on every submission. There is no multi-turn conversation or context carried over between interactions (the chat input UI exists but is disabled).
- **Single model at a time:** The `ModelSelection` class currently supports only one production model (`openai/gpt-oss-20b:free` via OpenRouter). Adding more models requires extending the `if/elif` block manually.
- **No result caching:** Every search triggers a live ScraperAPI call. Results are held only in Streamlit session state and lost on page refresh.

### Learning Context

> This tool was built as a **portfolio and learning project** by someone actively learning Gen-AI application development with LangChain. It demonstrates core concepts — LangChain PromptTemplates, LLM abstraction, async I/O with Streamlit, and API-based scraping — rather than being production-ready software.

---

## Future Improvements

- [ ] Add full-text retrieval via arXiv or Semantic Scholar APIs (no scraping needed)
- [ ] Enable multi-turn chat with conversation memory using LangChain's `ConversationBufferMemory`
- [ ] Support more LLM providers (Groq, Together AI, etc.) in `ModelSelection`
- [ ] Add result caching with `st.cache_data` to save ScraperAPI quota
- [ ] Implement a favourites / save-paper feature using a local SQLite database
- [ ] Deploy to Streamlit Community Cloud

---

## Author

**Amrit Kandel**
- GitHub: [@Amritkandel49](https://github.com/Amritkandel49)

---

## License

This project is open-source and available under the [MIT License](LICENSE).