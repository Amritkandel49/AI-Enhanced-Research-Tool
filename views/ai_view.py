import streamlit as st
import re
from llm import PromptGenerator, ModelSelection, build_rag_chain
from src import format_llm_output, TextProcessor
from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ── Embedding function (shared across the module) ────────────────────────────
_EMBEDDING_FN = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")


def _get_vector_store_for_paper(arxiv_id: str) -> Chroma:
    """
    Returns a Chroma vector store scoped to a single paper.

    Each paper gets its own collection name derived from its arXiv ID,
    so chunks from different papers never mix together.
    Chroma will reuse the collection if it already exists (i.e. the paper
    was loaded in a previous session), otherwise it creates a fresh one.

    Args:
        arxiv_id: Short arXiv ID (e.g. "1706.03762").

    Returns:
        A Chroma VectorStore instance for this paper.
    """
    # Chroma collection names must use only [a-zA-Z0-9._-] and start/end with alnum.
    sanitized_id = re.sub(r"[^a-zA-Z0-9._-]", "_", arxiv_id) or "unknown"
    collection_name = f"paper_{sanitized_id}"

    return Chroma(
        collection_name=collection_name,
        embedding_function=_EMBEDDING_FN,
        persist_directory="./chroma_db",
        create_collection_if_not_exists=True,
    )


def _load_paper_into_vector_store(arxiv_id: str, vector_store: Chroma) -> None:
    """
    Loads the full text of the paper into the vector store only if it
    hasn't been indexed yet (checked via the document count in the collection).

    This prevents re-downloading and re-embedding the same paper every time
    the user interacts with the page.

    Args:
        arxiv_id    : Short arXiv ID.
        vector_store: The Chroma store for this paper.
    """
    existing_count = vector_store._collection.count()

    if existing_count == 0:
        with st.spinner("📄 Loading full paper text and building index..."):
            documents = TextProcessor(arxiv_id).fetch_and_process_text()
            vector_store.add_documents(documents)



def update_llm_settings():
    st.session_state.llm_settings['model'] = st.session_state.llm_model_key
    st.session_state.llm_settings['temperature'] = st.session_state.llm_temp_key
    st.session_state.llm_settings['explanation_type'] = (
        st.session_state.explanation_type_key.lower().replace(" ", "_")
    )
    st.session_state.llm_settings['explanation_length'] = (
        st.session_state.explanation_length_key.lower().replace(" ", "_")
    )
    st.session_state.llm_submitted = True



def render_ai_view():
    paper = st.session_state.selected_paper
    arxiv_id = paper.get("arxiv_id", "")

    if "llm_settings" not in st.session_state:
        st.session_state.llm_settings = {
            "model": "openai/gpt-oss-20b:free",
            "temperature": 0.7,
            "explanation_type": "beginner_friendly",
            "explanation_length": "short",
        }
        st.session_state.llm_submitted = False

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("⬅ Back to Search Results"):
        st.session_state.selected_paper = None
        st.session_state.chat_history = []
        st.rerun()

    header = st.container(border=True, gap="small")
    with header:
        header_c1, header_c2 = header.columns([3, 1])

        with header_c1:
            header_c1.title(paper['title'])

            authors = paper['authors']
            author_str = (
                ', '.join(authors[:5]) +
                (f' +{len(authors)-5} more' if len(authors) > 5 else '')
            )
            header_c1.write(f"**Authors:** {author_str}")

            meta_cols = header_c1.columns(3)
            meta_cols[0].write(f"📅 **Published:** {paper.get('published', paper.get('year', ''))}")
            meta_cols[1].write(f"🏷️ **Category:** `{paper.get('venue', '')}`")
            categories = paper.get('categories', [])
            if categories:
                meta_cols[2].write(
                    '📂 **Tags:** ' + ', '.join(f'`{c}`' for c in categories[:4])
                )
            meta_cols[2].write(f"🔗 **arXiv ID:** `{paper.get('arxiv_id', '')}`")

            header_c1.subheader("Abstract")
            header_c1.write(paper['abstract'])

            btn_c1, btn_c2 = header_c1.columns(2)
            btn_c1.link_button(
                label="📄 Abstract Page",
                url=paper['link'],
                type="primary",
                use_container_width=True,
            )
            if paper.get('pdf_url'):
                btn_c2.link_button(
                    label="⬇️ Download PDF",
                    url=paper['pdf_url'],
                    use_container_width=True,
                )

        with header_c2:
            header_c2_form = header_c2.form("llm_setting_form")
            with header_c2_form:
                header_c2_form.selectbox(
                    "Select LLM Model",
                    ["openai/gpt-oss-20b:free", "google/gemini-2.5-flash-lite-preview-06-17"],
                    key="llm_model_key",
                )
                header_c2_form.slider(
                    "Set LLM Temperature",
                    min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                    key="llm_temp_key",
                )
                header_c2_form.selectbox(
                    "Select Explanation Type",
                    ["Mathematics oriented", "Beginner Friendly", "Summary"],
                    key="explanation_type_key",
                )
                header_c2_form.selectbox(
                    "Select Explanation Length",
                    ["Short", "Medium", "Long"],
                    key="explanation_length_key",
                )
                header_c2_form.form_submit_button(
                    "Ask AI",
                    width=200,
                    type="primary",
                    on_click=update_llm_settings,
                )

    if st.session_state.llm_submitted:
        with st.spinner("Generating explanation..."):
            model_selection = ModelSelection(st.session_state.llm_settings)
            llm = model_selection.get_model()
            pg = PromptGenerator(paper, st.session_state.llm_settings)
            prompt_str = pg.build_explanation_prompt()
            result = llm.invoke(prompt_str)
            result_text = result.content if hasattr(result, "content") else str(result)
            formatted_output = format_llm_output(result_text)

        left, middle, right = st.columns([1, 3, 1])
        with middle:
            st.markdown(formatted_output, unsafe_allow_html=True)
    else:
        st.info("Configure your settings above and click **Ask AI** to generate an explanation.")

    st.divider()

    st.subheader("💬 Ask the Paper (RAG)")
    st.caption(
        "Ask any detailed question — the answer is grounded in the full text of the paper."
    )

    vector_store = _get_vector_store_for_paper(arxiv_id)
    _load_paper_into_vector_store(arxiv_id, vector_store)

    # Render existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask about the methodology, results, equations...")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the paper..."):
                model_selection = ModelSelection(st.session_state.llm_settings)
                llm = model_selection.get_model()

                rag_chain = build_rag_chain(vector_store, llm, k=4)
                answer = rag_chain.invoke(user_question)

            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})