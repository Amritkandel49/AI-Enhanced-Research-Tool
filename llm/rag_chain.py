from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStore


# ── RAG Prompt Template ───────────────────────────────────────────────────────
# This prompt is injected with:
#   {context}  — the relevant chunks retrieved from the vector store
#   {question} — the user's raw question
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert research assistant. A user has a question about a specific research paper.
You are given relevant excerpts retrieved directly from the full text of that paper.

Use ONLY the provided excerpts to answer the question accurately.
Do not fabricate, invent, or assume any information that is not present in the excerpts.
If the excerpts do not contain enough information to answer the question, clearly say so.

Give your answer in markdown format. Use LaTeX-style notation (e.g. $x^2$) for any math expressions.

---
Relevant excerpts from the paper:
{context}
---

User's question: {question}

Answer:""",
)


def _format_docs(docs) -> str:
    """
    Joins a list of retrieved Document objects into a single string,
    separating each chunk with a clear divider so the LLM sees them
    as distinct passages rather than one continuous block.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store: VectorStore, llm, k: int = 4):
    """
    Builds and returns a LangChain LCEL RAG chain.

    The chain follows this flow:
        user question
            │
            ▼
        retriever.invoke(question)   ← fetches top-k relevant chunks
            │
            ▼
        _format_docs(chunks)         ← joins chunks into a single context string
            │
            ▼
        RAG_PROMPT.format(           ← builds the final prompt
            context=...,
            question=...
        )
            │
            ▼
        llm.invoke(prompt)           ← generates the answer
            │
            ▼
        StrOutputParser()            ← extracts plain text from AIMessage
            │
            ▼
        answer string

    Args:
        vector_store : A populated LangChain VectorStore (e.g. Chroma).
        llm          : Any LangChain-compatible chat model.
        k            : Number of top chunks to retrieve per query (default 4).

    Returns:
        A runnable LCEL chain. Call it with:
            chain.invoke("your question here")
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    rag_chain = (
        {
            "context":  retriever | _format_docs,   # retrieve → format
            "question": RunnablePassthrough(),       # pass question through unchanged
        }
        | RAG_PROMPT          # inject context + question into prompt template
        | llm                 # call the LLM
        | StrOutputParser()   # extract plain string from AIMessage
    )

    return rag_chain
