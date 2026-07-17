import io
import arxiv
import pymupdf                             # PyMuPDF
import requests
from langchain_core.documents import Document


class PaperLoader:
    """
    Loads the full text of an arXiv paper using its arXiv ID.

    Strategy:
        1. Use the `arxiv` library (v4) to resolve the paper ID and get
           its metadata + PDF URL.
        2. Download the PDF bytes via `requests`.
        3. Extract plain text from the PDF using `pymupdf` (fitz).
        4. Return a LangChain `Document` with:
              - page_content : complete plain text of the paper
              - metadata     : title, authors, published date, arxiv_id, pdf_url

    This approach bypasses the broken `langchain-community` ArxivLoader
    (which was written for arxiv v1.x and incompatible with v4.x) and gives
    us full control over the extraction pipeline.
    """

    def __init__(self):
        self.client = arxiv.Client(
            page_size=1,
            delay_seconds=1.0,
            num_retries=3,
        )

    def load(self, arxiv_id: str) -> Document:
        """
        Fetch and return the full-text Document for a given arXiv paper.

        Args:
            arxiv_id: Short arXiv ID (e.g. "1706.03762").

        Returns:
            A LangChain Document:
                - page_content : Full extracted text of the paper.
                - metadata     : Dict with title, authors, published,
                                 arxiv_id, and pdf_url.

        Raises:
            ValueError:   If arxiv_id is empty or None.
            RuntimeError: If the paper cannot be found or PDF fails to parse.
        """
        if not arxiv_id or not arxiv_id.strip():
            raise ValueError("arxiv_id must be a non-empty string.")

        arxiv_id = arxiv_id.strip()

        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        results = list(self.client.results(search))

        if not results:
            raise RuntimeError(
                f"No paper found on arXiv for ID '{arxiv_id}'."
            )

        paper = results[0]
        pdf_url = paper.pdf_url

        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            pdf_bytes = response.content
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to download PDF for '{arxiv_id}' from {pdf_url}: {e}"
            ) from e

        try:
            pdf_doc = pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            full_text = "\n".join(
                page.get_text() for page in pdf_doc
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract text from PDF for '{arxiv_id}': {e}"
            ) from e

        metadata = {
            "title":     paper.title,
            "authors":   [str(a) for a in paper.authors],
            "published": paper.published.strftime("%Y-%m-%d"),
            "arxiv_id":  arxiv_id,
            "pdf_url":   pdf_url,
        }

        return Document(page_content=full_text, metadata=metadata)

    def get_page_content(self, arxiv_id: str) -> str:
        """
        Convenience method: returns only the raw text of the paper.

        Args:
            arxiv_id: Short arXiv ID of the paper.

        Returns:
            Full text of the paper as a plain string.
        """
        return self.load(arxiv_id).page_content


if __name__ == "__main__":
    TEST_ID = "1706.03762"

    loader = PaperLoader()
    doc = loader.load(TEST_ID)

    print("=" * 60)
    print(f"Title     : {doc.metadata['title']}")
    print(f"Published : {doc.metadata['published']}")
    print(f"PDF URL   : {doc.metadata['pdf_url']}")
    print("=" * 60)
    print(f"Text length: {len(doc.page_content):,} characters")
    print("-" * 60)
    print("First 1000 characters of page_content:")
    print(doc.page_content[:1000])
