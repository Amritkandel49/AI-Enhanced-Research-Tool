from .paper_loader import PaperLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextProcessor:
    """
    Fetches the full text of an arXiv paper and splits it into
    overlapping chunks ready to be embedded and stored in a vector store.
    """

    def __init__(self, arxiv_id: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            arxiv_id    : Short arXiv ID (e.g. "1706.03762").
            chunk_size  : Maximum character length of each chunk.
            chunk_overlap: Characters of overlap between adjacent chunks
                           to preserve context at boundaries.
        """
        self.arxiv_id = arxiv_id
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def fetch_and_process_text(self) -> list[Document]:
        """
        Fetches the full paper text, strips the first 2000 characters
        (which typically contain arXiv licence headers / author blocks),
        splits it into chunks, and returns a list of LangChain Documents.

        Returns:
            List of Document objects, each with page_content set to
            one text chunk and metadata containing the arxiv_id.
        """
        paper_loader = PaperLoader()
        paper_text = paper_loader.get_page_content(self.arxiv_id)

        # Skip the first ~2000 characters — typically licence header / author list
        paper_text = paper_text[2000:].replace("\r", "")

        chunks = self.text_splitter.split_text(paper_text)

        return [
            Document(page_content=chunk, metadata={"arxiv_id": self.arxiv_id})
            for chunk in chunks
        ]


if __name__ == "__main__":
    arxiv_id = "1706.03762"
    processor = TextProcessor(arxiv_id)
    documents = processor.fetch_and_process_text()

    print(f"Number of chunks: {len(documents)}")
    print("*" * 40)
    for i, doc in enumerate(documents[:4]):
        print(f"Chunk {i+1}:\n{doc.page_content}\n{'-'*40}")