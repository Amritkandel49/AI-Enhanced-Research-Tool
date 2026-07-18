from paper_loader import PaperLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextProcessor:
    def __init__(self, arxiv_id: str):
        self.arxiv_id = arxiv_id
    
    def fetch_and_process_text(self):
        paper_loader = PaperLoader()
        paper_text = paper_loader.get_page_content(self.arxiv_id)
        
        paper_text = paper_text[2000:2000].replace("\r", "")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200 
        )
        chunks = text_splitter.split_text(paper_text)
        
        return [Document(page_content=chunk) for chunk in chunks]



if __name__ == "__main__":
    arxiv_id = "1706.03762"  # Example arXiv ID
    processor = TextProcessor(arxiv_id)
    documents = processor.fetch_and_process_text()
    print(documents)
    print("*" * 40)
    print(f"Number of chunks: {len(documents)}")
    for i, doc in enumerate(documents):
        if i<4:
            print(f"Chunk {i+1}:\n{doc}\n{'-'*40}")