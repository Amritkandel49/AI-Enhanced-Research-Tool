import json
import sys

from dotenv import load_dotenv

from src.fetch_papers import PaperFetcher


def main() -> None:
    load_dotenv()

    query = " ".join(sys.argv[1:]).strip() or "Perception of physical properties"
    fetcher = PaperFetcher(max_results=5)
    papers = fetcher.fetch_papers(query)

    print(json.dumps(papers, indent=2))


if __name__ == "__main__":
    main()