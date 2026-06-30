from scholarly import scholarly, ProxyGenerator
from dotenv import load_dotenv
import os
import asyncio
import json

load_dotenv()


class PaperFetcher:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self._setup_proxy()

    def _setup_proxy(self):
        """Initialize and validate the proxy."""
        api_key = os.getenv("SCRAPERAPI_KEY")
        if not api_key:
            raise ValueError("SCRAPERAPI_KEY not set in environment")

        pg = ProxyGenerator()
        if not pg.ScraperAPI(api_key):
            raise RuntimeError("Proxy setup failed — check your ScraperAPI key or quota")

        scholarly.use_proxy(pg)

    def _blocking_fetch(self, query: str) -> list[dict]:
        """Synchronous fetch — runs in a thread to avoid blocking the event loop."""
        papers = []
        try:
            search_query = scholarly.search_pubs(query)
            print(next(search_query))  # Debug: print the first result to verify the search is working
            for _ in range(self.max_results):
                try:
                    paper = next(search_query)
                    bib = paper.get("bib", {})
                    papers.append({
                        "title":    bib.get("title", ""),
                        "authors":  bib.get("author", []),
                        "link":     paper.get("pub_url", ""),
                        "venue":    bib.get("venue", ""),
                        "year":     bib.get("pub_year", ""),
                        "abstract": bib.get("abstract", ""),
                    })
                except StopIteration:
                    break
        except Exception as e:
            raise RuntimeError(f"Error fetching papers: {e}") from e

        return papers

    async def fetch_papers(self, query: str) -> list[dict]:
        """Async entry point — offloads blocking work to a thread."""
        return await asyncio.to_thread(self._blocking_fetch, query)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    pf = PaperFetcher()
    data = asyncio.run(pf.fetch_papers("Attention is all you need"))
    print(json.dumps(data, indent=4)) 