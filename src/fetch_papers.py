import arxiv
import json


class PaperFetcher:
    """Fetches research papers from the arXiv API."""

    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.client = arxiv.Client(
            page_size=max_results,
            delay_seconds=1.0,
            num_retries=3,
        )

    def fetch_papers(self, query: str) -> list[dict]:
        """
        Search arXiv with the given query string and return structured paper data.

        Returned fields per paper:
          - title       : Paper title
          - authors     : List of author name strings
          - link        : Landing page URL (abs)
          - pdf_url     : Direct PDF link
          - venue       : arXiv category / primary category
          - year        : Publication year (from published date)
          - abstract    : Full abstract text
          - arxiv_id    : Short arXiv ID (e.g. "2310.12345")
          - categories  : All arXiv category tags
          - published   : Full ISO-8601 published date string
        """
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        try:
            for result in self.client.results(search):
                papers.append({
                    "title":      result.title,
                    "authors":    [str(a) for a in result.authors],
                    "link":       result.entry_id,          # abs URL
                    "pdf_url":    result.pdf_url,
                    "venue":      result.primary_category,  # e.g. "cs.LG"
                    "year":       str(result.published.year),
                    "abstract":   result.summary.replace("\n", " "),
                    "arxiv_id":   result.get_short_id(),
                    "categories": result.categories,        # list of tags
                    "published":  result.published.strftime("%Y-%m-%d"),
                })
        except Exception as e:
            raise RuntimeError(f"Error fetching papers from arXiv: {e}") from e

        return papers


if __name__ == "__main__":
    pf = PaperFetcher(max_results=5)
    data = pf.fetch_papers("Attention is all you need")
    print(json.dumps(data, indent=4))