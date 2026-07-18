import streamlit as st
from src import PaperFetcher


def render_search_view():
    col1, main_col, col3 = st.columns([1, 3, 1])

    with main_col:
        st.title("AI Enhanced Research Tool")

        with st.form("user_input_form_col"):
            user_query = st.text_input(
                "Enter keywords or a paper title to search arXiv."
            )
            submit = st.form_submit_button("🔍 Search arXiv")

        if submit and user_query:
            st.session_state.selected_paper = None  # Reset selection on new search

            with st.spinner("Fetching papers from arXiv…"):
                pf = PaperFetcher(max_results=10)
                try:
                    st.session_state.fetched_papers = pf.fetch_papers(user_query)
                except RuntimeError as err:
                    st.error(f"Failed to fetch papers: {err}")
                    st.session_state.fetched_papers = []

        # Render results if they exist in state
        if st.session_state.get("fetched_papers"):
            papers = st.session_state.fetched_papers
            st.write(f"**{len(papers)} result(s)** found on arXiv for your query.")
            print(papers[0].get("arxiv_id", ""))
    
            for idx, paper in enumerate(papers):
                each = st.container(border=True, key=f"paper_{idx}", gap="small")

                each.subheader(f"[{paper['title']}]({paper['link']})")
                
                # Authors — truncate long lists
                authors = paper["authors"]
                author_str = (
                    ", ".join(authors[:5]) + (f" +{len(authors)-5} more" if len(authors) > 5 else "")
                )
                each.write(f"**Authors:** {author_str}")

                # Metadata row
                meta_cols = each.columns(4)
                meta_cols[0].write(f"📅 **Published:** {paper['published']}")
                meta_cols[1].write(f"🏷️ **Category:** `{paper['venue']}`")
                meta_cols[2].write(
                    f"📂 **All tags:** {', '.join(f'`{c}`' for c in paper['categories'][:4])}"
                )
                meta_cols[3].write(f"🔗 **arXiv ID:** `{paper['arxiv_id']}`")

                # Abstract (collapsible for long ones)
                with each.expander("Abstract", expanded=True):
                    st.write(paper["abstract"])

                # Action buttons
                btn_col1, btn_col2, btn_col3 = each.columns([0.35, 0.35, 0.30])
                btn_col1.link_button(
                    "📄 Abstract Page",
                    url=paper["link"],
                    use_container_width=True,
                )
                btn_col2.link_button(
                    "⬇️ Download PDF",
                    url=paper["pdf_url"],
                    use_container_width=True,
                )
                btn_col3.button(
                    key=f"paper_{idx}_ai",
                    label="🤖 ASK AI",
                    on_click=lambda p=paper: st.session_state.update(
                        {"selected_paper": p}
                    ),
                    use_container_width=True,
                    type="primary",
                )
