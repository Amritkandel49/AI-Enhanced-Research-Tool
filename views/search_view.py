import streamlit as st
import asyncio
from src import PaperFetcher
from src import papers as mock_papers

def render_search_view():
    from src import PaperFetcher
    from src import papers as mock_papers
    st.title("AI Enhanced Research Tool")
    
    with st.form("user_input_form_col"):
        user_query = st.text_input("Enter the title of the research paper.")
        submit = st.form_submit_button("Fetch Papers")
        
    if submit and user_query:
        st.session_state.selected_paper = None  # Reset selection on new search
        
        # with st.spinner("Fetching papers..."):
        #     pf = PaperFetcher()
        #     st.session_state.fetched_papers = asyncio.run(pf.fetch_papers(user_query))
        
        # Fallback to your mock data
        st.session_state.fetched_papers = mock_papers

    # Render results if they exist in state
    if st.session_state.fetched_papers:
        st.write("Showing results for your query:")
        
        for idx, paper in enumerate(st.session_state.fetched_papers):
            each = st.container(border=True, key=f'paper_{idx}', gap="small")
            
            
            each.subheader(f"[{paper['title']}]({paper['link']})")
            each.write(f"Authors: {', '.join(paper['authors'])}")
            each.write(f"Venue: {paper['venue']} ({paper['year']})")
            each.write(f"Abstract: {paper['abstract']}")
            
            col1, col2 = each.columns([0.7, 0.3])
            with col1:
                col1.write(f"[Read Paper]({paper['link']})")
            
            with col2:
                each.button(
                    key=f"paper_{idx}_ai", 
                    label="ASK AI", 
                    on_click=lambda p=paper: st.session_state.update({"selected_paper": p}), 
                    use_container_width=True
                )
    
    # st.html(
    #     """
    #     <style>
    #         .stContainer {
    #             padding: 5rem;
    #             border-radius: 5px;
    #             height: 150px;
    #         }
    #         .stContainer [data-testid="stVerticalBlock"] {
    #             gap: 0.05rem;
    #         }

    #     </style>
    #     """
    # )
                
                
