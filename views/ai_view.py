import streamlit as st

def render_ai_view():
    paper = st.session_state.selected_paper
    
    if st.button("⬅ Back to Search Results"):
        st.session_state.selected_paper = None
        # st.clear_cache()
        st.rerun()
        
    st.title("🤖 AI Research Explainer")
    st.header(paper['title'])
    st.write(f"**Authors:** {', '.join(paper['authors'])}")
    st.write(f"**Venue:** {paper['venue']} ({paper['year']})")
    
    st.subheader("Abstract")
    st.write(paper['abstract'])
    
    st.divider()
    st.subheader("Ask questions about this paper")
    
    user_question = st.chat_input("Ask me to summarize methods, findings, etc...")
    if user_question:
        st.chat_message("user").write(user_question)
        with st.chat_message("assistant"):
            st.write("Generating your explanation here soon...")