from dotenv import load_dotenv
import streamlit as st
# Import the custom views you just created
from views.search_view import render_search_view
from views.ai_view import render_ai_view

load_dotenv()

st.set_page_config(page_title="AI Research Tool", layout="wide")

# Initialize session state variables
if "fetched_papers" not in st.session_state:
    st.session_state.fetched_papers = None
if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = None

# Routing logic based on session state
if st.session_state.selected_paper is not None:
    render_ai_view()
else:
    render_search_view()
    

# render_ai_view()