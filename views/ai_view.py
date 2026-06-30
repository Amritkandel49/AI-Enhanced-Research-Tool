import streamlit as st
from src import papers as mock_papers
from llm import PromptGenerator

def update_llm_settings():
    st.session_state.llm_settings['model'] = st.session_state.llm_model_key
    st.session_state.llm_settings['temperature'] = st.session_state.llm_temp_key
    st.session_state.llm_settings['explanation_type'] = (
        st.session_state.explanation_type_key.lower().replace(" ", "_")
    )
    st.session_state.llm_settings['explanation_length'] = (
        st.session_state.explanation_length_key.lower().replace(" ", "_")
    )

def render_ai_view():
    # paper = st.session_state.selected_paper
    paper = mock_papers[0]
    # st.set_page_config(page_title=paper['title'], layout="wide")
    
    if "llm_settings" not in st.session_state:
        st.session_state.llm_settings = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "explanation_type": "beginner_friendly",
            "explanation_length": "short"
        }
    
    if st.button("⬅ Back to Search Results"):
        st.session_state.selected_paper = None
        # st.clear_cache()
        st.rerun()
        
    # st.title("AI Enhanced Research Tool")
    
    header = st.container(border=True, gap="small")
    with header:
        header_c1, header_c2 = header.columns([3,1])
        with header_c1:
            header_c1.title(paper['title'])
            
            header_c1.write(f"**Authors:** {', '.join(paper['authors'])}")
            header_c1.write(f"**Venue:** {paper['venue']} ({paper['year']})")
            
            header_c1.subheader("Abstract")
            header_c1.write(paper['abstract'])
            
            header_c1.link_button(
                label="Read Full Paper",
                url=paper['link'],
                type="primary")
            
        with header_c2:
            # header_c2.write(f"[Read Paper]({paper['link']})")
            header_c2_form = header_c2.form("llm_setting_form")
            
            with header_c2_form:
                llm_model = header_c2_form.selectbox(
                    "Select LLM Model", ["openai/gpt-oss-20b:free"], key="llm_model_key"
                )
                llm_temp = header_c2_form.slider(
                    "Set LLM Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, key="llm_temp_key"
                )
                explanation_type = header_c2_form.selectbox(
                    "Select Explanation Type",
                    ["Mathematics oriented", "Beginner Friendly", "Summary"],
                    key="explanation_type_key"
                )
                explanation_length = header_c2_form.selectbox(
                    "Select Explanation Length", ["Short", "Medium", "Long"], key="explanation_length_key"
                )

                submit_button = header_c2_form.form_submit_button(
                    "Ask AI",
                    width=200,
                    type="primary",
                    on_click=update_llm_settings
                )
                
    
    pg = PromptGenerator(paper, st.session_state.llm_settings)
    prompt_str = pg.build_explanation_prompt()
    
    
    
    # st.write("### Generated Prompt for LLM:")
    # st.code(prompt_str, language="json")
    
    # st.write(f'''
    #     LLM Settings:
    #     - Model: {st.session_state.llm_settings["model"]}
    #     - Temperature: {st.session_state.llm_settings["temperature"]}
    #     - Explanation Type: {st.session_state.llm_settings["explanation_type"].replace("_", " ").title()}
    #     - Explanation Length: {st.session_state.llm_settings["explanation_length"].replace("_", " ").title()}''')
    
    

    
   
    
    user_question = st.chat_input("Ask me to summarize methods, findings, etc...", disabled=True)
    # if user_question:
    #     st.chat_message("user").write(user_question)
    #     with st.chat_message("assistant"):
    #         st.write("Generating your explanation here soon...")
            