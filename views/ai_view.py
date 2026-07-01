import streamlit as st
from llm import PromptGenerator, ModelSelection
from src import format_llm_output
# from src import papers as mock_papers


test_output = '''
BERT’s pre‑training objective is a joint cross‑entropy over two tasks: masked language modeling (MLM) and next‑sentence prediction (NSP). For a token sequence (x=(x_1,\dots ,x_T)) and a set of masked positions (\mathcal{M}\subseteq{1,\dots ,T}), the MLM loss is

[ \mathcal{L}{\text{MLM}}(x)= -\sum{t\in\mathcal{M}}\log P_\theta(x_t\mid x_{\setminus t}), ]

where (P_\theta) is the softmax output of the Transformer encoder parameterised by (\theta). The NSP loss is

[ \mathcal{L}{\text{NSP}}(x)= -\log P\theta(\text{next}\mid x), ]

and the total loss is (\mathcal{L}=\mathcal{L}{\text{MLM}}+\mathcal{L}{\text{NSP}}). The authors argue that the original BERT training schedule—limited steps, modest batch size, and a suboptimal learning‑rate schedule—leads to under‑training, i.e., the model does not fully explore the parameter space that would minimise (\mathcal{L}).

The paper’s mathematical contribution lies in a systematic re‑engineering of the optimisation pipeline. They employ the AdamW optimiser with update rule

[ \theta_{t+1}= \theta_t-\alpha_t\frac{m_t}{\sqrt{v_t}+\epsilon}-\alpha_t\lambda\theta_t, ]

where (\alpha_t) follows a linear warm‑up followed by cosine decay, (m_t) and (v_t) are the first‑ and second‑moment estimates, and (\lambda) is a weight‑decay coefficient. Gradient clipping is applied by normalising the gradient vector to a maximum (L_2) norm (\tau). Crucially, the authors replace NSP with a “dynamic masking” strategy: each training batch re‑samples the mask positions (\mathcal{M}) on the fly, ensuring that the model sees a richer distribution of masked tokens. They also increase the batch size (B) (e.g., (B=256) or larger) and extend the total number of training steps (S) (often to (S\approx 1,\text{M})), thereby allowing the optimiser to converge to a lower (\mathcal{L}).

These modifications, while conceptually simple, yield a robust optimisation trajectory. By decoupling the learning‑rate schedule from the batch size, applying weight decay directly to the parameters, and removing the NSP objective (which has been shown to provide limited signal), the training process becomes more stable and efficient. Empirically, the resulting “Roberta” model surpasses all subsequent post‑BERT methods on standard benchmarks, demonstrating that careful tuning of the optimisation hyper‑parameters can unlock performance gains that were previously unattainable with the vanilla BERT recipe.
'''

def update_llm_settings():
    st.session_state.llm_settings['model'] = st.session_state.llm_model_key
    st.session_state.llm_settings['temperature'] = st.session_state.llm_temp_key
    st.session_state.llm_settings['explanation_type'] = (
        st.session_state.explanation_type_key.lower().replace(" ", "_")
    )
    st.session_state.llm_settings['explanation_length'] = (
        st.session_state.explanation_length_key.lower().replace(" ", "_")
    )
    st.session_state.llm_submitted = True

def render_ai_view():
    paper = st.session_state.selected_paper
    # paper = mock_papers[0]
    # st.set_page_config(page_title=paper['title'], layout="wide")
    
    if "llm_settings" not in st.session_state:
        st.session_state.llm_settings = {
            "model": "openai/gpt-oss-20b:free",
            "temperature": 0.7,
            "explanation_type": "beginner_friendly",
            "explanation_length": "short"
        }
        st.session_state.llm_submitted = False
    
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
                    "Select LLM Model", ["openai/gpt-oss-20b:free", "zai-org/GLM-5.2"], key="llm_model_key"
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
                
    if st.session_state.llm_submitted:
        with st.spinner("Generating explanation..."):
            # st.info("Generating your explanation here soon...")
            # model_selection = ModelSelection(st.session_state.llm_settings)
            # llm = model_selection.get_model()
            # pg = PromptGenerator(paper, st.session_state.llm_settings)
            # prompt_str = pg.build_explanation_prompt()

            # result = llm.invoke(prompt_str)
            
            formatted_output = format_llm_output(test_output)

            left, middle, right = st.columns([1, 3, 1])
            with middle:
                st.markdown(formatted_output, unsafe_allow_html=True)
    else:
        st.info("Configure your settings above and click **Ask AI** to generate an explanation.")
        
        
        
    
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
            