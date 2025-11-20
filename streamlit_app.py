import streamlit as st
import pandas as pd
import random
import json
import time
import test_planner
import prompt_engineer
import llm_service
import output_formatter

# -----------------------------------------------------------------
# App Configuration & Styling
# -----------------------------------------------------------------

st.set_page_config(
    page_title="Agentic Test Generator",
    layout="centered" 
)

# Load the key from Streamlit secrets
try:
    user_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.error("❌ OpenAI API Key not found in Secrets. Please add it to your Streamlit Cloud settings.")
    st.stop()
    
# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #191970 0%, #121245 50%, #191970 100%);
    }
    h1 { color: #FFFFFF !important; font-weight: 800 !important; }
    h2, h3 { color: #FFDB58 !important; }
    p, label, .stMarkdown { color: #FFFFFF !important; }
    
    /* Button styling - aggressive selectors to override all button text */
    .stButton>button {
        background-color: #FFDB58 !important;
        color: #151556 !important;
        border: 2px solid #191970 !important;
        border-radius: 8px !important; 
        padding: 12px 24px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 6px rgba(255, 219, 88, 0.5) !important;
    }
    .stButton>button:hover {
        background-color: #e5c350 !important;
        color: #151556 !important;
    }
    .stButton>button p {
        color: #151556 !important;
    }
    .stButton>button span {
        color: #151556 !important;
    }
    .stButton>button div {
        color: #151556 !important;
    }
    .stButton button {
        color: #151556 !important;
    }
    button[kind="primary"] {
        background-color: #FFDB58 !important;
        color: #151556 !important;
    }
    button[kind="primary"] p,
    button[kind="primary"] span,
    button[kind="primary"] div {
        color: #151556 !important;
    }
    
    /* Download buttons specifically */
    .stDownloadButton>button {
        background-color: #FFDB58 !important;
        color: #151556 !important;
    }
    .stDownloadButton>button p,
    .stDownloadButton>button span,
    .stDownloadButton>button div {
        color: #151556 !important;
    }
    .stDownloadButton>button:hover {
        background-color: #e5c350 !important;
        color: #151556 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #FFDB58 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    hr { border-color: #FFDB58 !important; }
    
    /* Alert/Info boxes - white background with dark text */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.95) !important; 
        border: 1px solid #FFDB58 !important;
        border-radius: 8px !important;
        color: #151556 !important;
    }
    .stAlert p, .stAlert div, .stAlert span {
        color: #151556 !important;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #FFDB58 !important; 
        border-radius: 8px !important;
        padding: 15px !important;
    }
    
    /* Expander - white background with dark text */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
    }
    .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid #FFDB58 !important;
    }
    .streamlit-expanderContent p, 
    .streamlit-expanderContent div,
    .streamlit-expanderContent span,
    .streamlit-expanderContent li {
        color: #151556 !important;
    }
    
    /* Input fields - white background with dark text */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>div,
    .stMultiSelect>div>div>div {
        background-color: #FFFFFF !important;
        color: #151556 !important;
    }
    
    /* Dropdown menus */
    [data-baseweb="select"] {
        background-color: #FFFFFF !important;
    }
    [data-baseweb="select"] span {
        color: #151556 !important;
    }
    
    /* Data editor/table */
    .stDataFrame {
        background-color: #FFFFFF !important;
    }
    .stDataFrame div[data-testid="stDataFrameResizable"] {
        color: #151556 !important;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: rgba(200, 255, 200, 0.95) !important;
        color: #151556 !important;
    }
    .stSuccess p, .stSuccess div {
        color: #151556 !important;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: rgba(255, 243, 205, 0.95) !important;
        color: #151556 !important;
    }
    .stWarning p, .stWarning div {
        color: #151556 !important;
    }
    
    /* Error messages */
    .stError {
        background-color: rgba(255, 200, 200, 0.95) !important;
        color: #151556 !important;
    }
    .stError p, .stError div {
        color: #151556 !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #FFFFFF !important;
    }
    .stRadio div[role="radiogroup"] label {
        color: #FFFFFF !important;
    }
    
    /* Captions */
    .stCaptionContainer, .caption {
        color: #CCCCCC !important;
    }
    
    /* Universal override for yellow buttons only */
    button[style*="background-color: rgb(255, 219, 88)"],
    button[style*="background-color:#FFDB58"],
    button[style*="background: rgb(255, 219, 88)"] {
        color: #151556 !important;
    }
    button[style*="background-color: rgb(255, 219, 88)"] *,
    button[style*="background-color:#FFDB58"] *,
    button[style*="background: rgb(255, 219, 88)"] * {
        color: #151556 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------
@st.cache_data
def load_example_banks():
    try:
        df_g = pd.read_csv("grammar_bank.csv")
        df_v = pd.read_csv("vocab_bank.csv")
        
        if "GSE Score" in df_g.columns:
            df_g = df_g.drop(columns=["GSE Score"])
        if "GSE Score" in df_v.columns:
            df_v = df_v.drop(columns=["GSE Score"])
            
        return {"grammar": df_g, "vocab": df_v}
    except FileNotFoundError:
        st.error("Error: Example bank CSVs not found.")
        return None
    except Exception as e:
        st.error(f"Error loading CSVs: {e}")
        return None

# -----------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------
@st.cache_data
def get_focus_options(q_type, cefr):
    if q_type == "Grammar":
        if cefr == "A1":
            return [
                "Present Simple ('be'/'have')", 
                "Prepositions of Time ('on'/'in'/'at')", 
                "Prepositions of Place ('on'/'in'/'at')",
                "Possessive Adjectives", 
                "Articles (a/an/the)",
                "this/that/these/those",
                "Plurals (regular/irregular)",
                "Modals ('can'/'can't' for ability)"
            ]
        elif cefr == "A2":
            return [
                "Past Simple (regular/irregular)", 
                "Countable/Uncountable Nouns (some/any)", 
                "Comparatives & Superlatives", 
                "Present Continuous",
                "Future ('going to' vs. 'will')",
                "like vs. would like",
                "Adverbs of Frequency",
                "Modals ('should'/'have to' for advice/obligation)"
            ]
        elif cefr == "B1":
            return [
                "Past Simple vs. Present Perfect", 
                "Conditionals (Type 1 & 2)", 
                "Modals of Obligation (must/have to/should)", 
                "Reported Speech (basic statements/questions)", 
                "Passive Voice (simple present/past)",
                "Gerunds & Infinitives (basic)",
                "Future Continuous",
                "Common Phrasal Verbs"
            ]
        elif cefr == "B2":
            return [
                "Conditionals (Type 3 & Mixed)", 
                "Passive (Causative - have/get something done)",
                "Passive (all tenses)",
                "Modals of Speculation (past/present)", 
                "Relative Clauses (defining/non-defining)", 
                "Reported Speech (advanced - suggest, advise)",
                "Future Perfect",
                "Gerunds & Infinitives (after specific verbs/prepositions)"
            ]
        elif cefr == "C1":
            return [
                "Inversion (e.g., 'Not only...')", 
                "Conditionals (Advanced Mixed, implied)", 
                "Passive (Advanced Forms, impersonal)", 
                "Modals (subtle meaning, nuance)", 
                "Future (Future Perfect Continuous)",
                "Cleft Sentences (e.g., 'What I need is...')",
                "Ellipsis",
                "Advanced Phrasal Verbs & Idioms"
            ]
    
    if q_type == "Vocabulary":
        if cefr == "A1":
            return [
                "Category Membership", 
                "Basic Antonym", 
                "Meaning-in-Sentence (Context Clue)", 
                "Basic Collocation (e.g., 'have breakfast')"
            ]
        elif cefr == "A2":
            return [
                "Meaning-in-Sentence (Context Clue)", 
                "Collocation (Verb+Noun)", 
                "Word Form (noun/verb/adj)", 
                "Functional Usage (e.g., 'What for?')", 
                "Basic Synonym"
            ]
        elif cefr == "B1":
            return [
                "Meaning-in-Sentence (Inference)", 
                "Collocation (Adverb+Adj)", 
                "Word Form (Affixes - un, re, able)", 
                "Functional Usage (e.g., 'I'd rather...')", 
                "Phrasal Verbs (common, separable/inseparable)"
            ]
        elif cefr == "B2":
            return [
                "Synonym (subtle difference)", 
                "Collocation (idiomatic, e.g., 'take into account')", 
                "Functional Usage (formal/informal register)", 
                "Phrasal Verbs (less common)", 
                "Word Form (noun/adj suffixes -tion, -ive)"
            ]
        elif cefr == "C1":
            return [
                "Synonym (high-level, low-frequency)", 
                "Idiomatic Expressions", 
                "Functional Usage (advanced nuance, persuasion)", 
                "Register Trap (formal vs. academic)", 
                "Collocation (academic, e.g., 'conduct research')"
            ]
    
    return ["No options loaded for this level"]

@st.cache_data
def get_topic_suggestions(cefr):
    if cefr == "A1":
        return ["Personal Information", "Family", "Food & Drink", "My Home", "Days & Times"]
    elif cefr == "A2":
        return ["Daily Routines", "Past Holidays", "Shopping", "Friends & Hobbies", "My Town", "Jobs"]
    elif cefr == "B1":
        return ["Work & Jobs", "The Environment", "Travel & Tourism", "Technology", "Health & Fitness", "Education"]
    elif cefr == "B2":
        return ["Media & News", "Crime & Society", "The Future", "Education Systems", "Business & Finance", "Global Issues"]
    elif cefr == "C1":
        return ["Philosophy & Ethics", "Scientific Research", "Global Politics", "Art & Literature", "Psychology"]
    
    return ["No topics loaded for this level"]

# Initialize session state
if 'last_batch' not in st.session_state:
    st.session_state.last_batch = None
if 'last_batch_strategy' not in st.session_state:
    st.session_state.last_batch_strategy = None
if 'sequential_stage1_data' not in st.session_state:
    st.session_state.sequential_stage1_data = None
if 'sequential_stage2_data' not in st.session_state:
    st.session_state.sequential_stage2_data = None
if 'sequential_stage3_data' not in st.session_state:
    st.session_state.sequential_stage3_data = None

# -----------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------

example_banks = load_example_banks()

st.title("🤖 AI Test Question Generator")

if example_banks is None:
    st.error("STOP: Failed to load example banks.")
    st.stop()

st.write("This tool uses a modular AI pipeline to generate test questions based on your exact specifications.")

tab1, tab2 = st.tabs(["🚀 Generator", "🔧 Refinement Workshop"])

# =============================
# TAB 1: GENERATOR
# =============================
with tab1:
    st.header("Batch Generation Settings")

    col1, col2 = st.columns(2)
    with col1:
        q_type = st.selectbox(
            "Question Type",
            ("Grammar", "Vocabulary"),
            key="q_type"
        )
        
        cefr = st.session_state.get('cefr', 'A1') 
        q_type_key = st.session_state.get('q_type', 'Grammar')
        
        focus_options = get_focus_options(q_type_key, cefr)
        selected_focus = st.multiselect(
            "Assessment Focus (Select one or more)",
            focus_options,
            key="assessment_focus"
        )

    with col2:
        cefr = st.selectbox(
            "CEFR Target",
            ("A1", "A2", "B1", "B2", "C1"),
            key="cefr"
        )

        strategy = st.selectbox(
            "Generation Strategy",
            ("Sequential (3-Call)", "Holistic (1-Call)", "Segmented (2-Call)"),
            help="Sequential: Highest quality. Holistic: Fast. Segmented: Options first, then Stem.",
            key="strategy"
        )

        batch_size = st.selectbox(
            "Batch Size",
            (1, 5, 10, 20, 30, 40, 50),
            index=2,
            key="batch_size"
        )
    
    st.divider()

    st.subheader("Context & Topic")
    context_topic = st.text_input(
        "Optional: Enter a specific context or topic",
        placeholder="e.g., 'A business email' or 'A story about a holiday'"
    )
    
    current_cefr = st.session_state.get('cefr', 'A1')
    with st.expander(f"View suggested topics for {current_cefr}..."):
        suggestions = get_topic_suggestions(current_cefr)
        st.info(" - " + "\n - ".join(suggestions))
    
    st.divider()

    if st.button("Generate Batch", type="primary", use_container_width=True):
        if not selected_focus:
            st.error("Please select at least one 'Assessment Focus'.")
        else:
            with st.spinner(f"Generating {batch_size} questions..."):
                try:
                    job_list = test_planner.create_job_list(
                        total_questions=batch_size,
                        q_type=q_type,
                        cefr_target=cefr,
                        selected_focus_list=selected_focus,
                        context_topic=context_topic if context_topic else "General",
                        generation_strategy=strategy 
                    )
                    
                    st.success(f"Planner created {len(job_list)} jobs!")
                    st.subheader("Planned Job List:")
                    st.dataframe(pd.DataFrame(job_list))
                    
                    if not user_api_key:
                        st.error("⛔ No API Key provided.")
                    else:
                        generated_questions = []
                        stage1_data_list = []
                        stage2_data_list = []
                        stage3_data_list = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for index, job in enumerate(job_list):
                            status_text.text(f"Generating question {index + 1} of {len(job_list)}...")
                            
                            if job['strategy'] == "Sequential (3-Call)":
                                # Stage 1: Stem + Context Clue
                                sys_msg_1, user_msg_1 = prompt_engineer.create_sequential_stage1_prompt(job, example_banks)
                                raw_stage1 = llm_service.call_llm([sys_msg_1, user_msg_1], user_api_key)
                                stage1_data, stage1_error = output_formatter.parse_response(raw_stage1)
                                
                                if stage1_error:
                                    st.error(f"Job {job['job_id']} Failed at Stage 1: {stage1_error}")
                                    continue
                                
                                stage1_data_list.append(stage1_data)
                                
                                # Stage 2: Distractors
                                sys_msg_2, user_msg_2 = prompt_engineer.create_sequential_stage2_prompt(job, stage1_data)
                                raw_stage2 = llm_service.call_llm([sys_msg_2, user_msg_2], user_api_key)
                                stage2_data, stage2_error = output_formatter.parse_response(raw_stage2)
                                
                                if stage2_error:
                                    st.error(f"Job {job['job_id']} Failed at Stage 2: {stage2_error}")
                                    continue
                                
                                stage2_data_list.append(stage2_data)
                                
                                # Stage 3: Quality Validation
                                sys_msg_3, user_msg_3 = prompt_engineer.create_sequential_stage3_prompt(job, stage1_data, stage2_data)
                                raw_stage3 = llm_service.call_llm([sys_msg_3, user_msg_3], user_api_key)
                                stage3_data, stage3_error = output_formatter.parse_response(raw_stage3)
                                
                                if stage3_error:
                                    st.error(f"Job {job['job_id']} Failed at Stage 3: {stage3_error}")
                                    continue
                                
                                stage3_data_list.append(stage3_data)
                                
                                # Construct final question
                                complete_sentence = stage1_data.get("Complete Sentence", "")
                                correct_answer = stage1_data.get("Correct Answer", "")
                                question_prompt = complete_sentence.replace(correct_answer, "____")
                                
                                options = [
                                    stage2_data.get("Distractor A", ""),
                                    stage2_data.get("Distractor B", ""),
                                    stage2_data.get("Distractor C", ""),
                                    correct_answer
                                ]
                                random.shuffle(options)
                                correct_letter = chr(65 + options.index(correct_answer))
                                
                                final_question = {
                                    "Item Number": stage1_data.get("Item Number", ""),
                                    "Assessment Focus": stage1_data.get("Assessment Focus", ""),
                                    "Question Prompt": question_prompt,
                                    "Answer A": options[0],
                                    "Answer B": options[1],
                                    "Answer C": options[2],
                                    "Answer D": options[3],
                                    "Correct Answer": correct_letter,
                                    "CEFR rating": stage1_data.get("CEFR rating", ""),
                                    "Category": stage1_data.get("Category", "")
                                }
                                generated_questions.append(final_question)
                                
                            elif job['strategy'] == "Segmented (2-Call)":
                                sys_msg_1, user_msg_1 = prompt_engineer.create_options_prompt(job, example_banks)
                                raw_options = llm_service.call_llm([sys_msg_1, user_msg_1], user_api_key)
                                options_data, options_error = output_formatter.parse_response(raw_options)
                                
                                if options_error:
                                    st.error(f"Job {job['job_id']} Failed at Options stage: {options_error}")
                                    continue
                                
                                options_json_string = json.dumps(options_data)
                                sys_msg_2, user_msg_2 = prompt_engineer.create_stem_prompt(job, options_json_string)
                                raw_response = llm_service.call_llm([sys_msg_2, user_msg_2], user_api_key)
                                question_data, error = output_formatter.parse_response(raw_response)
                                
                                if error:
                                    st.error(f"Job {job['job_id']} Failed: {error}")
                                else:
                                    generated_questions.append(question_data)
                                    
                            else:  # Holistic
                                sys_msg, user_msg = prompt_engineer.create_holistic_prompt(job, example_banks)
                                raw_response = llm_service.call_llm([sys_msg, user_msg], user_api_key)
                                question_data, error = output_formatter.parse_response(raw_response)
                                
                                if error:
                                    st.error(f"Job {job['job_id']} Failed: {error}")
                                else:
                                    generated_questions.append(question_data)

                            progress_bar.progress((index + 1) / len(job_list))
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if generated_questions:
                            st.success(f"Successfully generated {len(generated_questions)} questions!")
                            
                            final_df = pd.DataFrame(generated_questions)
                            st.dataframe(final_df)
                            
                            # Store in session state
                            st.session_state.last_batch = final_df
                            st.session_state.last_batch_strategy = strategy
                            
                            if strategy == "Sequential (3-Call)":
                                st.session_state.sequential_stage1_data = pd.DataFrame(stage1_data_list)
                                st.session_state.sequential_stage2_data = pd.DataFrame(stage2_data_list)
                                st.session_state.sequential_stage3_data = pd.DataFrame(stage3_data_list)
                            
                            csv = final_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Questions as CSV",
                                data=csv,
                                file_name=f"generated_test_{cefr}_{batch_size}q.csv",
                                mime="text/csv",
                            )
                    
                except Exception as e:
                    st.error(f"Error: {e}")


# =============================
# TAB 2: REFINEMENT WORKSHOP
# =============================
with tab2:
    st.header("🔧 Refinement Workshop")
    st.info("Edit and refine generated batches. Sequential batches show all 3 stages for review.")
    
    st.subheader("Select Input Source")
    
    input_source = st.radio(
        "Choose your batch source:",
        ("Recent batch from Generator", "Upload CSV file", "Manual text input"),
        key="input_source"
    )
    
    working_batch = None
    is_sequential_batch = False
    
    if input_source == "Recent batch from Generator":
        if st.session_state.last_batch is not None:
            st.success(f"✓ Recent batch loaded: {len(st.session_state.last_batch)} questions")
            st.caption(f"Strategy used: {st.session_state.last_batch_strategy}")
            
            working_batch = st.session_state.last_batch.copy()
            is_sequential_batch = (st.session_state.last_batch_strategy == "Sequential (3-Call)")
        else:
            st.warning("No recent batch found. Please generate a batch in the Generator tab first.")
    
    elif input_source == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                working_batch = pd.read_csv(uploaded_file)
                st.success(f"✓ File uploaded: {len(working_batch)} questions")
                is_sequential_batch = False
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    else:  # Manual input
        st.text_area("Paste question data (JSON format)", height=200, key="manual_input")
        if st.button("Load Manual Input"):
            st.info("Manual input parsing not yet implemented.")
    
    st.divider()
    
    # Display editing interface
    if working_batch is not None:
        if is_sequential_batch:
            st.subheader("📊 Sequential Pipeline View (3 Stages)")
            st.caption("View and edit outputs from each stage")
            
            st.markdown("### Stage 1: Stem + Context Clue")
            if st.session_state.sequential_stage1_data is not None:
                edited_stage1 = st.data_editor(
                    st.session_state.sequential_stage1_data,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="stage1_editor"
                )
                
                csv1 = edited_stage1.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Stage 1 Data",
                    data=csv1,
                    file_name="stage1_output.csv",
                    mime="text/csv",
                    key="download_stage1"
                )
            
            st.divider()
            
            st.markdown("### Stage 2: Distractors")
            if st.session_state.sequential_stage2_data is not None:
                edited_stage2 = st.data_editor(
                    st.session_state.sequential_stage2_data,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="stage2_editor"
                )
                
                csv2 = edited_stage2.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Stage 2 Data",
                    data=csv2,
                    file_name="stage2_output.csv",
                    mime="text/csv",
                    key="download_stage2"
                )
            
            st.divider()
            
            st.markdown("### Stage 3: Quality Validation")
            if st.session_state.sequential_stage3_data is not None:
                edited_stage3 = st.data_editor(
                    st.session_state.sequential_stage3_data,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="stage3_editor"
                )
                
                csv3 = edited_stage3.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Stage 3 Data",
                    data=csv3,
                    file_name="stage3_output.csv",
                    mime="text/csv",
                    key="download_stage3"
                )
            
            st.divider()
            
            st.markdown("### Final Generated Questions")
            edited_final = st.data_editor(
                working_batch,
                use_container_width=True,
                num_rows="dynamic",
                key="final_editor"
            )
            
            csv_final = edited_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Final Questions",
                data=csv_final,
                file_name="final_questions.csv",
                mime="text/csv",
                key="download_final"
            )
        
        else:
            st.subheader("📝 Simple Edit Mode")
            st.caption("Edit your batch directly. Changes can be downloaded below.")
            
            edited_batch = st.data_editor(
                working_batch,
                use_container_width=True,
                num_rows="dynamic",
                key="simple_editor"
            )
            
            csv = edited_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Edited Batch",
                data=csv,
                file_name="edited_questions.csv",
                mime="text/csv",
                key="download_edited"
            )
