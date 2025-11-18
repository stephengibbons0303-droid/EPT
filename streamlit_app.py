import streamlit as st
import pandas as pd
import random

# -----------------------------------------------------------------
# App Configuration & Styling
# -----------------------------------------------------------------

# Set page config (centered layout, not wide)
st.set_page_config(
    page_title="Agentic Test Generator",
    layout="centered" 
)

# Custom CSS for your color scheme
def apply_custom_css():
    st.markdown(f"""
    <style>
        /* Main title color */
        .stApp h1 {{
            color: #191970; /* Midnight Blue */
        }}

        /* Button color */
        .stButton > button {{
            background-color: #FFDB58; /* Gold */
            color: #191970; /* Midnight Blue Text */
            border: 2px solid #191970; /* Midnight Blue Border */
            font-weight: bold;
        }}

        /* Sidebar and tab headers (optional, but good for branding) */
        .stTabs [data-baseweb="tab"] {{
            background-color: #f0f2f6;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background-color: #FFFFFF;
            border-bottom: 2px solid #FFDB58; /* Gold underline */
        }}

    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------
# Placeholder Data & Functions (Mocking our modules)
# -----------------------------------------------------------------

@st.cache_data
def get_focus_options(q_type, cefr):
    """
    MOCK FUNCTION: This will eventually be powered by a CSV or 
    our data_loader.py to get real options.
    """
    if q_type == "Grammar":
        if cefr == "A1":
            return ["Present Simple ('be'/'have')", "Prepositions of Time ('on'/'in')", "Possessive Adjectives"]
        elif cefr == "A2":
            return ["Past Simple (irregular)", "Countable/Uncountable", "Comparatives", "Present Continuous"]
        elif cefr == "B1":
            return ["Past Simple vs. Present Perfect", "Conditionals (Type 1)", "Modals of Obligation", "Reported Speech (basic)"]
        elif cefr == "B2":
            return ["Conditionals (Mixed)", "Passive (Causative)", "Modals (Speculation)", "Relative Clauses (advanced)"]
        else:
            return ["Inversion", "Conditionals (Advanced Mixed)", "Passive (Advanced Forms)"]
    
    if q_type == "Vocabulary":
        if cefr == "A1" or cefr == "A2":
            return ["Meaning-in-Sentence", "Collocation (Verb+Noun)", "Word Form (noun/verb/adj)", "Category Membership", "Basic Antonym"]
        elif cefr == "B1":
            return ["Meaning-in-Sentence (Inference)", "Collocation (Adverb+Adj)", "Word Form (Affixes)", "Functional Usage"]
        else:
            return ["Synonym (subtle difference)", "Collocation (idiomatic)", "Functional Usage (formal/informal)", "Register Trap"]

@st.cache_data
def get_topic_suggestions(cefr):
    """
    MOCK FUNCTION: Returns a list of suggested topics.
    """
    if cefr == "A1":
        return ["Personal Information", "Family", "Food & Drink", "My Home"]
    elif cefr == "A2":
        return ["Daily Routines", "Past Holidays", "Shopping", "Friends & Hobbies"]
    elif cefr == "B1":
        return ["Work & Jobs", "The Environment", "Travel & Tourism", "Technology", "Health"]
    elif cefr == "B2":
        return ["Media & News", "Crime & Society", "The Future", "Education Systems"]
    else:
        return ["Philosophy & Ethics", "Scientific Research", "Global Politics", "Art & Literature"]

# -----------------------------------------------------------------
# Main Streamlit UI
# -----------------------------------------------------------------

# Apply the custom colors
apply_custom_css()

st.title("🤖 AI Test Question Generator")
st.write("This tool uses a modular AI pipeline to generate test questions based on your exact specifications.")

# Create the two tabs based on your design
tab1, tab2 = st.tabs(["🚀 Generator (Phase 1)", "🔧 Expert Controls (Phase 2)"])

# --- Tab 1: The Main Generator ---
with tab1:
    st.header("Batch Generation Settings")

    col1, col2 = st.columns(2)
    with col1:
        # 1. Question Type
        q_type = st.selectbox(
            "Question Type",
            ("Grammar", "Vocabulary"),
            key="q_type"
        )
        
        # 3. Assessment Focus (Multi-Select)
        # This is now dependent on q_type and cefr
        cefr = st.session_state.get('cefr', 'A1') # Get CEFR from session state
        q_type_key = st.session_state.get('q_type', 'Grammar') # Get Type from session state
        
        focus_options = get_focus_options(q_type_key, cefr)
        selected_focus = st.multiselect(
            "Assessment Focus (Select one or more)",
            focus_options,
            key="assessment_focus"
        )

    with col2:
        # 2. CEFR Target
        cefr = st.selectbox(
            "CEFR Target",
            ("A1", "A2", "B1", "B2", "C1"),
            key="cefr"
        )

        # 6. Batch Size
        batch_size = st.selectbox(
            "Batch Size",
            (1, 5, 10, 20, 30, 40, 50),
            index=2,  # Defaults to 10
            key="batch_size"
        )
    
    st.divider()

    # 4. & 5. Context/Topic and Suggestions
    st.subheader("Context & Topic")
    context_topic = st.text_input(
        "Optional: Enter a specific context or topic",
        placeholder="e.g., 'A business email' or 'A story about a holiday'"
    )
    
    with st.expander(f"View suggested topics for {cefr}...") :
        suggestions = get_topic_suggestions(cefr)
        st.info(" - " + "\n - ".join(suggestions))
    
    st.divider()
    
    # 7. Generate Button
    if st.button("Generate Batch", type="primary", use_container_width=True):
        # --- This is where we call the backend ---
        # 1. Validate inputs
        if not selected_focus:
            st.error("Please select at least one 'Assessment Focus'.")
        else:
            with st.spinner(f"Generating {batch_size} questions... This may take a moment."):
                # --- MOCKUP of the backend call ---
                # In a real app, we would call our `test_planner` and
                # loop, calling the `llm_service` for each job.
                
                # We'll just show the "Job List" for this demo
                job_list = []
                for i in range(batch_size):
                    job_list.append({
                        "job_id": i+1,
                        "type": q_type,
                        "cefr": cefr,
                        "focus": random.choice(selected_focus),
                        "context": context_topic
                    })
                
                st.success("Batch Generation Complete!")
                st.subheader("Planned Job List (Demo)")
                st.write("This is the list of jobs our 'Test Planner' created. The app would now generate one question for each job.")
                st.dataframe(pd.DataFrame(job_list))


# --- Tab 2: The Expert Controls ---
with tab2:
    st.header("🔧 Expert Refinement Controls")
    st.info("This section will hold the granular controls from your framework (Groups A-D) to refine single, existing questions.")
    
    st.subheader("A. Structural Complexity Controls")
    st.slider("Sentence length slider (words)", 6, 40, 15)
    st.select_slider("Clause complexity slider", ["simple", "complex", "embedded"])
    
    st.subheader("C. Distractor Complexity Controls")
    st.checkbox("Near-synonym trap")
    st.checkbox("Collocation trap")
    st.slider("Distractor Plausibility", 0.0, 1.0, 0.5)
