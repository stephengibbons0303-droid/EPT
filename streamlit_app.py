import streamlit as st
import pandas as pd
import random
import json
import time
import test_planner

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
# MODULE 1: Data Loader (Loads your CSVs)
# -----------------------------------------------------------------
@st.cache_data  # <-- Caching is crucial!
def load_example_banks():
    """
    This is our real data_loader module.
    It loads the CSVs from the root folder.
    """
    try:
        # --- Make sure your filenames match these ---
        df_g = pd.read_csv("grammar_bank.csv")
        df_v = pd.read_csv("vocab_bank.csv")
        
        # We'll also remove the GSE Score column as we discussed
        if "GSE Score" in df_g.columns:
            df_g = df_g.drop(columns=["GSE Score"])
        if "GSE Score" in df_v.columns:
            df_v = df_v.drop(columns=["GSE Score"])
            
        return {"grammar": df_g, "vocab": df_v}
    except FileNotFoundError:
        st.error("Error: Example bank CSVs not found. Make sure 'grammar_bank.csv' and 'vocab_bank.csv' are in the same folder as the app.")
        return None # Return None to signal an error
    except Exception as e:
        st.error(f"Error loading CSVs: {e}")
        return None

# -----------------------------------------------------------------
# POPULATED FUNCTIONS (For UI Dropdowns)
# -----------------------------------------------------------------

@st.cache_data
def get_focus_options(q_type, cefr):
    """
    This is the new, EXPANDED function for all Assessment Focus options.
    """
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
    
    # Fallback in case something goes wrong
    return ["No options loaded for this level"]

@st.cache_data
def get_topic_suggestions(cefr):
    """
    This is the populated function for all your Topic Suggestions.
    """
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
    
    # Fallback
    return ["No topics loaded for this level"]

# -----------------------------------------------------------------
# Main Streamlit UI
# -----------------------------------------------------------------

# Apply the custom colors
apply_custom_css()

# Load the data ONCE at the start of the script.
example_banks = load_example_banks()

st.title("🤖 AI Test Question Generator")

# Check if data loaded
if example_banks is None:
    st.error("STOP: Failed to load example banks. Please check your CSV file names and restart the app.")
    st.stop() # Halts the app if the CSVs are missing
else:
    # This is just a quiet confirmation in the terminal,
    # or you can use st.success() if you prefer a UI message
    print("Example banks loaded successfully.")

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
        # We need to get the *current* state of the widgets
        cefr = st.session_state.get('cefr', 'A1') 
        q_type_key = st.session_state.get('q_type', 'Grammar')
        
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
            key="cefr",
            # This 'on_change' is a bit advanced but helps clear
            # the focus list when the CEFR level changes.
            # You may need a small callback function for it.
            # For now, we'll keep it simple.
        )

        # NEW: Strategy Selector
        strategy = st.selectbox(
            "Generation Strategy",
            ("Holistic (1-Call)", "Segmented (2-Call)"),
            help="Holistic: Fast. Segmented: High quality (Options first, then Stem).",
            key="strategy"
        )

        # ... existing Batch Size code ...
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
    
    # Get the *current* CEFR level for the expander title
    current_cefr = st.session_state.get('cefr', 'A1')
    with st.expander(f"View suggested topics for {current_cefr}...") :
        suggestions = get_topic_suggestions(current_cefr)
        st.info(" - " + "\n - ".join(suggestions))
    
    st.divider()

    
    # 7. Generate Button
    if st.button("Generate Batch", type="primary", use_container_width=True):
        # 1. Validate inputs
        if not selected_focus:
            st.error("Please select at least one 'Assessment Focus'.")
        else:
            with st.spinner(f"Generating {batch_size} questions..."):
                
                # --- CALL THE REAL PLANNER ---
                try:
                    # We pass the UI values directly to the planner module.
                    # Note: 'strategy' comes from the dropdown we added earlier.
                    job_list = test_planner.create_job_list(
                        total_questions=batch_size,
                        q_type=q_type,
                        cefr_target=cefr,
                        selected_focus_list=selected_focus,
                        context_topic=context_topic if context_topic else "General",
                        generation_strategy=strategy 
                    )
                    
                    st.success(f"Planner successfully created {len(job_list)} jobs!")
                    
                    # Display the plan (Debugging step - serves as confirmation)
                    st.subheader("Planned Job List:")
                    st.dataframe(pd.DataFrame(job_list))
                    
                    # --- NEXT STEP: THE LLM LOOP WILL GO HERE ---
                    
                except Exception as e:
                    st.error(f"Error in Test Planner: {e}")


# --- Tab 2: The Expert Controls ---
with tab2:
    st.header("🔧 Expert Refinement Controls")
    st.info("This section is planned for Phase 2. It will hold the granular controls (Groups A-D) to refine single, existing questions.")
    
    # We can lay out the *disabled* controls as a preview
    st.subheader("A. Structural Complexity Controls")
    st.slider("Sentence length slider (words)", 6, 40, 15, disabled=True)
    st.select_slider("Clause complexity slider", ["simple", "complex", "embedded"], disabled=True)
    
    st.subheader("C. Distractor Complexity Controls")
    st.checkbox("Near-synonym trap", disabled=True)
    st.checkbox("Collocation trap", disabled=True)
    st.slider("Distractor Plausibility", 0.0, 1.0, 0.5, disabled=True)
