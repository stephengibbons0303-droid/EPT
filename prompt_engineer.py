import json
import pandas as pd


# --------------------------------------------------------------------------
# Helper: Get Examples (FIXED: Robust Column Handling)
# --------------------------------------------------------------------------
def get_few_shot_examples(job, example_banks):
    """
    Retrieves 2-3 examples from the CSV based on CEFR and Type.
    Robustly handles column name mismatches (e.g., extra spaces).
    """
    bank = example_banks.get(job['type'].lower())
    if bank is None or bank.empty: 
        return ""
    
    # 1. Clean column names to ensure we can find 'CEFR rating'
    # This fixes the "KeyError" if your CSV has "CEFR rating " (with a space)
    bank.columns = [c.strip() for c in bank.columns]
    
    # 2. Filter by CEFR
    if 'CEFR rating' in bank.columns:
        # Force both sides to string and strip whitespace for safe comparison
        relevant = bank[bank['CEFR rating'].astype(str).str.strip() == str(job['cefr']).strip()]
    else:
        # If column is truly missing, just use random examples rather than crashing
        relevant = bank

    # 3. Sample logic
    if len(relevant) >= 2:
        samples = relevant.sample(2)
    elif len(bank) >= 2:
        samples = bank.sample(2) 
    else:
        return "" 

    # 4. Format as string
    output = ""
    for _, row in samples.iterrows():
        # Use .get() to safely retrieve values even if columns are missing
        ex_dict = {
            "Question Prompt": row.get("Question Prompt", "N/A"),
            "Answer A": row.get("Answer A", "N/A"),
            "Answer B": row.get("Answer B", "N/A"),
            "Answer C": row.get("Answer C", "N/A"),
            "Answer D": row.get("Answer D", "N/A"),
            "Correct Answer": row.get("Correct Answer", "N/A")
        }
        output += "### EXAMPLE:\n" + json.dumps(ex_dict) + "\n\n"
    return output

# --------------------------------------------------------------------------
# Strategy A: Holistic (1-Call) - FIXED AND ENHANCED
# --------------------------------------------------------------------------
def create_holistic_prompt(job, example_banks):
    examples = get_few_shot_examples(job, example_banks)
    
    system_msg = "You are an expert ELT content creator. Output ONLY valid JSON."
    
    # --- ROBUST CONTEXT PARSING (FIXES NAMEERROR) ---
    raw_context = job.get('context', 'General')
    main_topic = raw_context
    micro_style = "general conversation" # Default style

    if " (Style: " in raw_context:
        try:
            parts = raw_context.split(" (Style: ")
            main_topic = parts[0]
            micro_style = parts[1].replace(")", "")
        except:
            pass # Fallback to defaults if parsing fails

    # --- END OF PARSING ---

    user_msg = f"""
    TASK: Generate a {job['cefr']} {job['type']} question.
    FOCUS: {job['focus']}
    TOPIC: {main_topic}
    
    INSTRUCTIONS:
    1. **STYLE/TONE (CRITICAL):** The question prompt must be written in the style of: **{micro_style}**.
    2. **NEGATIVE CONSTRAINT (VERBOSITY/EXPLANATION):** Question prompts must be concise (max 1-2 sentences) and must **NOT** be explanatory, introductory, or descriptive (e.g., avoid preambles like "Imagine your friend said...").
    3. **NEGATIVE CONSTRAINT (METALANGUAGE):** The question prompt must **NEVER** use grammar terminology (e.g., 'relative clause,' 'past participle,' 'non-defining'). Focus on simple gap-fills or direct transformations.
    4. **NEGATIVE CONSTRAINT (LEXICAL OVERLAP):** Do not repeat the core test word, its root, or topic-specific low-frequency vocabulary in both the prompt and the options.
    5. Create 4 plausible options (A, B, C, D) that are grammatically parallel.
    6. Ensure distractors are common learner errors.
    
    Output Format:
    {{
      "Item Number": "{job['job_id']}",
      "Assessment Focus": "{job['focus']}",
      "Question Prompt": "...",
      "Answer A": "...",
      "Answer B": "...",
      "Answer C": "...",
      "Answer D": "...",
      "Correct Answer": "...",
      "CEFR rating": "{job['cefr']}",
      "Category": "{job['type']}"
    }}
    
    REPLICATE THIS STYLE:
    {examples}
    """
    return system_msg, user_msg

# --------------------------------------------------------------------------
# Strategy B: Segmented (2-Call)
# --------------------------------------------------------------------------

# Step 1: Generate Options Only - ENHANCED FOR VARIANCE
def create_options_prompt(job, example_banks):
    system_msg = "You are an expert ELT test designer. Output ONLY valid JSON."
    
    raw_context = job.get('context', 'General')
    main_topic = raw_context
    if " (Style: " in raw_context:
        try:
            main_topic = raw_context.split(" (Style: ")[0]
        except:
            pass

    user_msg = f"""
    TASK: Generate 4 answer choices for a {job['cefr']} {job['type']} question.
    FOCUS: {job['focus']}
    TOPIC: {main_topic}
    
    RULES:
    1. **NEGATIVE CONSTRAINT (LEXICAL OVERLAP):** Do not use any form of the core test word or its root in the options. Options must be varied.
    2. Provide 4 options (A, B, C, D) that are grammatically parallel.
    3. Indicate which one is the Correct Answer.
    4. The distractors must be plausible "near misses" or common errors appropriate for the CEFR level.
    
    Output Format:
    {{
      "Answer A": "...",
      "Answer B": "...",
      "Answer C": "...",
      "Answer D": "...",
      "Correct Answer": "A/B/C/D"
    }}
    """
    return system_msg, user_msg

# Step 2: Generate Stem from Options - ENHANCED FOR VARIANCE
def create_stem_prompt(job, options_json_string):
    system_msg = "You are an expert ELT writer. Output ONLY valid JSON."
    
    # --- ROBUST CONTEXT PARSING (Fixes potential NameError here too) ---
    raw_context = job.get('context', 'General')
    micro_style = "general conversation"
    if " (Style: " in raw_context:
        try:
            micro_style = raw_context.split(" (Style: ")[1].replace(")", "")
        except:
            pass
    # --- END OF PARSING ---
    
    user_msg = f"""
    TASK: Write a question stem that fits a specific set of options.
    
    INPUT OPTIONS:
    {options_json_string}
    
    INSTRUCTIONS:
    1. **STYLE/TONE (CRITICAL):** Write the sentence in the style of: **{micro_style}**.
    2. **NEGATIVE CONSTRAINT (VERBOSITY/EXPLANATION):** Question stems must be concise (max 1-2 sentences) and must **NOT** contain preambles (e.g., 'Imagine you are talking to...').
    3. **NEGATIVE CONSTRAINT (METALANGUAGE):** The prompt must **NEVER** use grammar terminology.
    4. **NEGATIVE CONSTRAINT (LEXICAL OVERLAP):** Do not repeat any words used in the answer options within the question stem (except for common function words like 'the', 'a', 'is').
    5. Analyze the 'Correct Answer' vs the distractors.
    6. Write a {job['cefr']} level sentence with a gap (or a direct transformation, like "John: 'What time is it?' Reported: ____") where ONLY the Correct Answer fits.
    
    Output Format:
    {{
      "Item Number": "{job['job_id']}",
      "Assessment Focus": "{job['focus']}",
      "Question Prompt": "...",
      "Answer A": "...",
      "Answer B": "...",
      "Answer C": "...",
      "Answer D": "...",
      "Correct Answer": "...",
      "CEFR rating": "{job['cefr']}",
      "Category": "{job['type']}"
    }}
    """
    return system_msg, user_msg
