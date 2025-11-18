import json
import pandas as pd

# --------------------------------------------------------------------------
# Helper: Get Examples
# --------------------------------------------------------------------------
def get_few_shot_examples(job, example_banks):
    """
    Retrieves 2-3 examples from the CSV based on CEFR and Type.
    """
    bank = example_banks.get(job['type'].lower())
    if bank is None: return ""
    
    # Filter by CEFR
    relevant = bank[bank['CEFR rating'] == job['cefr']]
    
    # Sample logic
    if len(relevant) >= 2:
        samples = relevant.sample(2)
    elif len(bank) >= 2:
        samples = bank.sample(2) 
    else:
        return "" 

    # Format as string
    output = ""
    for _, row in samples.iterrows():
        # Stripping metadata to focus on content
        ex_dict = {
            "Question Prompt": row.get("Question Prompt"),
            "Answer A": row.get("Answer A"),
            "Answer B": row.get("Answer B"),
            "Answer C": row.get("Answer C"),
            "Answer D": row.get("Answer D"),
            "Correct Answer": row.get("Correct Answer")
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
    2. Create 4 plausible options (A, B, C, D).
    3. Write a question stem where ONLY one option is correct.
    4. Ensure distractors are common learner errors.
    
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
    1. Provide 4 options (A, B, C, D).
    2. Indicate which one is the Correct Answer.
    3. The distractors must be plausible "near misses" or common errors.
    4. All options must be grammatically parallel (e.g. all verbs, or all nouns).
    
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
    2. Analyze the 'Correct Answer' vs the distractors.
    3. Write a {job['cefr']} level sentence with a gap (or a question) where ONLY the Correct Answer fits.
    4. The distractors must be clearly wrong in this specific context.
    
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
