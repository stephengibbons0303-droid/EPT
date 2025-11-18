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
# Strategy A: Holistic (1-Call)
# --------------------------------------------------------------------------
def create_holistic_prompt(job, example_banks):
    examples = get_few_shot_examples(job, example_banks)
    
    system_msg = "You are an expert ELT content creator. Output ONLY valid JSON."
    
    user_msg = f"""
    TASK: Generate a {job['cefr']} {job['type']} question.
    FOCUS: {job['focus']}
    TOPIC: {job['context']}
    
    INSTRUCTIONS:
    1. Create 4 plausible options (A, B, C, D).
    2. Write a question stem where ONLY one option is correct.
    3. Ensure distractors are common learner errors.
    
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

# Step 1: Generate Options Only
def create_options_prompt(job, example_banks):
    system_msg = "You are an expert ELT test designer. Output ONLY valid JSON."
    
    user_msg = f"""
    TASK: Generate 4 answer choices for a {job['cefr']} {job['type']} question.
    FOCUS: {job['focus']}
    TOPIC: {job['context']}
    
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

# Step 2: Generate Stem from Options
def create_stem_prompt(job, options_json_string):
    system_msg = "You are an expert ELT writer. Output ONLY valid JSON."
    
    user_msg = f"""
    TASK: Write a question stem that fits a specific set of options.
    
    INPUT OPTIONS:
    {options_json_string}
    
    INSTRUCTIONS:
    1. Analyze the 'Correct Answer' vs the distractors.
    2. Write a {job['cefr']} level sentence with a gap (or a question) where ONLY the Correct Answer fits.
    3. The distractors must be clearly wrong in this specific context (grammatically or logically).
    4. TOPIC: {job['context']}
    
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
