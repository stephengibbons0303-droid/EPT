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
    bank.columns = [c.strip() for c in bank.columns]
    
    # 2. Filter by CEFR
    if 'CEFR rating' in bank.columns:
        relevant = bank[bank['CEFR rating'].astype(str).str.strip() == str(job['cefr']).strip()]
    else:
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
# Strategy: Sequential (3-Call) - NEW DEFAULT STRATEGY
# --------------------------------------------------------------------------

# Sequential Stage 1: Integrated Stem and Context Clue Construction
def create_sequential_stage1_prompt(job, example_banks):
    """
    Generates a complete sentence with the correct answer and an embedded context clue.
    The model must explicitly identify which element functions as the context clue.
    """
    examples = get_few_shot_examples(job, example_banks)
    
    system_msg = "You are an expert ELT content creator. Output ONLY valid JSON."
    
    # Parse context
    raw_context = job.get('context', 'General')
    main_topic = raw_context
    micro_style = "general conversation"
    
    if " (Style: " in raw_context:
        try:
            parts = raw_context.split(" (Style: ")
            main_topic = parts[0]
            micro_style = parts[1].replace(")", "")
        except:
            pass
    
    user_msg = f"""
    TASK: Generate a complete sentence containing the correct answer and an embedded context clue for a {job['cefr']} {job['type']} question.
    FOCUS: {job['focus']}
    TOPIC: {main_topic}
    
    INSTRUCTIONS:
    1. **STYLE/TONE (CRITICAL):** Write the sentence in the style of: **{micro_style}**.
    2. **INTEGRATED CONSTRUCTION:** First, place the correct answer within an authentic sentence appropriate to the {job['cefr']} level.
    3. **CONTEXT CLUE ENGINEERING:** The sentence MUST contain at least one linguistic element that logically constrains the answer choice. This context clue must be semantically integrated into the sentence's meaning, not merely adjacent to the gap.
    4. **METALINGUISTIC REFLECTION (REQUIRED):** You must explicitly identify which portion of the sentence functions as the context clue and explain why it eliminates alternative possibilities.
    5. **NEGATIVE CONSTRAINT (VERBOSITY):** Sentence must be concise (max 2 sentences). No preambles.
    6. **NEGATIVE CONSTRAINT (METALANGUAGE):** NEVER use grammar terminology in the sentence itself.
    
    Output Format:
    {{
      "Item Number": "{job['job_id']}",
      "Assessment Focus": "{job['focus']}",
      "Complete Sentence": "...[sentence with answer visible]...",
      "Correct Answer": "...",
      "Context Clue Location": "...[which phrase/clause]...",
      "Context Clue Explanation": "...[why this eliminates alternatives]...",
      "CEFR rating": "{job['cefr']}",
      "Category": "{job['type']}"
    }}
    
    REPLICATE THIS STYLE:
    {examples}
    """
    return system_msg, user_msg


# Sequential Stage 2: Distractor Generation Against Known Constraints
def create_sequential_stage2_prompt(job, stage1_output):
    """
    Generates three distractors that are invalidated by the identified context clue.
    Enforces 3-word maximum and requires explicit justification for each distractor.
    """
    system_msg = "You are an expert ELT test designer. Output ONLY valid JSON."
    
    user_msg = f"""
    TASK: Generate 3 distractors for a {job['cefr']} {job['type']} question with a known correct answer and context clue.
    
    INPUT FROM STAGE 1:
    {json.dumps(stage1_output, indent=2)}
    
    RULES:
    1. **WORD COUNT LIMIT (CRITICAL):** Each distractor must be MAXIMUM 3 words. This is non-negotiable.
    2. **GRAMMATICAL PARALLELISM:** All distractors must match the grammatical form of the correct answer.
    3. **CONTEXT CLUE AWARENESS:** Each distractor must be definitively eliminated by the context clue identified in Stage 1.
    4. **JUSTIFICATION REQUIRED:** For each distractor, explain why the specific context clue eliminates it as defensible.
    5. **PSYCHOMETRIC APPROPRIATENESS:** Distractors must represent plausible learner errors at the {job['cefr']} level - confusion patterns documented in second language acquisition.
    6. **NEGATIVE CONSTRAINT (LEXICAL OVERLAP):** Do not use any form of the correct answer word or its root in the distractors.
    
    Output Format:
    {{
      "Item Number": "{job['job_id']}",
      "Distractor A": "...[max 3 words]...",
      "Why A is Wrong": "...[explain how context clue eliminates it]...",
      "Distractor B": "...[max 3 words]...",
      "Why B is Wrong": "...[explain how context clue eliminates it]...",
      "Distractor C": "...[max 3 words]...",
      "Why C is Wrong": "...[explain how context clue eliminates it]..."
    }}
    """
    return system_msg, user_msg


# Sequential Stage 3: Quality Validation and Revision Trigger
def create_sequential_stage3_prompt(job, stage1_output, stage2_output):
    """
    Acts as quality assurance - evaluates complete question and flags issues.
    Adopts adversarial stance to test for ambiguity.
    """
    system_msg = "You are an independent quality assurance expert for language testing. Output ONLY valid JSON."
    
    # Construct the complete question for review
    complete_sentence = stage1_output.get("Complete Sentence", "")
    correct_answer = stage1_output.get("Correct Answer", "")
    context_clue = stage1_output.get("Context Clue Location", "")
    
    # Create the question prompt by replacing answer with gap
    question_prompt = complete_sentence.replace(correct_answer, "____")
    
    distractors = [
        stage2_output.get("Distractor A", ""),
        stage2_output.get("Distractor B", ""),
        stage2_output.get("Distractor C", "")
    ]
    
    user_msg = f"""
    TASK: Evaluate this complete question item for quality issues using an adversarial stance.
    
    COMPLETE QUESTION:
    Question Prompt: {question_prompt}
    Correct Answer: {correct_answer}
    Distractor 1: {distractors[0]}
    Distractor 2: {distractors[1]}
    Distractor 3: {distractors[2]}
    
    DECLARED CONTEXT CLUE: {context_clue}
    
    EVALUATION CRITERIA (test against negative constraints):
    1. **AMBIGUITY TEST:** Attempt to construct arguments for why a competent {job['cefr']} learner might reasonably select each distractor given the stem. Can you justify ANY distractor as defensible?
    2. **CONTEXT CLUE STRENGTH:** Does the identified context clue actually and unambiguously invalidate ALL distractors?
    3. **METALANGUAGE CHECK:** Does the question prompt use any grammar terminology?
    4. **VERBOSITY CHECK:** Is the prompt unnecessarily wordy or contain preambles?
    5. **LEXICAL OVERLAP CHECK:** Does the stem repeat words from the answer options?
    
    INSTRUCTIONS:
    - If you can construct a reasonable argument for ANY distractor, flag it as ambiguous
    - If ANY evaluation criterion fails, mark item as "Requires Revision"
    - Provide specific guidance on which element needs modification and why
    
    Output Format:
    {{
      "Item Number": "{job['job_id']}",
      "Overall Quality": "Pass" or "Requires Revision",
      "Ambiguity Issues": ["list any distractors that could be justified"],
      "Context Clue Assessment": "Strong/Weak/Absent - with explanation",
      "Other Issues": ["list any violations of metalanguage, verbosity, or lexical overlap constraints"],
      "Revision Recommendations": "Specific guidance if revision needed, or 'None' if passed"
    }}
    """
    return system_msg, user_msg


# --------------------------------------------------------------------------
# Strategy A: Holistic (1-Call)
# --------------------------------------------------------------------------
def create_holistic_prompt(job, example_banks):
    examples = get_few_shot_examples(job, example_banks)
    
    system_msg = "You are an expert ELT content creator. Output ONLY valid JSON."
    
    raw_context = job.get('context', 'General')
    main_topic = raw_context
    micro_style = "general conversation"

    if " (Style: " in raw_context:
        try:
            parts = raw_context.split(" (Style: ")
            main_topic = parts[0]
            micro_style = parts[1].replace(")", "")
        except:
            pass

    user_msg = f"""
    TASK: Generate a {job['cefr']} {job['type']} question.
    FOCUS: {job['focus']}
    TOPIC: {main_topic}
    
    INSTRUCTIONS:
    1. **STYLE/TONE (CRITICAL):** The question prompt must be written in the style of: **{micro_style}**.
    2. **NEGATIVE CONSTRAINT (AMBIGUITY - THE "CONTEXT CLUE" RULE):** If the question tests meaning (like frequency, verb tense, or vocabulary), you MUST provide a context clause that invalidates the distractors. 
       - *BAD:* "How often do you swim?" (Options: Always/Never) -> Both are possible.
       - *GOOD:* "I hate water, so I ____ swim." (Answer: Never) -> Context invalidates "Always".
    3. **NEGATIVE CONSTRAINT (VERBOSITY):** Prompts must be concise (max 2 sentences). No preambles like "Imagine you are..." or "Choose the best option...".
    4. **NEGATIVE CONSTRAINT (METALANGUAGE):** NEVER use grammar terminology (e.g., "Which sentence uses the present perfect?"). Test the *use* of the grammar, not the *name* of it.
    5. **NEGATIVE CONSTRAINT (LEXICAL OVERLAP):** Do not repeat the answer word in the question prompt.
    6. **WORD COUNT LIMIT:** Each answer option must be MAXIMUM 3 words.
    7. Create 4 plausible options (A, B, C, D) that are grammatically parallel.
    8. Ensure distractors are common learner errors.
    
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
    1. **WORD COUNT LIMIT (CRITICAL):** Each option must be MAXIMUM 3 words. This is non-negotiable.
    2. **NEGATIVE CONSTRAINT (LEXICAL OVERLAP):** Do not use any form of the core test word or its root in the options. Options must be varied.
    3. Provide 4 options (A, B, C, D) that are grammatically parallel.
    4. Indicate which one is the Correct Answer.
    5. The distractors must be plausible "near misses" or common errors appropriate for the CEFR level.
    
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


def create_stem_prompt(job, options_json_string):
    system_msg = "You are an expert ELT writer. Output ONLY valid JSON."
    
    raw_context = job.get('context', 'General')
    micro_style = "general conversation"
    if " (Style: " in raw_context:
        try:
            micro_style = raw_context.split(" (Style: ")[1].replace(")", "")
        except:
            pass
    
    user_msg = f"""
    TASK: Write a question stem that fits a specific set of options.
    
    INPUT OPTIONS:
    {options_json_string}
    
    INSTRUCTIONS:
    1. **STYLE/TONE (CRITICAL):** Write the sentence in the style of: **{micro_style}**.
    2. **NEGATIVE CONSTRAINT (AMBIGUITY / CONTEXT CLUE):** The stem must provide a clear context or clue that **logically invalidates ALL distractors**, leaving only the correct answer possible.
       - *Example:* If options are frequency adverbs (always/never), the stem MUST contain a phrase like "I hate it, so I..." to force a choice.
    3. **NEGATIVE CONSTRAINT (VERBOSITY/EXPLANATION):** Question stems must be concise (max 1-2 sentences) and must **NOT** contain preambles.
    4. **NEGATIVE CONSTRAINT (METALANGUAGE):** The prompt must **NEVER** use grammar terminology.
    5. **NEGATIVE CONSTRAINT (LEXICAL OVERLAP):** Do not repeat any words used in the answer options within the question stem (except function words).
    6. Analyze the 'Correct Answer' vs the distractors.
    7. Write a {job['cefr']} level sentence with a gap or transformation where ONLY the Correct Answer fits.
    
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
