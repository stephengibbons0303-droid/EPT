import json
import pandas as pd


# --------------------------------------------------------------------------
# Helper: Get Examples
# --------------------------------------------------------------------------
def get_few_shot_examples(job, example_banks):
    """
    Retrieves 2-3 examples from the CSV based on CEFR and Type.
    Robustly handles column name mismatches (e.g., extra spaces).
    """
    bank = example_banks.get(job['type'].lower())
    if bank is None or bank.empty: 
        return ""
    
    bank.columns = [c.strip() for c in bank.columns]
    
    if 'CEFR rating' in bank.columns:
        relevant = bank[bank['CEFR rating'].astype(str).str.strip() == str(job['cefr']).strip()]
    else:
        relevant = bank

    if len(relevant) >= 2:
        samples = relevant.sample(2)
    elif len(bank) >= 2:
        samples = bank.sample(2) 
    else:
        return "" 

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
# Strategy: Sequential BATCH MODE (3-Call) - FINAL VERSION
# --------------------------------------------------------------------------

def create_sequential_batch_stage1_prompt(job_list, example_banks):
    """
    Generates complete sentences with correct answers and context clues for ALL jobs at once.
    Includes question-type-aware constraints for grammatical vs semantic exclusivity.
    """
    examples = get_few_shot_examples(job_list[0], example_banks) if job_list else ""
    
    system_msg = f"""You are an expert ELT content creator. You will generate exactly {len(job_list)} complete test questions in a single JSON response. 

CRITICAL: Your entire response must be a JSON object with a "questions" key containing an array of exactly {len(job_list)} question objects. Do not generate fewer questions than requested."""
    
    # Build the batch specification
    job_specs = []
    has_grammar_distinction = False
    has_vocabulary = False
    
    for job in job_list:
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
        
        job_specs.append({
            "job_id": job['job_id'],
            "cefr": job['cefr'],
            "type": job['type'],
            "focus": job['focus'],
            "topic": main_topic,
            "style": micro_style
        })
        
        if job['type'] == 'Grammar' and 'vs' in job['focus'].lower():
            has_grammar_distinction = True
        if job['type'] == 'Vocabulary':
            has_vocabulary = True
    
    # Determine appropriate constraint instruction
    constraint_instruction = ""
    
    if has_grammar_distinction:
        constraint_instruction += """
GRAMMATICAL EXCLUSIVITY RULE (for grammar distinction questions):
When the Assessment Focus contains "vs" (e.g., "going to vs will", "Past Simple vs Present Perfect"), 
the Complete Sentence MUST include a GRAMMATICAL SIGNAL that makes only the correct answer structurally valid.

Examples of grammatical signals:
- Time markers: "yesterday" forces Past Simple, eliminates Present Perfect
- Evidence phrases: "Look at those clouds" forces "going to", eliminates "will"
- Hypothetical markers: "If I were you" forces Type 2 conditional, eliminates Type 1
- Duration markers: "for five years" forces Present Perfect, eliminates Past Simple
- Definiteness markers: "already" forces Present Perfect, eliminates Past Simple

The distractors should be GRAMMATICALLY INCOMPATIBLE with the sentence structure, not merely semantically weaker.
"""
    
    if has_vocabulary:
        constraint_instruction += """
SEMANTIC EXCLUSIVITY RULE (for vocabulary questions):
The Complete Sentence must contain SEMANTIC CONTEXT CLUES that make only the correct answer logically appropriate.

Context clue strategies by level:
- A1-A2: Category membership, clear antonyms, basic collocations (verb-noun pairings)
- B1-B2: Connotation distinctions, phrasal verb meanings, word form requirements, collocation violations
- C1: Precise semantic distinctions, idiomatic expressions, academic collocations

The distractors should be SEMANTICALLY INCOMPATIBLE or IDIOMATICALLY WRONG with the context, even if grammatically valid.
"""
    
    user_msg = f"""
TASK: Create exactly {len(job_list)} complete, original test questions from scratch.

You must generate ALL {len(job_list)} questions in this single response. Each question specification below MUST have a corresponding question in your output.

JOB SPECIFICATIONS (one question for each):
{json.dumps(job_specs, indent=2)}

{constraint_instruction}

GENERATION INSTRUCTIONS FOR EACH QUESTION:
1. **ANTI-REPETITION (CRITICAL):** Each question must have a UNIQUE topic and scenario. Do NOT reuse themes, contexts, or vocabulary across questions.
2. **STYLE/TONE:** Write each sentence in the specified style from the job specifications.
3. **INTEGRATED CONSTRUCTION:** Place the correct answer within an authentic sentence appropriate to the CEFR level.
4. **CONTEXT CLUE ENGINEERING:** Each sentence MUST contain at least one linguistic element that logically constrains the answer. The context clue must be semantically integrated.
5. **METALINGUISTIC REFLECTION (REQUIRED):** Explicitly identify which portion functions as the context clue and explain why it eliminates alternatives.
6. **NEGATIVE CONSTRAINT (VERBOSITY):** Sentences must be concise (max 2 sentences). No preambles. Do NOT use imperative commands.
7. **NEGATIVE CONSTRAINT (METALANGUAGE):** NEVER use grammar terminology in the sentence itself.

MANDATORY OUTPUT FORMAT:
{{
  "questions": [
    {{
      "Item Number": "...",
      "Assessment Focus": "...",
      "Complete Sentence": "...[sentence with answer visible]...",
      "Correct Answer": "...",
      "Context Clue Location": "...[which phrase/clause]...",
      "Context Clue Explanation": "...[why this eliminates alternatives]...",
      "CEFR rating": "...",
      "Category": "..."
    }},
    {{
      "Item Number": "...",
      "Assessment Focus": "...",
      "Complete Sentence": "...",
      "Correct Answer": "...",
      "Context Clue Location": "...",
      "Context Clue Explanation": "...",
      "CEFR rating": "...",
      "Category": "..."
    }}
    ... (continue until you have exactly {len(job_list)} question objects)
  ]
}}

VERIFICATION: Count your question objects before submitting. You must have exactly {len(job_list)} items in the "questions" array.

STYLE REFERENCE (format guide only - do not copy scenarios):
{examples}
"""
    return system_msg, user_msg


def create_sequential_batch_stage2_prompt(job_list, stage1_outputs):
    """
    Generates distractors for ALL questions at once, ensuring variety across the batch.
    ENHANCED: Enforces Assessment Focus alignment - distractors must match the grammatical category being tested.
    """
    system_msg = f"""You are an expert ELT test designer. You will generate distractors for exactly {len(job_list)} questions in a single JSON response with a "distractors" key."""
    
    user_msg = f"""
TASK: Generate 3 distractors for ALL {len(job_list)} questions.

INPUT FROM STAGE 1 (Complete sentences with correct answers):
{json.dumps(stage1_outputs, indent=2)}

CRITICAL RULE - ASSESSMENT FOCUS ALIGNMENT:
Each distractor set MUST align with the Assessment Focus specified in Stage 1. Distractors must test the SAME grammatical feature or semantic category as the correct answer.

Grammar Distinction Examples:
- Focus: "Future ('going to' vs 'will')" → Distractors MUST be future forms: will, going to, won't, will not
- Focus: "Past Simple vs Present Perfect" → Distractors MUST contrast these tenses: did/have done, went/have gone
- Focus: "Conditionals (Type 1 & 2)" → Distractors MUST be conditional forms: will/would, can/could
- Focus: "Modals of Obligation" → Distractors MUST be obligation modals: must, have to, should, ought to

Vocabulary Examples:
- Focus: "Synonym distinction" → Distractors MUST be near-synonyms testing subtle meaning differences
- Focus: "Collocation" → Distractors MUST be verbs/prepositions that violate the collocation pattern
- Focus: "Phrasal Verbs" → Distractors MUST be other phrasal verbs with different particles

DO NOT MIX GRAMMAR CATEGORIES. If testing future tense, do not include obligation modals as distractors.

RULES FOR EACH QUESTION:
1. **ASSESSMENT FOCUS ALIGNMENT (CRITICAL):** All distractors must come from the same grammatical/semantic category as the correct answer.
2. **WORD COUNT LIMIT (CRITICAL):** Each distractor must be MAXIMUM 3 words. This is non-negotiable.
3. **GRAMMATICAL PARALLELISM:** All distractors must match the grammatical form of the correct answer.
   - If correct answer is two words ("going to"), distractors must also be two words ("will be", "won't be")
   - If correct answer is a single modal, distractors should be single modals from the same category
4. **CONTEXT CLUE AWARENESS:** Each distractor must be definitively eliminated by the context clue identified in Stage 1.
5. **JUSTIFICATION REQUIRED:** For each distractor, explain why the specific context clue eliminates it.
6. **PSYCHOMETRIC APPROPRIATENESS:** Distractors must represent plausible learner errors at the specified CEFR level.
7. **NEGATIVE CONSTRAINT (LEXICAL OVERLAP):** Do not use any form of the correct answer word or its root in the distractors.
8. **ANTI-REPETITION:** Avoid using the same distractor words across multiple questions in this batch.

MANDATORY OUTPUT FORMAT:
{{
  "distractors": [
    {{
      "Item Number": "...",
      "Distractor A": "...[max 3 words]...",
      "Why A is Wrong": "...",
      "Distractor B": "...[max 3 words]...",
      "Why B is Wrong": "...",
      "Distractor C": "...[max 3 words]...",
      "Why C is Wrong": "..."
    }},
    ... (exactly {len(job_list)} distractor sets)
  ]
}}

VERIFICATION: You must generate exactly {len(job_list)} distractor sets, one for each question from Stage 1. Each distractor set must align with its corresponding Assessment Focus.
"""
    return system_msg, user_msg


def create_sequential_batch_stage3_prompt(job_list, stage1_outputs, stage2_outputs):
    """
    Quality validation for ALL questions at once, can identify cross-question issues.
    ENHANCED: Specifically checks for Assessment Focus alignment in distractor selection.
    """
    system_msg = f"""You are an independent quality assurance expert for language testing. You will evaluate exactly {len(job_list)} questions and return your assessments in a JSON object with a "validations" key."""
    
    # Construct complete questions for review
    complete_questions = []
    for i, (job, s1, s2) in enumerate(zip(job_list, stage1_outputs, stage2_outputs)):
        complete_sentence = s1.get("Complete Sentence", "")
        correct_answer = s1.get("Correct Answer", "")
        question_prompt = complete_sentence.replace(correct_answer, "____")
        
        complete_questions.append({
            "Item Number": s1.get("Item Number", ""),
            "Assessment Focus": s1.get("Assessment Focus", ""),
            "Question Prompt": question_prompt,
            "Correct Answer": correct_answer,
            "Distractor 1": s2.get("Distractor A", ""),
            "Distractor 2": s2.get("Distractor B", ""),
            "Distractor 3": s2.get("Distractor C", ""),
            "Context Clue": s1.get("Context Clue Location", ""),
            "CEFR": job['cefr']
        })
    
    user_msg = f"""
TASK: Evaluate ALL {len(job_list)} complete question items for quality issues.

COMPLETE QUESTIONS BATCH:
{json.dumps(complete_questions, indent=2)}

EVALUATION CRITERIA FOR EACH QUESTION:
1. **AMBIGUITY TEST:** Can you construct arguments for why a competent learner might reasonably select ANY distractor?
2. **CONTEXT CLUE STRENGTH:** Does the identified context clue actually and unambiguously invalidate ALL distractors?
3. **ASSESSMENT FOCUS ALIGNMENT:** Do all distractors align with the Assessment Focus? (e.g., if testing "going to vs will", are distractors all future forms, or do they incorrectly include modals of obligation?)
4. **GRAMMATICAL CATEGORY CONSISTENCY:** If testing a grammar distinction, are all options from the same grammatical category?
5. **METALANGUAGE CHECK:** Does the question prompt use any grammar terminology?
6. **VERBOSITY CHECK:** Is the prompt unnecessarily wordy or contain preambles?
7. **LEXICAL OVERLAP CHECK:** Does the stem repeat words from the answer options?
8. **CROSS-QUESTION CHECK:** Are there any repeated themes or excessive similarity between questions in this batch?

INSTRUCTIONS:
- If ANY evaluation criterion fails, mark item as "Requires Revision"
- Specifically flag if distractors are from the wrong grammatical/semantic category for the Assessment Focus
- Flag any cross-question repetition issues
- Provide specific guidance on which element needs modification

MANDATORY OUTPUT FORMAT:
{{
  "validations": [
    {{
      "Item Number": "...",
      "Overall Quality": "Pass" or "Requires Revision",
      "Ambiguity Issues": ["list any distractors that could be justified"],
      "Context Clue Assessment": "Strong/Weak/Absent - with explanation",
      "Assessment Focus Alignment": "Correct/Incorrect - note if distractors are from wrong category",
      "Other Issues": ["list any violations"],
      "Cross-Question Issues": ["note any similarities to other questions in batch"],
      "Revision Recommendations": "Specific guidance or 'None'"
    }},
    ... (exactly {len(job_list)} validation reports)
  ]
}}

VERIFICATION: You must provide exactly {len(job_list)} validation reports.
"""
    return system_msg, user_msg


# --------------------------------------------------------------------------
# Legacy/Fallback Strategies
# --------------------------------------------------------------------------

def create_sequential_stage1_prompt(job, example_banks):
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
    
    # Determine constraint type
    constraint_text = ""
    if job['type'] == 'Grammar' and 'vs' in job.get('focus', '').lower():
        constraint_text = """
**GRAMMATICAL EXCLUSIVITY:** This question tests a grammar distinction. Include a grammatical signal 
(time marker, evidence phrase, or structural constraint) that makes only the correct answer structurally valid.
"""
    elif job['type'] == 'Vocabulary':
        constraint_text = """
**SEMANTIC EXCLUSIVITY:** Include context clues that make only the correct answer semantically/idiomatically appropriate.
"""
    
    user_msg = f"""
TASK: Generate a complete sentence containing the correct answer and an embedded context clue for a {job['cefr']} {job['type']} question.
FOCUS: {job['focus']}
TOPIC: {main_topic}

{constraint_text}

INSTRUCTIONS:
1. **STYLE/TONE (CRITICAL):** Write the sentence in the style of: **{micro_style}**.
2. **INTEGRATED CONSTRUCTION:** First, place the correct answer within an authentic sentence appropriate to the {job['cefr']} level.
3. **CONTEXT CLUE ENGINEERING:** The sentence MUST contain at least one linguistic element that logically constrains the answer choice.
4. **METALINGUISTIC REFLECTION (REQUIRED):** Explicitly identify which portion functions as the context clue and explain why.
5. **NEGATIVE CONSTRAINT (VERBOSITY):** Sentence must be concise (max 2 sentences). No preambles.
6. **NEGATIVE CONSTRAINT (METALANGUAGE):** NEVER use grammar terminology in the sentence itself.

Output Format:
{{
  "Item Number": "{job['job_id']}",
  "Assessment Focus": "{job['focus']}",
  "Complete Sentence": "...",
  "Correct Answer": "...",
  "Context Clue Location": "...",
  "Context Clue Explanation": "...",
  "CEFR rating": "{job['cefr']}",
  "Category": "{job['type']}"
}}

REPLICATE THIS STYLE:
{examples}
"""
    return system_msg, user_msg


def create_sequential_stage2_prompt(job, stage1_output):
    system_msg = "You are an expert ELT test designer. Output ONLY valid JSON."
    
    # Determine focus alignment requirement
    focus = job.get('focus', '')
    alignment_note = ""
    if 'vs' in focus.lower():
        alignment_note = f"""
CRITICAL - ASSESSMENT FOCUS ALIGNMENT:
This question tests "{focus}". All distractors must come from the same grammatical category as the correct answer.
For example, if testing future forms, all options must be future forms (will, going to, won't, will not).
"""
    
    user_msg = f"""
TASK: Generate 3 distractors for a {job['cefr']} {job['type']} question.

INPUT FROM STAGE 1:
{json.dumps(stage1_output, indent=2)}

{alignment_note}

RULES:
1. **WORD COUNT LIMIT (CRITICAL):** Each distractor must be MAXIMUM 3 words.
2. **GRAMMATICAL PARALLELISM:** All distractors must match the grammatical form of the correct answer.
3. **CONTEXT CLUE AWARENESS:** Each distractor must be eliminated by the context clue from Stage 1.
4. **JUSTIFICATION REQUIRED:** Explain why each distractor is wrong.
5. **PSYCHOMETRIC APPROPRIATENESS:** Represent plausible learner errors.
6. **NEGATIVE CONSTRAINT (LEXICAL OVERLAP):** Avoid using the correct answer word or its root.

Output Format:
{{
  "Item Number": "{job['job_id']}",
  "Distractor A": "...",
  "Why A is Wrong": "...",
  "Distractor B": "...",
  "Why B is Wrong": "...",
  "Distractor C": "...",
  "Why C is Wrong": "..."
}}
"""
    return system_msg, user_msg


def create_sequential_stage3_prompt(job, stage1_output, stage2_output):
    system_msg = "You are an independent quality assurance expert. Output ONLY valid JSON."
    
    complete_sentence = stage1_output.get("Complete Sentence", "")
    correct_answer = stage1_output.get("Correct Answer", "")
    context_clue = stage1_output.get("Context Clue Location", "")
    question_prompt = complete_sentence.replace(correct_answer, "____")
    assessment_focus = stage1_output.get("Assessment Focus", "")
    
    distractors = [
        stage2_output.get("Distractor A", ""),
        stage2_output.get("Distractor B", ""),
        stage2_output.get("Distractor C", "")
    ]
    
    user_msg = f"""
TASK: Evaluate this question for quality issues.

Assessment Focus: {assessment_focus}
Question: {question_prompt}
Correct: {correct_answer}
Distractors: {', '.join(distractors)}
Context Clue: {context_clue}

EVALUATE: 
- Ambiguity
- Context clue strength
- Assessment Focus alignment (are distractors from the correct grammatical/semantic category?)
- Metalanguage
- Verbosity
- Lexical overlap

Output Format:
{{
  "Item Number": "{job['job_id']}",
  "Overall Quality": "Pass" or "Requires Revision",
  "Ambiguity Issues": [],
  "Context Clue Assessment": "...",
  "Assessment Focus Alignment": "...",
  "Other Issues": [],
  "Revision Recommendations": "..."
}}
"""
    return system_msg, user_msg


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
1. **STYLE/TONE:** Write in the style of: **{micro_style}**.
2. **CONTEXT CLUE RULE:** Provide context that invalidates distractors.
3. **VERBOSITY:** Max 2 sentences. No preambles.
4. **METALANGUAGE:** Never use grammar terminology.
5. **LEXICAL OVERLAP:** Don't repeat answer word in prompt.
6. **WORD LIMIT:** Each option max 3 words.
7. Create 4 parallel options (A, B, C, D).
8. Distractors should be common learner errors.

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
1. **WORD LIMIT:** Each option max 3 words.
2. **NO LEXICAL OVERLAP:** Don't use test word or root in options.
3. Provide 4 parallel options (A, B, C, D).
4. Indicate correct answer.
5. Distractors should be plausible errors for CEFR level.

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
TASK: Write a question stem for these options.

OPTIONS: {options_json_string}

INSTRUCTIONS:
1. **STYLE:** Write in style of: **{micro_style}**.
2. **CONTEXT CLUE:** Provide context that invalidates ALL distractors.
3. **VERBOSITY:** Max 1-2 sentences. No preambles.
4. **METALANGUAGE:** Never use grammar terminology.
5. **NO LEXICAL OVERLAP:** Don't repeat option words in stem.
6. Write {job['cefr']} level sentence where ONLY correct answer fits.

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
