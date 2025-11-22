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
# Strategy: Sequential BATCH MODE (3-Call) - SPLIT ARCHITECTURE
# --------------------------------------------------------------------------

def create_sequential_batch_stage1_prompt(job_list, example_banks):
    """
    Generates complete sentences with correct answers and context clues for ALL jobs at once.
    ENHANCED: Includes multi-word phrase splitting strategy and distinguishes between 
    grammatical versus semantic constraint requirements.
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

3. **INTEGRATED CONSTRUCTION - GRAMMAR QUESTIONS:** For multi-word grammatical constructions being tested (such as "going to", "have to", "used to"), strategically position elements to create structural constraints. If testing "going to" versus "will", consider placing auxiliary verbs in the stem such as "It's _____ rain" where the contracted auxiliary eliminates "will" structurally. Multi-word answers should be separated across stem and answer slot when this creates grammatical enforcement.

4. **INTEGRATED CONSTRUCTION - VOCABULARY QUESTIONS:** Place the target vocabulary word within an authentic sentence that provides semantic context clues. Ensure the target word is at an appropriate lexical level for the CEFR rating. For higher-level vocabulary items, the sentence context must make the meaning clear enough that learners can discriminate it from phonetically similar alternatives.

5. **CONTEXT CLUE ENGINEERING - GRAMMAR QUESTIONS:** Include grammatical signals that structurally eliminate incorrect options. Time markers such as "yesterday" or "for five years", structural elements such as auxiliary verb placement, or syntactic requirements such as "enjoy" requiring a gerund create grammatical incompatibility rather than semantic implausibility. The context clue must make distractors grammatically wrong, not just semantically odd.

6. **CONTEXT CLUE ENGINEERING - VOCABULARY QUESTIONS:** Include semantic context clues that make only the correct answer logically appropriate while keeping all options grammatically valid. For verb collocations, ensure the full sentence structure eliminates incorrect verbs through constructions such as benefactive phrases. The context clue must make distractors semantically incompatible while remaining grammatically acceptable.

7. **METALINGUISTIC REFLECTION (REQUIRED):** Explicitly identify which portion functions as the context clue and explain whether it creates grammatical elimination or semantic elimination based on question type.

8. **NEGATIVE CONSTRAINT (VERBOSITY):** Sentences must be concise (max 2 sentences). No preambles. Do NOT use imperative commands.

9. **NEGATIVE CONSTRAINT (METALANGUAGE):** NEVER use grammar terminology in the sentence itself.

10. **LOGICAL COHERENCE CHECK:** Review your complete sentence to ensure it is semantically coherent and factually plausible. Avoid nonsensical combinations such as "The meeting was cancelled so we put it off until next month" where the actions contradict each other.

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


def create_sequential_batch_stage2_grammar_prompt(job_list, stage1_outputs):
    """
    Generates distractors for GRAMMAR questions only.
    Focused exclusively on grammatical incorrectness requirements and structural constraints.
    """
    system_msg = f"""You are an expert ELT test designer specializing in grammar assessment. You will generate distractors for exactly {len(job_list)} grammar questions in a single JSON response with a "distractors" key."""
    
    user_msg = f"""
TASK: Generate 3 distractors for ALL {len(job_list)} GRAMMAR questions.

INPUT FROM STAGE 1 (Complete sentences with correct answers):
{json.dumps(stage1_outputs, indent=2)}

SENTENCE-LEVEL VALIDATION PROCEDURE:

For EACH proposed distractor, you MUST:
1. Take the Complete Sentence from Stage 1
2. Replace the Correct Answer with your proposed distractor
3. Evaluate whether the resulting sentence is grammatically correct
4. REJECT any distractor that produces a grammatically correct sentence

GRAMMAR DISTRACTOR REQUIREMENTS:

**GRAMMATICAL INCORRECTNESS IS REQUIRED:** Distractors must produce grammatically incorrect sentences when inserted into the stem. A distractor that creates a grammatically correct alternative sentence fails validation and must be replaced.

Example of validation:
- Stem: "I have lived in this city for five years."
- Correct Answer: "have lived"
- INVALID Distractor: "lived" → Creates "I lived in this city for five years." (Grammatically correct, therefore REJECTED)
- VALID Distractor: "am living" → Creates "I am living in this city for five years." (Grammatically incorrect with duration marker, therefore ACCEPTED)

**ASSESSMENT FOCUS ALIGNMENT FOR DISTINCTION QUESTIONS:** When the Assessment Focus contains "vs" (such as "going to vs will" or "Past Simple vs Present Perfect"), at least ONE distractor must be the contrasting form from the stated distinction. Additional distractors may come from proximate grammatical areas such as related tenses or modal categories.

**AUTHENTIC LEARNER ERRORS ARE ACCEPTABLE:** Distractors may include malformed constructions that represent authentic learner mistakes. Options such as "is study" or "does goes" are valid because they reflect actual errors, even though they are grammatically incomplete or incorrect structures.

**MULTI-WORD CONSTRUCTION SPLITTING:** When the correct answer is part of a multi-word phrase where elements are split between stem and answer slot (such as "It's _____ rain" testing "going to"), distractors must account for structural constraints. Options such as "will" or "might" fail because they cannot follow the contracted auxiliary in the stem.

STRUCTURAL RULES:

1. **WORD COUNT LIMIT:** Each distractor must be MAXIMUM 3 words.

2. **GRAMMATICAL PARALLELISM:** All distractors must match the word count and construction type of the correct answer. If the correct answer is two words, distractors should be two words.

3. **JUSTIFICATION REQUIRED:** For each distractor, explain the specific grammatical violation it creates when inserted into the complete sentence.

4. **PSYCHOMETRIC APPROPRIATENESS:** Distractors must represent plausible errors for learners at the specified CEFR level. Avoid obviously wrong options that no learner would select.

5. **NO LEXICAL OVERLAP:** Do not use any form of the correct answer word or its root in distractors unless testing word form distinctions.

6. **ANTI-REPETITION:** Avoid using identical distractor words across multiple questions in this batch unless required by the Assessment Focus.

MANDATORY OUTPUT FORMAT:
{{
  "distractors": [
    {{
      "Item Number": "...",
      "Distractor A": "...[max 3 words]...",
      "Why A is Wrong": "...[Explain the grammatical violation created]...",
      "Distractor B": "...[max 3 words]...",
      "Why B is Wrong": "...[Explain the grammatical violation created]...",
      "Distractor C": "...[max 3 words]...",
      "Why C is Wrong": "...[Explain the grammatical violation created]..."
    }},
    ... (exactly {len(job_list)} distractor sets)
  ]
}}

VERIFICATION CHECKLIST:
1. Have you generated exactly {len(job_list)} distractor sets?
2. For EACH distractor, have you confirmed it produces a grammatically INCORRECT sentence?
3. For distinction questions, have you included at least one distractor testing the stated contrast?
4. Have you explained the specific grammatical violation for each distractor?
"""
    return system_msg, user_msg


def create_sequential_batch_stage2_vocabulary_prompt(job_list, stage1_outputs):
    """
    Generates distractors for VOCABULARY questions only.
    Focused exclusively on semantic incompatibility while maintaining grammatical correctness.
    """
    system_msg = f"""You are an expert ELT test designer specializing in vocabulary assessment. You will generate distractors for exactly {len(job_list)} vocabulary questions in a single JSON response with a "distractors" key."""
    
    user_msg = f"""
TASK: Generate 3 distractors for ALL {len(job_list)} VOCABULARY questions.

INPUT FROM STAGE 1 (Complete sentences with correct answers):
{json.dumps(stage1_outputs, indent=2)}

SENTENCE-LEVEL VALIDATION PROCEDURE:

For EACH proposed distractor, you MUST:
1. Take the Complete Sentence from Stage 1
2. Replace the Correct Answer with your proposed distractor
3. Evaluate whether the resulting sentence is grammatically correct AND semantically coherent
4. REJECT any distractor that creates a grammatically malformed sentence
5. REJECT any distractor that creates a semantically plausible sentence

VOCABULARY DISTRACTOR REQUIREMENTS:

**GRAMMATICAL CORRECTNESS IS REQUIRED:** Distractors must produce grammatically correct sentences when inserted into the stem. A distractor that creates a grammatically malformed sentence fails validation and must be replaced.

**SEMANTIC INCOMPATIBILITY IS REQUIRED:** Distractors must be semantically eliminated by context clues while remaining grammatically valid. The complete sentence with the distractor must be grammatically acceptable but semantically implausible, contradictory, or idiomatically wrong.

Example of validation:
- Stem: "I would _____ stay home tonight than go to the party."
- Correct Answer: "rather"
- INVALID Distractor: "prefer" → Creates "I would prefer stay home..." (Grammatically incorrect, missing "to", therefore REJECTED)
- VALID Distractor: "often" → Creates "I would often stay home..." (Grammatically correct but semantically odd with "than", therefore ACCEPTED)

**LEXICAL LEVEL MATCHING:** When the correct answer is a higher-level vocabulary item for the CEFR rating, include at least ONE distractor that matches this lexical sophistication. Avoid creating distractor sets where the correct answer is obviously more complex than all alternatives.

Example:
- Target word "postpone" at A2 level is more advanced than "cancel" or "finish"
- Include distractors such as "possess" or "position" (phonetically similar, lexically matched)
- This prevents selection based on word sophistication rather than meaning comprehension

**PHONETIC SIMILARITY STRATEGY:** For higher-level vocabulary items (B1 and above), include at least ONE distractor that is phonetically similar to the correct answer but semantically unrelated. This creates genuine vocabulary discrimination tasks.

**FULL SENTENCE CONTEXT TESTING:** For verb collocation questions, evaluate distractors against the complete sentence structure, not merely against the immediate noun. A verb may collocate acceptably with the noun in isolation but fail due to other sentence elements.

Example:
- "She always _____ breakfast for her children before school."
- "cooks breakfast" is an acceptable collocation in isolation
- But "cooks breakfast for her children" is also grammatically and semantically acceptable
- Better distractors: "eats", "gives", "takes" which fail with the benefactive "for her children" construction

**CATEGORY MATCHING FOR FUNCTIONAL WORDS:** For questions testing adverbs, adjectives, or other functional categories, ensure distractors come from proximate but semantically distinct areas rather than creating near-synonyms.

Example:
- When testing intensifiers such as "incredibly", use adverbs of manner such as "quickly" or "carefully"
- Avoid other intensifiers such as "extremely" which would be semantically acceptable

STRUCTURAL RULES:

1. **WORD COUNT LIMIT:** Each distractor must be MAXIMUM 3 words.

2. **GRAMMATICAL PARALLELISM:** All distractors must match the word count and form structure of the correct answer.

3. **JUSTIFICATION REQUIRED:** For each distractor, explain the semantic incompatibility while explicitly confirming grammatical validity.

4. **PSYCHOMETRIC APPROPRIATENESS:** Distractors must represent plausible vocabulary confusions for learners at the specified CEFR level.

5. **NO LEXICAL OVERLAP:** Do not use any form of the correct answer word or its root in distractors.

6. **ANTI-REPETITION:** Avoid using identical distractor words across multiple questions in this batch.

MANDATORY OUTPUT FORMAT:
{{
  "distractors": [
    {{
      "Item Number": "...",
      "Distractor A": "...[max 3 words]...",
      "Why A is Wrong": "...[Explain semantic incompatibility and confirm grammatical validity]...",
      "Distractor B": "...[max 3 words]...",
      "Why B is Wrong": "...[Explain semantic incompatibility and confirm grammatical validity]...",
      "Distractor C": "...[max 3 words]...",
      "Why C is Wrong": "...[Explain semantic incompatibility and confirm grammatical validity]..."
    }},
    ... (exactly {len(job_list)} distractor sets)
  ]
}}

VERIFICATION CHECKLIST:
1. Have you generated exactly {len(job_list)} distractor sets?
2. For EACH distractor, have you confirmed it produces a grammatically CORRECT sentence?
3. For EACH distractor, have you confirmed it is semantically incompatible with the context?
4. For higher-level words, have you included lexically matched or phonetically similar distractors?
5. For verb collocations, have you tested against the full sentence structure?
"""
    return system_msg, user_msg


def create_sequential_batch_stage3_prompt(job_list, stage1_outputs, stage2_outputs):
    """
    Quality validation for ALL questions at once, can identify cross-question issues.
    ENHANCED: Specifically validates sentence-level grammatical correctness testing and 
    distinguishes between grammar versus vocabulary distractor requirements.
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
            "Category": s1.get("Category", ""),
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

VALIDATION PROCEDURE FOR EACH QUESTION:

Step 1: Identify question type from Category field (Grammar or Vocabulary).

Step 2: For EACH distractor, perform sentence reconstruction:
- Insert the distractor into the Question Prompt to replace the blank
- Evaluate the resulting complete sentence

Step 3: Apply type-specific validation:

FOR GRAMMAR QUESTIONS:
- Each distractor should produce a grammatically INCORRECT sentence
- Flag any distractor that creates a grammatically correct alternative as INVALID
- Verify that at least one distractor tests the stated Assessment Focus distinction (if applicable)
- Confirm distractors represent authentic learner errors for the CEFR level

FOR VOCABULARY QUESTIONS:
- Each distractor should produce a grammatically CORRECT sentence that is semantically wrong
- Flag any distractor that creates a grammatically malformed sentence as INVALID
- Flag any distractor that is both grammatically correct AND semantically plausible as INVALID
- For higher-level target words, verify lexical level matching or phonetic similarity
- For verb collocations, verify full sentence structure eliminates distractors

EVALUATION CRITERIA:

1. **SENTENCE RECONSTRUCTION TEST:** Does each distractor meet the grammatical correctness requirement for its question type?

2. **AMBIGUITY TEST:** Can a competent learner legitimately select any distractor based on semantic plausibility (after grammatical validation)?

3. **CONTEXT CLUE STRENGTH:** Does the context clue eliminate distractors appropriately (grammatically for grammar questions, semantically for vocabulary questions)?

4. **ASSESSMENT FOCUS ALIGNMENT:** Do distractors appropriately test the stated focus?

5. **LOGICAL COHERENCE:** Is the complete sentence semantically coherent and factually plausible?

6. **METALANGUAGE CHECK:** Does the prompt use grammar terminology?

7. **CROSS-QUESTION CHECK:** Are there repeated themes or excessive similarity between questions?

MANDATORY OUTPUT FORMAT:
{{
  "validations": [
    {{
      "Item Number": "...",
      "Overall Quality": "Pass" or "Requires Revision",
      "Sentence Reconstruction Results": "Pass/Fail - identify which distractors fail and why",
      "Ambiguity Issues": ["list semantically plausible distractors after grammatical validation"],
      "Context Clue Assessment": "Strong/Weak/Absent - with type-appropriate explanation",
      "Assessment Focus Alignment": "Correct/Incorrect - note specific issues",
      "Other Issues": ["list violations of criteria 5-6"],
      "Cross-Question Issues": ["note similarities to other questions"],
      "Revision Recommendations": "Specific guidance with replacement suggestions or 'None'"
    }},
    ... (exactly {len(job_list)} validation reports)
  ]
}}

VERIFICATION: Provide exactly {len(job_list)} validation reports with complete sentence reconstruction analysis.
"""
    return system_msg, user_msg


# --------------------------------------------------------------------------
# Legacy/Fallback Strategies (maintained for backward compatibility)
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
