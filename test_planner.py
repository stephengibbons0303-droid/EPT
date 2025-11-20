import random

def create_job_list(
    total_questions, 
    q_type, 
    cefr_target, 
    selected_focus_list, 
    context_topic,
    generation_strategy
):
    """
    Generates a list of job dictionaries with unique Topic and Style for each job 
    to prevent content repetition (overfitting to few-shot examples).
    """
    job_list = []
    
    micro_contexts = [
        "a simple fact", "a polite suggestion", "a common phrase", 
        "a cause and effect statement", "a short dialogue line", 
        "a general observation", "a brief instruction", "a personal opinion"
    ]

    random_domains = [
        "Health & Fitness", "Technology & Computers", "Cooking & Food", 
        "Money & Shopping", "Daily Routine", "Art & Music", 
        "Weather & Nature", "Work & Jobs", "Education & Learning", 
        "Transport & Cities", "Family & Relationships", "Current Events"
    ]
    
    user_provided_topic = True
    if not context_topic or context_topic.strip() == "":
        user_provided_topic = False
    
    for i in range(total_questions):
        current_focus = random.choice(selected_focus_list)
        job_id = f"{q_type[0].upper()}{cefr_target}-{i+1}"
        
        micro_slant = random.choice(micro_contexts)
        
        if user_provided_topic:
            main_topic = context_topic
        else:
            current_domain = random_domains[i % len(random_domains)]
            main_topic = current_domain
            
        full_context = f"{main_topic} (Style: {micro_slant})"
        
        job = {
            "job_id": job_id,
            "type": q_type,
            "cefr": cefr_target,
            "focus": current_focus,
            "context": full_context,
            "strategy": generation_strategy
        }
        
        job_list.append(job)
        
    return job_list
