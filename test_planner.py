import random

def create_job_list(
    total_questions, 
    q_type, 
    cefr_target, 
    selected_focus_list, 
    context_topic,
    generation_strategy  # Received from UI
):
    """
    Generates a list of job dictionaries.
    """
    job_list = []
    
    for i in range(total_questions):
        current_focus = random.choice(selected_focus_list)
        # Unique ID for tracking
        job_id = f"{q_type[0].upper()}{cefr_target}-{i+1}" 
        
        job = {
            "job_id": job_id,
            "type": q_type,
            "cefr": cefr_target,
            "focus": current_focus,
            "context": context_topic,
            "strategy": generation_strategy  # Stored as 'strategy' for the prompt engineer
        }
        
        job_list.append(job)
        
    return job_list
