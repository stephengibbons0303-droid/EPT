import json
import re

def parse_response(raw_response):
    """
    Takes the raw string from the LLM and converts it into a Python dictionary.
    It aggressively cleans the string to handle Markdown code blocks.
    """
    if not raw_response:
        return None, "Empty response from LLM."

    if raw_response.startswith("Error:"):
        return None, raw_response

    try:
        clean_text = raw_response.strip()
        
        if "```" in clean_text:
            pattern = r"```(?:json)?\s*(.*?)```"
            match = re.search(pattern, clean_text, re.DOTALL)
            if match:
                clean_text = match.group(1).strip()
        
        data = json.loads(clean_text)
        return data, None
        
    except json.JSONDecodeError:
        print(f"FAILED JSON: {raw_response}") 
        return None, "Failed to parse JSON. The AI response was malformed."
    except Exception as e:
        return None, f"Unexpected error parsing output: {str(e)}"
