import json
import re

def parse_response(raw_response):
    """
    Takes the raw string from the LLM and converts it into a Python dictionary.
    It aggressively cleans the string to handle Markdown code blocks.
    """
    # 1. Handle empty responses
    if not raw_response:
        return None, "Empty response from LLM."

    # 2. Handle existing error messages passed from the LLM service
    if raw_response.startswith("Error:"):
        return None, raw_response

    try:
        # 3. Clean up Markdown code blocks (```json ... ```)
        clean_text = raw_response.strip()
        
        # If the AI wrapped the JSON in backticks (common behavior), extract just the code
        if "```" in clean_text:
            # This pattern finds everything between ```json (or just ```) and the closing ```
            pattern = r"```(?:json)?\s*(.*?)```"
            match = re.search(pattern, clean_text, re.DOTALL)
            if match:
                clean_text = match.group(1).strip()
        
        # 4. Attempt to parse the clean JSON
        data = json.loads(clean_text)
        return data, None
        
    except json.JSONDecodeError:
        # If it fails, print the raw response to your logs (for debugging) and return an error
        print(f"FAILED JSON: {raw_response}") 
        return None, "Failed to parse JSON. The AI response was malformed."
    except Exception as e:
        return None, f"Unexpected error parsing output: {str(e)}"
