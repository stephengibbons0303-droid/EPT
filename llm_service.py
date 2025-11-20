from openai import OpenAI

def call_llm(messages, api_key, model="gpt-4-turbo-preview"):
    """
    Sends a message history to the OpenAI API using the provided API key.
    """
    if not api_key:
        return "Error: API Key is missing. Please enter it in the sidebar."

    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": messages[0]},
                {"role": "user", "content": messages[1]}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"
