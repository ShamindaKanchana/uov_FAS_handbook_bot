import cohere

def get_clean_response(prompt, model="command-a-03-2025"):
    """
    Get a clean response from Cohere's API
    
    Args:
        prompt (str): The user's input prompt
        model (str): The model to use (default: "command-a-03-2025")
        
    Returns:
        str: Cleaned response text
    """
    try:
        # Initialize the Cohere client
        co = cohere.ClientV2("sdiq9Sm0NXjI1oTzFbzHG51co3SOFShhkj0vwTXn")
        
        # Get the response from the model
        response = co.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract the clean response text
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            # For newer versions of the Cohere API
            content_items = response.message.content
            if isinstance(content_items, list) and len(content_items) > 0:
                return content_items[0].text
        
        # Fallback to string representation if structure is different
        return str(response)
        
    except Exception as e:
        return f"Error getting response: {str(e)}"

def main():
    """Main function to handle user interaction."""
    print("AI Assistant (type 'exit' to quit)")
    print("=" * 30)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit condition
        if user_input.lower() in ('exit', 'quit'):
            print("Goodbye!")
            break
            
        # Get and display the response
        response = get_clean_response(user_input)
        print("\nAI:", response)

if __name__ == "__main__":
    main()
