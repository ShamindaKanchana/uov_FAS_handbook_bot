from typing import List, Dict, Any
from dataclasses import dataclass
from src.retrieval.retriever import SearchResult
from .nlp import get_clean_response

@dataclass
class GenerationConfig:
    model_name: str = "command-a-03-2025"
    temperature: float = 0.7
    max_tokens: int = 500

class ResponseGenerator:
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
    
    def generate_response(
        self,
        query: str,
        search_results: List[SearchResult],
        **generation_kwargs
    ) -> str:
        """
        Generate a response using the retrieved context.
        
        Args:
            query: User's original query
            search_results: List of retrieved SearchResult objects
            **generation_kwargs: Additional generation parameters
        """
        try:
            # Format the prompt with context
            prompt = self._format_prompt(query, search_results)
            
            # Get response using the get_clean_response function
            return get_clean_response(
                prompt=prompt,
                model=self.config.model_name
            )
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _format_prompt(
        self,
        query: str,
        search_results: List[SearchResult]
    ) -> str:
        """Format the prompt with query and retrieved context."""
        # Format context with source information
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source = result.metadata.get('source', 'Unknown source')
            page = result.metadata.get('page', 'N/A')
            context_parts.append(
                f"Source {i} (Relevance: {result.score*100:.1f}%):\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content: {result.text}\n"
            )
        
        context = "\n\n".join(context_parts)
        
        return f"""You are a helpful assistant for the University of Vavuniya's Faculty of Applied Sciences.
            
            
            
            Your task is to answer questions based on the provided context from the university handbook.

            Question: {query}

            Context:
            {context}

            Instructions:
            1. Answer the question based on the provided context.
            2. Be concise and accurate.
            3. If the context doesn't contain enough information, say "I couldn't find enough information to answer that question."
            4. Include relevant details from the context to support your answer.
            5. If the question is about specific requirements or procedures, be sure to mention any important conditions or steps.
            6. Format your response in clear, easy-to-read paragraphs.

            Answer:
"""

def main():
    """Standalone test function for the generator."""
    print("RAG Generator Test (type 'exit' to quit)")
    print("=" * 50)
    
    # Initialize the generator
    generator = ResponseGenerator()
    
    while True:
        try:
            # Get user input
            user_input = input("\nYour question: ").strip()
            
            if user_input.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
                
            # Mock search results for testing
            mock_results = [
                SearchResult(
                    id="test1",
                    score=0.95,
                    text="The Environmental Science program requires 120 credits to graduate.",
                    metadata={"source": "Program Handbook", "page": 12}
                )
            ]
            
            # Generate and display response
            print("\n🤖 Generating response...")
            response = generator.generate_response(user_input, mock_results)
            print(f"\n💬 {response}\n")
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()