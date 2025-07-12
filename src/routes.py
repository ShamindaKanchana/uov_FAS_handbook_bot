from flask import Blueprint, request, jsonify
from retrieval.retriever import QueryEngine, SearchResult
from generation.generator import ResponseGenerator, GenerationConfig
from generation.nlp import get_clean_response
import logging
import os
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the RAG components
query_engine = QueryEngine()

def format_prompt(query: str, results: List[SearchResult]) -> str:
    """Format the prompt with query and retrieved context."""
    context_parts = []
    for i, result in enumerate(results, 1):
        source = result.metadata.get('source', 'Unknown source')
        page = result.metadata.get('page', 'N/A')
        context_parts.append(f"Source {i} (Page {page}, {source}):\n{result.text}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful assistant for the University of Vavuniya Faculty of Applied Science Handbook.
    
Context information from the handbook:
{context}

Question: {query}

Please provide a clear and concise answer based on the context above. If the answer isn't in the context, say you don't know.

Answer:"""
    
    return prompt

# Create a Blueprint
bp = Blueprint('chat', __name__)

def format_chat_response(response_text: str, results: List[SearchResult] = None) -> Dict[str, Any]:
    """Format the response for the chat interface."""
    response = {
        'answer': response_text,
        'sources': []
    }
    
    if results:
        for result in results:
            source = {
                'content': result.text[:200] + '...' if len(result.text) > 200 else result.text,
                'metadata': {
                    'source': result.metadata.get('source', 'Unknown'),
                    'page': result.metadata.get('page', 'N/A'),
                    'section': result.metadata.get('section', ''),
                    'title': result.metadata.get('title', '')
                },
                'relevance': round(result.score * 100, 1) if hasattr(result, 'score') else 0
            }
            response['sources'].append(source)
    
    return response

@bp.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        query = data.get('question', '').strip()
        
        if not query:
            return jsonify({'error': 'No question provided'}), 400
        
        # Log the query
        logger.info(f"Received query: {query}")
        
        # Get search results
        results = query_engine.search(query, top_k=3)
        
        if not results or not any(results):
            return jsonify(format_chat_response(
                "I couldn't find any relevant information in the handbook. Could you try rephrasing your question?"
            ))
        
        # Format the prompt with the query and results
        prompt = format_prompt(query, results)
        
        # Generate response using the nlp module
        response_text = get_clean_response(prompt)
        
        # If there was an error with the API call
        if response_text.startswith("Error"):
            logger.error(f"Error generating response: {response_text}")
            return jsonify({
                'error': 'Error generating response',
                'details': response_text
            }), 500
        
        # Format and return the response
        return jsonify(format_chat_response(response_text, results))
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500
