import sys
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from flask import Blueprint, request, jsonify, current_app

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
SearchResult = Dict[str, Any]  # Define or import the actual SearchResult type

# Create a Blueprint
bp = Blueprint('chat', __name__)

def get_query_engine():
    """
    Get the query engine from the app config.
    
    Returns:
        QueryEngine: The query engine instance or None if not found
    """
    try:
        return current_app.config.get('query_engine')
    except Exception as e:
        logger.error(f"Error getting query engine: {e}")
        return None

def format_prompt(query: str, results: List[SearchResult]) -> str:
    """
    Format the prompt with query and retrieved context.
    
    Args:
        query: The user's query
        results: List of search results (SearchResult objects)
        
    Returns:
        str: Formatted prompt string
    """
    if not results:
        return f"""You are a helpful assistant for the University of Vavuniya Faculty of Applied Science Handbook.
        
No relevant information was found in the handbook for this query.

Question: {query}

Answer: I couldn't find any relevant information in the handbook to answer your question."""

    context_parts = []
    for i, result in enumerate(results, 1):
        # Access attributes directly from SearchResult object
        source = result.metadata.get('source', 'Unknown source')
        page = result.metadata.get('page', 'N/A')
        text = result.text
        context_parts.append(f"Source {i} (Page {page}, {source}):\n{text}")
    
    context = "\n\n".join(context_parts)
    
    return f"""You are a helpful assistant for the University of Vavuniya Faculty of Applied Science Handbook.
    
Context information from the handbook:
{context}

Question: {query}

Please provide a clear and concise answer based on the context above. If the answer isn't in the context, say you don't know.

Answer:"""

def format_chat_response(response_text: str, results: List[SearchResult] = None) -> Dict[str, Any]:
    """
    Format the response for the chat interface.
    
    Args:
        response_text: The generated response text
        results: List of SearchResult objects
        
    Returns:
        Dict containing the response and sources
    """
    if not results:
        results = []
        
    response = {
        'answer': response_text,
        'sources': []
    }
    
    for result in results:
        try:
            # Access attributes directly from SearchResult object
            text = result.text
            source = {
                'content': (text[:197] + '...') if len(text) > 200 else text,
                'metadata': {
                    'source': result.metadata.get('source', 'Unknown'),
                    'page': result.metadata.get('page', 'N/A'),
                    'section': result.metadata.get('section', ''),
                    'title': result.metadata.get('title', '')
                },
                'relevance': round(float(result.score) * 100, 1) if hasattr(result, 'score') else 0
            }
            response['sources'].append(source)
        except Exception as e:
            logger.error(f"Error formatting search result: {e}", exc_info=True)
            continue
    
    return response

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        query_engine = get_query_engine()
        if not query_engine:
            return jsonify({'status': 'error', 'message': 'Query engine not initialized'}), 500
            
        # Test a simple search
        test_query = "test"
        try:
            results = query_engine.search(test_query, top_k=1)
            return jsonify({
                'status': 'ok',
                'database': 'connected',
                'results_found': len(results) > 0 if results else 0
            })
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f'Search test failed: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Health check failed: {str(e)}'
        }), 500

@bp.route('/ask', methods=['POST'])
def ask():
    """
    Handle chat requests and return responses based on the handbook content.
    
    Expected JSON payload:
    {
        "question": "Your question here"
    }
    """
    try:
        # Get and validate query engine
        query_engine = get_query_engine()
        if not query_engine:
            logger.error("Query engine not initialized")
            return jsonify({
                'error': 'Service temporarily unavailable. Please try again later.'
            }), 503
        
        # Parse and validate request
        data = request.get_json() or {}
        query = data.get('question', '').strip()
        
        if not query:
            return jsonify({
                'error': 'No question provided. Please ask a question about the handbook.'
            }), 400
        
        logger.info(f"Processing query: {query}")
        
        try:
            # Get search results with error handling for Qdrant client
            try:
                results = query_engine.search(query, top_k=3)
                logger.info(f"Found {len(results) if results else 0} relevant results")
                
                if not results:
                    return jsonify(format_chat_response(
                        "I couldn't find any relevant information in the handbook to answer your question. "
                        "Please try rephrasing your question or ask about a different topic.",
                        results=[]
                    ))
                
                # Format the prompt with query and context
                prompt = format_prompt(query, results)
                
                # Generate response using the nlp module
                from src.generation.nlp import get_clean_response
                generated_text = get_clean_response(prompt)
                
                # Format the response for the chat interface
                return jsonify(format_chat_response(generated_text, results))
                
            except RuntimeError as re:
                if "QdrantLocal instance is closed" in str(re):
                    logger.error("Qdrant client is closed. Attempting to reconnect...")
                    # Try to reinitialize the query engine
                    try:
                        from src.retrieval.retriever import QueryEngine
                        from src.embedding.qdrant_singleton import QdrantClientSingleton
                        
                        storage_path = str(Path(__file__).parent / "qdrant_handbook")
                        qdrant_singleton = QdrantClientSingleton(storage_path)
                        query_engine = QueryEngine(storage_path=storage_path)
                        
                        # Update the app config with the new query engine
                        from flask import current_app
                        current_app.config['query_engine'] = query_engine
                        current_app.config['qdrant_singleton'] = qdrant_singleton
                        
                        # Retry the search
                        results = query_engine.search(query, top_k=3)
                        if results:
                            prompt = format_prompt(query, results)
                            from src.generation.nlp import get_clean_response
                            generated_text = get_clean_response(prompt)
                            return jsonify(format_chat_response(generated_text, results))
                        
                    except Exception as retry_error:
                        logger.error(f"Failed to reconnect to Qdrant: {str(retry_error)}", exc_info=True)
                
                # If we get here, either reconnection failed or it wasn't a connection error
                logger.error(f"Search error: {str(re)}", exc_info=True)
                return jsonify({
                    'error': 'There was an issue accessing the knowledge base. Please try again.'
                }), 503
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return jsonify({
                'error': 'An error occurred while searching the handbook. Please try again.'
            }), 500
            
    except Exception as e:
        logger.critical(f"Unexpected error in /ask endpoint: {e}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred. Our team has been notified.',
            'details': str(e)
        }), 500
