import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to Python path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Import our routes
from routes import bp as chat_bp

def create_app():
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static'
    )
    
    # Enable CORS for development
    CORS(app)
    
    # Initialize Qdrant client when the app starts
    try:
        from src.retrieval.retriever import QueryEngine
        from src.embedding.qdrant_singleton import QdrantClientSingleton
        
        # Get the singleton instance but don't store the client directly
        storage_path = str(Path(__file__).parent / "qdrant_handbook")
        qdrant_singleton = QdrantClientSingleton(storage_path)
        
        # Initialize the query engine with the singleton
        app.config['query_engine'] = QueryEngine(storage_path=storage_path)
        logger.info("Successfully initialized QueryEngine with Qdrant client")
        
        # Store the singleton in app config for cleanup
        app.config['qdrant_singleton'] = qdrant_singleton
        
    except Exception as e:
        logger.error(f"Failed to initialize QueryEngine: {e}")
        # Create a mock query engine in case of failure
        class MockQueryEngine:
            def search(self, *args, **kwargs):
                return []
        app.config['query_engine'] = MockQueryEngine()
    
    # Register blueprints
    app.register_blueprint(chat_bp, url_prefix='/api')
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal server error',
            'details': str(error) if app.debug else None
        }), 500
    
    return app

def run_app():
    app = create_app()
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.critical(f"Application failed to start: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    run_app()
