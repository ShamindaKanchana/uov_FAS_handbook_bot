from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os

# Import our routes
from routes import bp as chat_bp

def create_app():
    app = Flask(__name__)
    
    # Enable CORS for development
    CORS(app)
    
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
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
