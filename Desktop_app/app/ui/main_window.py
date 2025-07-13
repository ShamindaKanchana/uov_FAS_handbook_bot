"""Main window for the UoV FAS Handbook Assistant."""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLineEdit, QLabel, QStatusBar, QSizePolicy, QApplication
)
from PyQt6.QtCore import Qt, QSize, QTimer, QObject, pyqtSignal, QTime
from PyQt6.QtGui import QIcon, QTextCursor, QTextDocument, QTextCharFormat, QColor

# Import QueryEngine
try:
    from src.retrieval.retriever import QueryEngine
    QUERY_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import QueryEngine: {e}")
    QUERY_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self, parent=None):
        """Initialize the main window."""
        super().__init__(parent)
        self.setWindowTitle("UoV FAS Handbook Assistant")
        self.setMinimumSize(800, 600)
        
        # Set up the main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # Set up UI components
        self._setup_chat_area()
        self._setup_input_area()
        self._setup_status_bar()
        
        # Initialize query engine
        self.query_engine = None
        self._init_query_engine()
        
        # Show welcome message
        self._show_welcome_message()
    
    def _init_query_engine(self):
        """Initialize the QueryEngine with the correct parameters."""
        if not QUERY_ENGINE_AVAILABLE:
            self._append_message(
                "System",
                "Warning: QueryEngine is not available. Using placeholder responses."
            )
            return
            
        try:
            # Get the root project directory (go up two levels from Desktop_app)
            project_root = Path(__file__).parent.parent.parent.parent
            storage_path = project_root / "qdrant_handbook"
            
            # Check if the storage directory exists
            if not storage_path.exists():
                error_msg = (
                    f"Qdrant data not found at: {storage_path}\n\n"
                    "The Qdrant database containing the handbook data is missing.\n\n"
                    "To fix this, please run the following steps from the project root directory:\n"
                    "1. Activate your virtual environment\n"
                    "2. Run: python -m src.embedding.process_handbook_embeddings\n\n"
                    "This will process the handbook data and populate the Qdrant database.\n"
                    "The desktop app requires the Qdrant database to be present in the root project directory."
                )
                logger.error(error_msg)
                self._append_message("System", error_msg)
                return
            
            logger.info(f"Initializing QueryEngine with Qdrant data from: {storage_path}")
            
            # Initialize the QueryEngine with the same collection name as the web app
            self.query_engine = QueryEngine(
                collection_name="handbook_chunks",  # Match the web app's collection name
                storage_path=str(storage_path),
                model_name="all-MiniLM-L6-v2"
            )
            
            # Test the connection and check if the collection has data
            test_results = self.query_engine.search("test", top_k=1)
            if not test_results:
                warning_msg = (
                    "The Qdrant database exists but appears to be empty.\n\n"
                    "To populate it with handbook data, please run the following from the project root:\n"
                    "1. Activate your virtual environment\n"
                    "2. Run: python -m src.embedding.process_handbook_embeddings\n\n"
                    "The application will use placeholder responses until the database is populated."
                )
                logger.warning(warning_msg)
                self._append_message("System", warning_msg)
            else:
                logger.info("Successfully connected to Qdrant database with data")
            
        except Exception as e:
            logger.error(f"Failed to initialize QueryEngine: {e}", exc_info=True)
            self._append_message(
                "System",
                f"Error initializing search engine: {str(e)}\n"
                "The application will use placeholder responses.\n"
                "To fix this, please run the data processing script from the project root:\n"
                "python -m src.embedding.process_handbook_embeddings"
            )
            self.query_engine = None
    
    def _show_welcome_message(self):
        """Show a welcome message with instructions."""
        welcome_msg = (
            "Welcome to the UoV FAS Handbook Assistant!\n\n"
            "You can ask questions about the Faculty of Applied Sciences handbook, "
            "including information about courses, programs, regulations, and more.\n\n"
            "Try asking questions like:\n"
            "• How do I calculate my GPA?\n"
            "• What are the degree requirements for Computer Science?\n"
            "• Tell me about the registration process."
        )
        self._append_message("Assistant", welcome_msg)
    
    def _setup_chat_area(self):
        """Set up the chat display area."""
        # Create chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_document = QTextDocument()
        self.chat_display.setDocument(self.chat_document)
        
        # Set default font for the chat
        font = self.chat_display.font()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.chat_display.setFont(font)
        
        # Set document margins
        self.chat_document.setDocumentMargin(10)
        
        # Store the base style for messages
        self.message_style = """
            <style>
                body {
                    font-family: Arial, sans-serif;
                    font-size: 12pt;
                    margin: 10px;
                }
                .user-message {
                    background-color: #e3f2fd;
                    border-radius: 15px;
                    padding: 8px 12px;
                    margin: 5px 0;
                    max-width: 80%;
                    float: right;
                    clear: both;
                }
                .assistant-message {
                    background-color: #f5f5f5;
                    border-radius: 15px;
                    padding: 8px 12px;
                    margin: 5px 0;
                    max-width: 80%;
                    float: left;
                    clear: both;
                }
                .message-sender {
                    font-weight: bold;
                    margin-bottom: 2px;
                }
                .message-time {
                    font-size: 0.8em;
                    color: #666;
                    text-align: right;
                    margin-top: 2px;
                }
            </style>
        """
        
        # Add initial HTML with styles
        self.chat_display.setHtml(f"""
            <!DOCTYPE html>
            <html>
                <head>{self.message_style}</head>
                <body>
                    <div style='text-align: center; color: #6c757d; margin: 20px 0;'>
                        <h3>Welcome to UoV FAS Handbook Assistant</h3>
                        <p>Ask me anything about the Faculty of Applied Sciences handbook.</p>
                    </div>
                </body>
            </html>
        """)
        
        self.main_layout.addWidget(self.chat_display, stretch=1)
    
    def _setup_input_area(self):
        """Set up the input area with text field and send button."""
        # Input area container
        input_container = QWidget()
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(10)
        
        # Message input
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask a question about the UoV FAS Handbook...")
        self.message_input.returnPressed.connect(self._on_send_clicked)
        self.message_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #ced4da;
                border-radius: 18px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #80bdff;
                outline: none;
            }
        """)
        input_layout.addWidget(self.message_input, stretch=1)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.setFixedWidth(80)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 18px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QPushButton:pressed {
                background-color: #0056b3;
            }
        """)
        self.send_button.clicked.connect(self._on_send_clicked)
        input_layout.addWidget(self.send_button)
        
        self.main_layout.addWidget(input_container)
    
    def _setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _on_send_clicked(self):
        """Handle send button click or Enter key press."""
        message = self.message_input.text().strip()
        if not message:
            return
            
        # Clear input
        self.message_input.clear()
        
        # Display user message
        self._append_message("You", message)
        
        # TODO: Process the message with the query engine
        self._process_query(message)
    
    def _append_message(self, sender: str, message: str):
        """Append a message to the chat display.
        
        Args:
            sender: The name of the message sender
            message: The message text
        """
        try:
            # Get current HTML content
            current_html = self.chat_display.toHtml()
            
            # Determine if this is a user message
            is_user = sender.lower() == 'you'
            
            # Create message HTML with proper styling
            message_class = "user-message" if is_user else "assistant-message"
            message_html = f"""
            <div class='{message_class}'>
                <div class='message-sender'>{'You' if is_user else 'Assistant'}</div>
                <div class='message-text'>{message}</div>
                <div class='message-time'>{QTime.currentTime().toString('hh:mm')}</div>
            </div>
            """
            
            # Insert the new message before the closing body tag
            if "</body>" in current_html:
                new_html = current_html.replace(
                    "</body>", 
                    f"{message_html}\n</body>"
                )
                self.chat_display.setHtml(new_html)
            else:
                # Fallback in case the HTML structure is unexpected
                self.chat_display.append(f"<b>{'You' if is_user else 'Assistant'}:</b> {message}")
            
            # Scroll to bottom
            scrollbar = self.chat_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
            # Force update the display
            QApplication.processEvents()
            
        except Exception as e:
            print(f"Error appending message: {e}")
    
    def _process_query(self, query: str):
        """Process a user query."""
        try:
            self.status_bar.showMessage("Processing query...")
            QApplication.processEvents()  # Update UI immediately
            
            # Simulate processing with a small delay
            QTimer.singleShot(100, lambda: self._generate_response(query))
            
        except Exception as e:
            self._append_message("System", f"Error processing query: {str(e)}")
            self.status_bar.showMessage("Error processing query")
    
    def _generate_response(self, query: str):
        """
        Generate and display the response to a query using the QueryEngine.
        
        Args:
            query: The user's query string
        """
        try:
            if not query.strip():
                self._append_message("Assistant", "Please enter a valid question.")
                return
                
            # Show typing indicator
            self.status_bar.showMessage("Searching for information...")
            QApplication.processEvents()  # Update UI
            
            if not self.query_engine:
                self._append_message(
                    "Assistant",
                    "I'm sorry, the search engine is not available. "
                    "Please check the error messages above for details."
                )
                return
            
            # Expand query with common variations
            expanded_queries = [
                query,
                f"{query} in UoV FAS handbook",
                f"{query} calculation method",
                f"{query} formula and regulations",
                f"how to calculate {query}"
            ]
            
            # Try different queries until we get results or run out of options
            search_results = []
            for q in expanded_queries:
                search_results = self.query_engine.search(
                    query=q,
                    top_k=5,  # Get more results to increase chances of a good match
                    score_threshold=0.2  # Lower threshold to catch more potential matches
                )
                if search_results:
                    logger.info(f"Found results for query: {q}")
                    break
            
            # If still no results, try a broader search
            if not search_results and 'gpa' in query.lower():
                search_results = self.query_engine.search(
                    query="grade point average calculation method",
                    top_k=5,
                    score_threshold=0.15  # Even lower threshold for GPA-specific search
                )
            
            if not search_results:
                # Provide a helpful response with common handbook topics
                response = (
                    "I couldn't find specific information about that in the handbook. "
                    "Here are some common topics you might find helpful:\n\n"
                    "• Academic regulations and procedures\n"
                    "• Grading system and classifications\n"
                    "• Course registration and examinations\n"
                    "• Degree requirements and program structures\n\n"
                    "You might also try rephrasing your question, for example:\n"
                    "• 'What is the grading system at UoV FAS?'\n"
                    "• 'How are grades calculated?'\n"
                    "• 'Where can I find information about academic standing?'"
                )
                self._append_message("Assistant", response)
                return
            
            # Format the response with sources
            response_parts = [
                "Here's what I found in the UoV FAS Handbook:\n"
            ]
            
            for i, result in enumerate(search_results, 1):
                # Access SearchResult attributes directly
                content = getattr(result, 'text', '').strip()
                if not content:
                    continue
                    
                # Get metadata with safe attribute access
                metadata = getattr(result, 'metadata', {})
                source = metadata.get('source', 'Handbook')
                page = metadata.get('page', 'N/A')
                
                response_parts.append(f"\n{i}. {content}\n")
                response_parts.append(f"   [Source: {source}, Page {page}]\n")
            
            # Add a helpful note
            if len(search_results) > 0:
                response_parts.append(
                    "\nNote: The information above is based on the UoV FAS Handbook. "
                    "For the most accurate and up-to-date information, please consult the official handbook "
                    "or contact the Faculty of Applied Sciences directly."
                )
            else:
                response_parts.append(
                    "\nI couldn't find specific information about this in the handbook. "
                    "You might want to rephrase your question or check the official documentation."
                )
            
            # Join all parts and send the response
            full_response = ''.join(response_parts)
            self._append_message("Assistant", full_response)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            self._append_message(
                "System",
                f"An error occurred while processing your request: {str(e)}\n"
                "Please try again or rephrase your question."
            )
        finally:
            self.status_bar.showMessage("Ready")
    
    def closeEvent(self, event):
        """Handle window close event."""
        logger.info("Application closing...")
        try:
            # Clean up resources
            if hasattr(self, 'query_engine') and self.query_engine is not None:
                logger.info("Cleaning up query engine...")
                try:
                    # Close Qdrant client connection if it exists
                    if hasattr(self.query_engine, 'client'):
                        self.query_engine.client.close()
                    # Clear model from memory if it exists
                    if hasattr(self.query_engine, 'model'):
                        if hasattr(self.query_engine.model, 'cpu'):
                            self.query_engine.model.cpu()
                        if hasattr(self.query_engine.model, 'to'):
                            import torch
                            self.query_engine.model.to('cpu')
                            if hasattr(torch, 'cuda'):
                                torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error cleaning up query engine: {e}", exc_info=True)
                finally:
                    self.query_engine = None
            
            # Save any necessary state here
            # (e.g., window size/position, chat history, etc.)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
        finally:
            # Ensure the application exits cleanly
            QApplication.processEvents()
            # Call parent's closeEvent
            super().closeEvent(event)
