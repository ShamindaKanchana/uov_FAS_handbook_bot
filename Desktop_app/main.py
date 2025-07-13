#!/usr/bin/env python3
"""
Main entry point for the UoV FAS Handbook Desktop Application.
"""
import sys
import os
import signal
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from app.ui.main_window import MainWindow
from app.utils.logger import setup_logging

class Application(QApplication):
    """Custom application class for better signal handling."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.handle_sigint)
        self._main_window = None
        
        # Create a timer to periodically process Python signals
        self._timer = QTimer()
        self._timer.timeout.connect(lambda: None)
        self._timer.start(200)  # Process signals every 200ms
    
    def set_main_window(self, window):
        """Set the main window instance."""
        self._main_window = window
    
    def handle_sigint(self, signum, frame):
        """Handle SIGINT (Ctrl+C) gracefully."""
        if self._main_window:
            self._main_window.close()
        self.quit()

def main():
    """Initialize and start the application."""
    # Set up logging
    log_file = Path("app.log")
    logger = setup_logging(log_file)
    
    try:
        # Create application instance
        app = Application(sys.argv)
        app.setApplicationName("UoV FAS Handbook Assistant")
        app.setApplicationDisplayName("UoV FAS Handbook Assistant")
        
        # Enable high DPI scaling with compatibility check
        try:
            # PyQt6.2+ style
            from PyQt6.QtCore import QGuiApplication
            QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
        except (ImportError, AttributeError):
            # Fallback for older PyQt6 versions
            try:
                if hasattr(Qt, 'AA_EnableHighDpiScaling'):
                    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
                if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
                    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
            except Exception:
                # If DPI scaling is not available, continue without it
                pass
        
        # Create and show main window
        main_window = MainWindow()
        app.set_main_window(main_window)
        main_window.show()
        
        # Start the event loop
        exit_code = app.exec()
        
        # Ensure the application exits cleanly
        sys.exit(exit_code)
        
    except Exception as e:
        logger.critical(f"Application error: {e}", exc_info=True)
        QMessageBox.critical(
            None,
            "Application Error",
            f"A critical error occurred:\n{str(e)}\n\nCheck app.log for details."
        )
        sys.exit(1)
    finally:
        # Ensure all resources are cleaned up
        if 'app' in locals():
            app.closeAllWindows()

if __name__ == "__main__":
    main()
