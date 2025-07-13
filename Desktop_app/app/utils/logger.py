"""Logging configuration for the application."""
import logging
from pathlib import Path
from typing import Optional

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to the log file. If None, logs to console only.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("uov_handbook")
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file is specified)
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
