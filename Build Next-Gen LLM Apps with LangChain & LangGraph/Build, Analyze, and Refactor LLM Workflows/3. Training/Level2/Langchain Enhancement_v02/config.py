# config.py

import logging

# Model configuration
MODEL_NAME   = "phi3:mini"
TEMPERATURE  = 0
SEED         = 42
MAX_ATTEMPTS = 3

# Logger setup
def setup_logger() -> logging.Logger:   #The -> is a return type annotation — it's a hint indicating what type the function is expected to return.
    """Configure logger with console and file handlers."""
    logger = logging.getLogger("langchain_app")
    '''
    getLogger("langchain_app") either creates a new logger with that name 
     or returns the existing one if it's already been created. 
    Logger names form a hierarchy — "langchain_app.module1" is a child of "langchain_app". 
    We use one name for the whole app so all modules share the same logger 
     configuration.
    '''

    logger.setLevel(logging.DEBUG)  # Capture everything at logger level
    '''
    Sets the logger's minimum severity level. Levels from low to high: 
    DEBUG < INFO < WARNING < ERROR < CRITICAL'''
    
    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    '''
    Creates a formatter that defines how each log line looks.
    Format string breakdown:

    %(asctime)s — timestamp (controlled by datefmt)
    %(levelname)-8s — severity level, left-aligned in 8 characters ("INFO    " not "INFO")
    %(module)s — which Python file logged this (e.g., chains, main)
    %(message)s — the actual log message

    Example output: 2026-02-24 14:23:01 | INFO     | chains   | Model initialized
    '''
    
    # Console handler - INFO and above only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    '''
    StreamHandler writes to sys.stderr (the console/terminal).
    Sets level to INFO, meaning DEBUG messages won't appear on screen 
     (they're still captured by the logger, but this handler ignores them). 
    Attaches the formatter so console output uses our format.
    '''
    
    # File handler - DEBUG and above (captures everything)
    file_handler = logging.FileHandler("run.log", mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    '''
    FileHandler writes to run.log file.
    mode="a" means append — each run adds to the file rather than overwriting. 
    encoding="utf-8" prevents issues with non-ASCII characters (especially 
     important on Windows). This handler captures DEBUG and above, so the 
     file has more detail than the console.
    '''
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    '''
    Attaches both handlers to the logger. Now when you do 
     logger.info("message"), it goes through the logger (level: DEBUG), 
     gets sent to both handlers, and each handler decides whether to emit 
     it based on its own level.
    '''

    return logger

logger = setup_logger()
'''
Back in config.py at module level: Calls the function and stores the logger 
in a module-level variable. This executes immediately when config.py is 
imported. The logger is now configured and ready.
'''