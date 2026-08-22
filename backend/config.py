import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Centralized Model & Backend Configuration
DEFAULT_MODEL_NAME = "qwen/qwen3.6-27b"
MODEL_NAME = os.getenv("GROQ_MODEL", DEFAULT_MODEL_NAME)
