import os

# Centralized Model & Backend Configuration
DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
MODEL_NAME = os.getenv("GROQ_MODEL", DEFAULT_MODEL_NAME)
