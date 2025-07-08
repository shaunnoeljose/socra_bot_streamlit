import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Google API Key for Gemini
# It's crucial to set this as an environment variable (e.g., in a .env file)
# GOOGLE_API_KEY="YOUR_API_KEY_HERE"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in a .env file or your system environment.")