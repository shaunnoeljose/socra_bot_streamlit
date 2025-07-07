# main.py (or your primary Streamlit file)
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now, access your API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Make sure it's actually loaded (optional, for debugging)
if google_api_key:
    print("Google API Key loaded successfully!")
else:
    print("Error: GOOGLE_API_KEY not found in .env file or environment.")

# Now, when you initialize your LangChain Google Generative AI model,
# the `ChatGoogleGenerativeAI` class will typically automatically pick up
# the `GOOGLE_API_KEY` from the environment if it's set.
# You usually don't need to pass it explicitly in the constructor
# if it's in os.environ.
from langchain_google_genai import ChatGoogleGenerativeAI

# Your existing code for socratic_bot_logic.py:
# If you don't explicitly pass the API key,
# ChatGoogleGenerativeAI will look for GOOGLE_API_KEY in the environment.
# Make sure this part of your code is executed *after* load_dotenv()
# in your Streamlit app.
# Example usage (assuming socratic_bot_logic.py is imported AFTER os.environ is populated):

# Assuming socratic_bot_logic.py imports this, it will implicitly pick it up.
# If you were to pass it explicitly (not usually needed if env var is set):
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=google_api_key)
# But simply having it in os.environ is typically enough.

# ... your existing Streamlit code
# For example, if socratic_bot_logic.py is imported here:
from socratic_bot_logic import socratic_graph, SocraticAgentState

# The rest of your Streamlit application logic