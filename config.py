import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

SCORING_WEIGHTS = {
    "technical_score": 10,
    "title_relevance": 4,
    "inferred_score": 2,
    "soft_skill_score": 2
}

def get_model():
    """Get the Gemini model with proper API key configuration"""
    import google.generativeai as genai
    
    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")