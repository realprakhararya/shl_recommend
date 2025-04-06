import os
from dotenv import load_dotenv
import google.generativeai as genai
from langsmith import Client

# Load environment variables from .env file
load_dotenv()

# LangSmith Configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "shl-recommender"

# Ensure API key is set
if not os.environ.get("LANGCHAIN_API_KEY"):
    print("⚠️ Warning: LANGCHAIN_API_KEY not set in environment or .env file")
    print("   Tracing will not work without this key.")

# Initialize LangSmith client
try:
    langsmith_client = Client()
except Exception as e:
    print(f"⚠️ Warning: Could not initialize LangSmith client: {e}")
    langsmith_client = None

def get_model():
    """Get the Gemini model with tracing enabled"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model

SCORING_WEIGHTS = {
    "technical_score": 10,
    "title_relevance": 4,
    "inferred_score": 2,
    "soft_skill_score": 2
}

def log_trace(run_type, inputs, outputs=None, error=None, metadata=None):
    """Helper function to log a trace to LangSmith"""
    if not langsmith_client:
        return None
    
    try:
        # Create a run
        run = langsmith_client.create_run(
            project_name="shl-recommender",
            name=run_type,
            inputs=inputs,
            outputs=outputs if outputs else {},
            error=str(error) if error else None,
            metadata=metadata if metadata else {}
        )
        return run.id
    except Exception as e:
        print(f"⚠️ Warning: Failed to log trace: {e}")
        return None