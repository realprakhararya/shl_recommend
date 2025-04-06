from fastapi import FastAPI, Request
from pydantic import BaseModel
from recommender import parse_query_with_gemini, recommend_assessments
from langsmith import Client
from uuid import uuid4
import logging
import uvicorn

# Setup logging for tracing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommender-api")

app = FastAPI(title="Assessment Recommender API")

class QueryModel(BaseModel):
    query: str

client = Client()

@app.post("/recommend")
async def recommend(query: QueryModel, request: Request):
    """
    Recommend assessments based on natural language query
    """
    run_id = str(uuid4())
    logger.info(f"🔍 Received query: {query.query} (run_id: {run_id})")
    
    with client.trace(
        project_name="shl-recommender",
        run_id=run_id,
        name="recommend_api_call"
    ) as run:
        filters = parse_query_with_gemini(query.query)
        logger.info(f"🧠 Parsed filters: {filters}")
        
        results = recommend_assessments(filters)
        logger.info(f"✅ Top recommendations: {[r['title'] for r in results]}")
        
        # Add metadata
        run.end(outputs={"recommendations_count": len(results)})
    
    return {
        "recommendations": results,
        "run_id": run_id  # Include the run ID in response for debugging
    }

@app.get("/")
async def root():
    """
    API root endpoint with basic information
    """
    return {
        "name": "Assessment Recommender API",
        "description": "API for recommending assessments based on natural language queries",
        "endpoints": {
            "/recommend": "POST endpoint for getting assessment recommendations"
        }
    }

if __name__ == "__main__":
    logger.info("Starting web server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    logger.info("Web server stopped.")