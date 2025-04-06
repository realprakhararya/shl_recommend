from fastapi import FastAPI
from pydantic import BaseModel
from recommender import parse_query_with_gemini, recommend_assessments

app = FastAPI()

class QueryModel(BaseModel):
    query: str

@app.post("/recommend")
def recommend(query: QueryModel):
    filters = parse_query_with_gemini(query.query)
    results = recommend_assessments(filters)
    return {"recommendations": results}
