# SHL Assessment Recommender System (Free & FOSS)

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

# Load the SHL data
shl_df = pd.read_csv("shl_clean.csv", encoding='latin1')

# Combine title and description for embedding
shl_df['combined_text'] = shl_df['Topic'] + ". " + shl_df['Description']


# Load Sentence Transformer (MiniLM is small and free)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for SHL entries
shl_embeddings = model.encode(shl_df['combined_text'].tolist(), convert_to_tensor=True)

# Load title -> URL map
url_map = {}
with open("final.txt", 'r', encoding='utf-8') as f:
    for line in f:
        if "=>" in line:
            title, url = line.strip().split("=>")
            url_map[title.strip()] = url.strip()

# API with FastAPI
app = FastAPI()

class QueryInput(BaseModel):
    query: str
    top_k: int = 5

@app.post("/recommend")
async def recommend(query_data: QueryInput):
    query = query_data.query
    top_k = query_data.top_k

    # Embed user query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = util.cos_sim(query_embedding, shl_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    # Format response
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        row = shl_df.iloc[idx.item()]
        results.append({
            "title": row['title'],
            "url": url_map.get(row['title'], "N/A"),
            "remote_testing": row['remote'],
            "adaptive_support": row['adaptive irt'],
            "duration": row['duration'],
            "test_type": row['test type'],
            "score": round(score.item(), 4)
        })

    return {"recommendations": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
