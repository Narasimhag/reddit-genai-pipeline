import os
from dotenv import load_dotenv
from pinecone import Pinecone
import pandas as pd
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# init pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "reddit-genai"
index = pc.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

def query_pinecone(user_query, top_k=5):
    # Embed the query using the sentence transformer model
    query_vector = model.encode(user_query).tolist()
    print(len(query_vector))
    # Query the Pinecone index to get the most similar vectors with metadata
    res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    results = []

    for match in res.matches:
        results.append({
            "id": match.id,
            "score": match.score,
            "subreddit": match.metadata.get("subreddit", ""),
            "title": match.metadata.get("title", ""),
            "selftext_clean": match.metadata.get("selftext_clean", ""),
            "created_day": match.metadata.get("created_day", ""),
            "text_length": match.metadata.get("text_length", 0),
        })

    return pd.DataFrame(results).sort_values(by="score", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    results = query_pinecone(user_query, top_k=25)
    print("Query Results:")
    print(results)