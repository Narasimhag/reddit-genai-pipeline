import os
from dotenv import load_dotenv
from pinecone import Pinecone
import pandas as pd
from sentence_transformers import SentenceTransformer

load_dotenv()

class Retriever:
    def __init__(self, index_name="reddit-genai", model_name="all-MiniLM-L6-v2"):
        self.index_name = index_name
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(self.index_name)
        self.encoder = SentenceTransformer(model_name)

    def search(self, query, top_k=10):
        query_vector = self.encoder.encode(query).tolist()
        res = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
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
    retriever = Retriever()
    results = retriever.search(user_query, top_k=25)
    print("Query Results:")
    print(results)
    results.to_parquet("data/retrieved/query_results.parquet", index=False)