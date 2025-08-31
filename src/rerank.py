import pandas as pd
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query:str, retrieved_df: pd.DataFrame, top_k: int = 5):
        """
        Reranks retrived documents using cross-encoder
        """
        pairs = [(query, text) for text in retrieved_df["selftext_clean"].fillna("")]
        scores = self.model.predict(pairs)

        retrieved_df = retrieved_df.copy()
        retrieved_df["rerank_score"] = scores
        reranked = retrieved_df.sort_values(by="rerank_score", ascending=False).head(top_k)
        return reranked
    
if __name__ == "__main__":
    # After retrieve.py is run and results are stored in parquet
    df = pd.read_parquet("data/retrieved/query_results.parquet")
    reranker = Reranker()
    query = "How to learn GenAI?"
    reranked_df = reranker.rerank(query, df)
    print(reranked_df)
