import os
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import tqdm

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# init pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "reddit-genai"
# Create the index if it doesn't exist
# if index_name not in pc.list_indexes():
#     pc.create_index(index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(index_name)

df = pd.read_parquet("data/vectorized/reddit_posts_embeddings.parquet")

#Clean nulls in the data
def sanitize_metadata(row):
    metadata = {
        "subreddit": str(row.get("subreddit", "")),
        "title": str(row.get("title", "")),
        "selftext_clean": str(row.get("selftext_clean", "")),
        "created_day": str(row.get("created_day", "")),
        "score": float(row.get("score", 0)),
        "text_length": int(row.get("text_length", 0))
    }
    # ensure no "nan" or "None" strings
    return {k: ("" if v.lower() in ["nan", "none"] else v) if isinstance(v, str) else v for k, v in metadata.items()}


# Upload in batches
batch_size = 100
emb_cols = [c for c in df.columns if c.startswith("emb_")]
for start in tqdm.tqdm(range(0, len(df), batch_size)):
    end = start + batch_size
    batch = df.iloc[start:end]

    vectors = []
    for i, row in batch.iterrows():
        vector = row[emb_cols].tolist()
        metadata = sanitize_metadata(row)
        vectors.append((str(i), vector, metadata))
    
    if vectors:
        index.upsert(vectors=vectors)

print(f"✅ Finished uploading vectors to Pinecone index '{index_name}'")