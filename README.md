## Project Roadmap
This project aims to build an end-to-end GenAI powered pipeline using Reddit data.
- [x] **Extract**: Collect raw Reddit data from the API.
- [x] **Clean**: Preprocess text (normalize, remove noise, structure metadata).
- [x] **Vectorize**: Generate TF-IDF features and sentence embeddings for text.
- [x] **Store & Index**: Load vectors + metadata into a vector database.
- [ ] **Retrieve**: Implement semantic search across Reddit posts using embeddings.
- [ ] **Generate**: Build a simple RAG demo using an open-source LLM.
- [ ] **Deploy**: Wrap the pipeline in a streamlist service.
- [ ] **Optimize & Scale**: Experiment with batching, efficient storage and larger models.

## Step 1: Extract
- Input: None
- Process: Use praw library to get top 100 posts of subreddits
- Output: Create data files to location '/data/raw'

## Step 2: Clean
- Input: Multiple raw Reddit CSVs from '/data/raw'
- Process: Combines all files, cleans text (lowercasing, remove URLs/punctutation), adds features (text length, posting day)
- Output: Single processed dataset in'/data/processed/reddit_posts_cleaned.csv'

## Step 3: Vectorize
- Implemented 'vectorize.py' to transform cleaned reddit posts into numerica representations:
    - **TF-IDF vectors** for sparse, interpretable features.
    - **Sentence embeddings** for dense, semantic features.
- Both outputs are saved as Parquet files
- Metadata columns are retained to allow future analysis and joinin with vectorized features.

## Step 4: Index
- Added `index.py` to store embeddings + metadata in pinecone.
- Created `reddit-genai` index (cosine similarity, 384 dim).
- Uploaded vectors and verified in Pinecone dashboard.

## Step 5: Retrieve
- Once the embeddings are indexed in Pinecone, run queries against them
- `retrive.py` script, converts the query results into a structured dataframe

