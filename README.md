# reddit-genai-pipeline
This repo builds a GenAI-powered Q&A system using Reddit data, showcasing DE + AI integration.

### Step 1: Extract
- Input: None
- Process: Use praw library to get top 100 posts of subreddits
- Output: Create data files to location '/data/raw'

### Step 2: Transform
- Input: Multiple raw Reddit CSVs from '/data/raw'
- Process: Combines all files, cleans text (lowercasing, remove URLs/punctutation), adds features (text length, posting day)
- Output: Single processed dataset in'/data/processed/reddit_posts_cleaned.csv'

