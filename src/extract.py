# import modules
import os
from dotenv import load_dotenv
import praw
import pandas as pd

# Load environment variables
load_dotenv()
# Set up the environment variables for Reddit API credentials
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
# Define the function to extract Reddit data
def extract_reddit_data(subreddit_name, num_posts=100):
    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent='my_reddit_data_extractor/0.1 by u/narryRG'
    )
    # Fetch subreddit
    subreddit = reddit.subreddit(subreddit_name)
    # Extract posts
    posts = []
    for submission in subreddit.new(limit=num_posts):
        post_data = {
            'title': submission.title,
            'score': submission.score,
            'id': submission.id,
            'url': submission.url,
            'num_comments': submission.num_comments,
            'created_utc': submission.created_utc,
            'author': str(submission.author) if submission.author else 'deleted',
            'selftext': submission.selftext,
            'comments': [comment.body for comment in submission.comments.list() if hasattr(comment, 'body')]
        }
        posts.append(post_data)
    # Create DataFrame
    df = pd.DataFrame(posts)
    # Save to CSV
    df.to_csv(f'data/raw/{subreddit_name}_posts.csv', index=False)
    print(f"Extracted {len(posts)} posts from r/{subreddit_name} and saved to {subreddit_name}_posts.csv")  

# Call the function with a specific subreddit
if __name__ == "__main__":
    extract_reddit_data('datascience', num_posts=100)  