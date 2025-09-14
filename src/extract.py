# import modules
import logging
import os
from dotenv import load_dotenv
import praw
from prawcore.exceptions import TooManyRequests
import pandas as pd
import time

from requests import RequestException

# Load environment variables
load_dotenv()
# Set up the environment variables for Reddit API credentials
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')

# Initialize Reddit instance
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent='my_reddit_data_extractor/0.1 by u/narryRG'
)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :%(message)s")

#Config
POSTS_PER_SUBREDDIT = 150  # Number of posts to extract per subreddit
MAX_RETRIES = 3  # Maximum number of retries for API calls
SLEEP_BETWEEN_SUBS = 30  # Sleep time between subreddit extractions (in seconds)
SLEEP_INITIAL = 2  # Initial sleep time before starting the extraction (in seconds)

# Define the function to extract Reddit data

def extract_reddit_data(subreddit_name, num_posts=100):
    # Extract posts
    posts = []
    # Fetch subreddit
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            subreddit = reddit.subreddit(subreddit_name)
            logging.info(f"Fetching {num_posts} posts from r/{subreddit_name} (Attempt {attempt})")
            for submission in subreddit.new(limit=num_posts):
                submission.comments.replace_more(limit=0)
                post_data = {
                    'subreddit': subreddit_name,
                    'title': submission.title,
                    'score': submission.score,
                    'id': submission.id,
                    'url': submission.url,
                    'num_comments': submission.num_comments,
                    'created_utc': submission.created_utc,
                    'author': str(submission.author) if submission.author else 'deleted',
                    'selftext': submission.selftext,
                    'comments': [comment.body for comment in submission.comments.list()[:5] if hasattr(comment, 'body')]
                }
                posts.append(post_data)
             # Create DataFrame
            df = pd.DataFrame(posts)
            # Save to CSV
            df.to_csv(f'data/raw/{subreddit_name}_posts.csv', index=False)
            logging.info(f"Extracted {len(posts)} posts from r/{subreddit_name} and saved to {subreddit_name}_posts.csv")
            return
        except TooManyRequests as e:
            retry_after = getattr(e, 'retry_after', None)
            wait = retry_after if retry_after else SLEEP_INITIAL * (2 ** (attempt - 1))
            time.sleep(wait)
        except RequestException as e:
            logging.warning(f"RequestException for r/{subreddit_name}: {e}")
            time.sleep(5 * attempt)
        except Exception as e:
            logging.error(f"Unexpected error for r/{subreddit_name}: {e}")
            break
    logging.error(f"Failed to fetch posts from r/{subreddit_name} after {MAX_RETRIES} attempts.")

# Call the function with a specific subreddit
if __name__ == "__main__":
    subreddit_to_extract = ['genai', 'MachineLearning', 'dataengineering', 'datascience', 'learnmachinelearning', 'tollywood', 'SunrisersHyderabad', 'artificial', 'technology', 'deloitte', 'meta']
    for subreddit in subreddit_to_extract:
        extract_reddit_data(subreddit, num_posts=POSTS_PER_SUBREDDIT)
        logging.info(f"Sleeping for {SLEEP_BETWEEN_SUBS} seconds before next subreddit...")
        time.sleep(SLEEP_BETWEEN_SUBS)
