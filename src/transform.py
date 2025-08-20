# import modules
import pandas as pd
import glob
import re
import string
import os

'''
Cleans the input text by removing unwanted characters and formatting.
Input: "Hello, World! Visit https://example.com"
Output: "hello world visit"
'''
def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

'''
Transform the raw Reddit data into a cleaned and structured format.
Input: raw Reddit CSV files
Output : cleaned and structured DataFrame written to a CSV file
'''
def transform_data(input_folder, output_path):
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    df_list = [pd.read_csv(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)
    # Keep only relevant columns
    cols_to_keep = [c for c in ["subreddit", "title", "selftext", "created_utc", "score"] if c in df.columns]
    df = df[cols_to_keep]
    # Clean text columns
    if "title" in df.columns:
        df["title_clean"] = df["title"].apply(clean_text)
    if "selftext" in df.columns:
        df["selftext_clean"] = df["selftext"].apply(clean_text)
    # Create text length feature and day of week feature
    if "selftext_clean" in df.columns:
        df["text_length"] = df["selftext_clean"].apply(len)
    if "created_utc" in df.columns:
        df["created_day"] = pd.to_datetime(df["created_utc"], unit='s').dt.day_name()

    # Save the transformed DataFrame
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Transformed data saved to {output_path}")

if __name__ == "__main__":
    transform_data("data/raw", "data/processed/reddit_posts_cleaned.csv")
