# import modules
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

class Vectorizer:
    def __init__(self, input_file, output_file, method="tfidf"):
        '''
        method: 'tfidf' or 'embeddings'
        '''
        self.method = method
        self.input_file = input_file
        self.output_file = output_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english'
            )
        elif method == "embeddings":
            self.vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            raise ValueError("Invalid method. Choose 'tfidf' or 'embeddings'.")
        
    def load_data(self):
        return pd.read_csv(self.input_file)

    def save_parquet(self, df, name):
        output_file = os.path.join(self.output_file, f"{name}.parquet")
        df.to_parquet(output_file, index=False)
        print(f"✅ Saved {name}  to {output_file}")

    def fit_transform(self, df):
        if self.method == "tfidf":
            X = self.vectorizer.fit_transform(df["selftext_clean"].fillna(""))
            tf_idf_df = pd.DataFrame(X.toarray(), columns=[f"tfidf_{i}" for i in range(X.shape[1])])
            return pd.concat([df.reset_index(drop=True), tf_idf_df], axis=1)
        elif self.method == "embeddings":
            X = self.vectorizer.encode(df["selftext_clean"].fillna("").to_list(), show_progress_bar=True)
            emb_df = pd.DataFrame(X, columns=[f"emb_{i}" for i in range(X.shape[1])])
            return pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    def run(self):
        df = self.load_data()
        if self.method == "tfidf":
            tfidf_out = self.fit_transform(df)
            self.save_parquet(tfidf_out, "reddit_posts_tfidf")
        elif self.method == "embeddings":
            embeddings_out = self.fit_transform(df)
            self.save_parquet(embeddings_out, "reddit_posts_embeddings")

if __name__ == "__main__":
    vectorizer = Vectorizer(
        input_file="data/processed/reddit_posts_cleaned.csv",
        output_file="data/vectorized",
        method="tfidf"
    )
    vectorizer.run()
    vectorizer = Vectorizer(
        input_file="data/processed/reddit_posts_cleaned.csv",
        output_file="data/vectorized",
        method="embeddings"
    )
    vectorizer.run()