import os
import pandas as pd
from dotenv import load_dotenv
from search import Search
from generate import Generate
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

load_dotenv()

class Evaluate:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.search = Search()
        self.generate = Generate()
        self.embedder = SentenceTransformer(model_name)

    def evaluate(self, eval_file='data/eval/eval_set.csv'):
        df = pd.read_csv(eval_file)
        results = []
        for _, row in df.iterrows():
            question = row['question']
            expected_keywords = [kw.strip().lower() for kw in row['expected_keywords'].split(',')]
            reference_answer = row.get('reference_answer', '')

            retrieved_docs = self.search.search(question)
            generated_answer = self.generate.summarize(question, retrieved_docs)

            gen_text_lower = generated_answer.lower()
            hits = sum(1 for kw in expected_keywords if kw in gen_text_lower)
            precision = hits / len(expected_keywords) if expected_keywords else 0

            if reference_answer:
                gen_emb = self.embedder.encode([generated_answer])
                ref_emb = self.embedder.encode([reference_answer])
                similarity = cosine_similarity(gen_emb, ref_emb)[0][0]
            else:
                similarity = None

            results.append({
                'question': question,
                'generated_answer': generated_answer,
                'reference_answer': reference_answer,
                'precision': precision,
                'similarity': similarity
            })

        results_df = pd.DataFrame(results)
        results_path = 'data/eval/eval_results.csv'
        os.makedirs('data/eval', exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"✅ Evaluation results saved to {results_path}")
        return results_df

if __name__ == "__main__":
    evaluator = Evaluate()
    evaluator.evaluate()