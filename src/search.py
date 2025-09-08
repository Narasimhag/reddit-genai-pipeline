from .rerank import Reranker
from .retrieve import Retriever

class Search:
    def __init__(self, index_name="reddit-genai", top_k_retrieve=50, top_k_rerank=5):
        self.retriever = Retriever(index_name=index_name)
        self.reranker = Reranker()
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

    def search(self, query):
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.search(query, top_k=self.top_k_retrieve)

        # Step 2: Rerank the retrieved documents
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=self.top_k_rerank)

        return reranked_docs

if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    search = Search(top_k_rerank=20)
    results = search.search(user_query)
    print("Search Results:")
    print(results)