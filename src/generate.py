from .llm_utils import LLMProvider

class Generate:
    def __init__(self, max_docs = 5):
        self.max_docs = max_docs
        self.llm_provider = LLMProvider()

    def summarize(self, results):
        text = "\n\n".join([result["text"] for result in results[:self.max_docs]])
        return self.llm_provider.summarize_text(text)

    def answer(self, question, results):
        # print(results)
        context = "\n\n".join([result["text"] for result in results[:self.max_docs]])
        prompt = f"Answer the following question based on the context provided:\n\nQuestion:\n{question}\n\nContext:\n{context}"
        return self.llm_provider.summarize_text(context,prompt)
