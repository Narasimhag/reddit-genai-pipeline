import os
from openai import OpenAI

class Generator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"
        self.max_tokens = 200
        self.temperature = 0.0
    
    def generate_answer(self, query, docs):
        context = "\n\n".join([doc[:500] for doc in docs[:3]])

        prompt = f"""
        Summarize the following context in relation to the query.

        Query: {query}

        Context: {context}

        Please provide a concise summary based on the context and query (max 3 sentences).
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return response.choices[0].message.content.strip()