import yaml
import os
import ollama
from openai import OpenAI

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class LLMProvider:
    def __init__(self):
        self.provider = config.get("llm_provider")

    def summarize_text(self, text, prompt):
        system_prompt = prompt or config.get("summarization_prompt", "Summarize the following text:")

        if self.provider == "openai":
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=config.get("openai_model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
        )
            return response.choices[0].message.content.strip()

        elif self.provider == "ollama":
            response = ollama.chat(
                model=config.get("ollama_model", "mistral"),
                messages=[
                    {"role": "system", "content": "You are a QA assistant. Answer the question ONLY using the provided context. If the context is irrelevant or empty, say ' I don't have enough information from the data. Do not summarize all context, extract only what answers the query."},
                    {"role": "user", "content": text}
                ]
            )
            print("Ollama raw response:", response)
            return response.message.content.strip()

        else:
            raise ValueError(f"Invalid LLM provider: {self.provider}")

