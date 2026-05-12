import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from stores.llm.LLMInterface import LLMInterface

load_dotenv()


class OpenAIProvider(LLMInterface):

    def __init__(self):
        self.base_url = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
        self.api_key = os.getenv("OPENAI_API_KEY", "ollama")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.generation_model = os.getenv("GENERATE_RESPONSE_MODEL", "qwen2.5:3b")
        self.embedding_model = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
        self.embedding_size = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    def set_generation_model(self, model: str):
        self.generation_model = model

    def set_embedding_model(self, model: str, embedding_size: int):
        self.embedding_model = model
        self.embedding_size = embedding_size

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.generation_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict job description assistant. Answer only from the provided context. Never use your own knowledge. Always respond in English only."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
        )
        return response.choices[0].message.content

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def construct_prompt(self, context: str, query: str) -> str:
        return (
            f"You are a job description assistant. Answer ONLY using the context below.\n"
            f"If the context does not contain the answer, say exactly: "
            f"'The provided context does not contain information about this role.'\n"
            f"Keep your answer under 3 sentences. Do not use your own knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )


if __name__ == "__main__":
    provider = OpenAIProvider()

    print("Testing embedding...")
    vector = provider.embed_text("What Python skills are required?")
    print(f"Vector dimension: {len(vector)}")

    print("\nTesting generation...")
    answer = provider.generate_response("What is RAG in AI? Answer in one sentence.")
    print(f"Answer: {answer}")