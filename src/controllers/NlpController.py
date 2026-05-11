"""
samaXjana
NLP Controller
 → Embed the search query
 → Find top-K similar chunks
 →  Build the prompt from retrieved chunks
 → Call LLM API → Get answer
"""

import os
from stores.llm.LLMFactory import LLMFactory
from stores.llm.tempelate.template_parser import TemplateParser
from stores.vectordb.provider.qdrant_provider import QdrantProvider


class NlpController:

    def __init__(self):
        self.llm = LLMFactory.create()
        self.template_parser = TemplateParser(language="en")
        self.db = QdrantProvider(
            db_path=os.getenv("VECTOR_DB_PATH", "assets/db/qdrant_data"),
            collection_name=os.getenv("COLLECTION_NAME", "job_documents"),
            vector_size=int(os.getenv("EMBEDDING_DIMENSION", "768")),
        )

    def answer_query(self, query: str, top_k: int = 5) -> dict:
        query_vector = self.llm.embed_text(query)

        results = self.db.search(query_vector=query_vector, top_k=top_k)

        # Remove duplicate chunks (same text from same file)
        seen = set()
        unique_results = []
        for r in results:
            if r["text"] not in seen:
                seen.add(r["text"])
                unique_results.append(r)
        results = unique_results

        if not results:
            return {
                "query": query,
                "answer": "No relevant documents found in the database.",
                "retrieved_chunks": []
            }

        #build the prompt
        chunks_text = [r["text"] for r in results]
        prompt = self.template_parser.build_prompt(
            context_chunks=chunks_text,
            query=query
        )

        #generate the answer
        answer = self.llm.generate_response(prompt)
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": results
        }


if __name__ == "__main__":
    controller = NlpController()

    query = "What are the responsibilities of a sales manager?"
    print(f"Query: {query}\n")

    result = controller.answer_query(query, top_k=5)

    print(f"Answer:\n{result['answer']}\n")
    print(f"Retrieved {len(result['retrieved_chunks'])} chunks:")
    for i, chunk in enumerate(result['retrieved_chunks']):
        print(f"\n  Chunk {i+1} (score: {chunk['score']:.3f}) from {chunk['source_file']}")
        print(f"  {chunk['text'][:150]}...")