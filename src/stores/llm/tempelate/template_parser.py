"""
builds the final prompt from retrieved chunks and user query.
"""

from stores.llm.tempelate.locales.en import rag as en_rag


class TemplateParser:

    def __init__(self, language: str = "en"):
        self.language = language

    def build_prompt(self, context_chunks: list[str], query: str) -> str:
        context = "\n\n---\n\n".join(context_chunks)

        if self.language == "en":
            return en_rag.RAG_PROMPT.format(context=context, query=query)

        # default fallback
        return en_rag.RAG_PROMPT.format(context=context, query=query)

    def get_system_prompt(self) -> str:
        if self.language == "en":
            return en_rag.SYSTEM_PROMPT
        return en_rag.SYSTEM_PROMPT