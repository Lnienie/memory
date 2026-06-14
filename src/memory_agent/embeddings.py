"""Embedding model configuration for LangGraph Store semantic search."""

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    check_embedding_ctx_length=False,
)

__all__ = ["embeddings"]
