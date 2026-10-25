import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536-dimension vectors


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Embeds chunks using OpenAI's embedding model."""
    embeddings = []

    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model=EMBEDDING_MODEL
        )
        embeddings.append(response.data[0].embedding)

    return embeddings


def embed_user_query(query: str) -> List[float]:
    """Embeds a user query using OpenAI's embedding model."""
    response = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding
