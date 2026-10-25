from embedder import embed_user_query
from vectorstore import search_in_pinecone
from llm import query_llm_with_context


def process_user_query(query: str):
    # Embed the user's query to create a vector representation
    query_vector = embed_user_query(query)

    # Search the vector DB to find top matching chunks related to the user's question
    matched_chunks = search_in_pinecone(query_vector)

    # Send the user query and search results (query + context) to the LLM for generating response
    generated_response = query_llm_with_context(
        query,
        matched_chunks
    )

    print(generated_response)


if __name__ == "__main__":
    user_query = "What is the work timing policy?"
    process_user_query(user_query)
