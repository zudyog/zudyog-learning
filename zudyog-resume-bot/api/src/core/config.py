
import os
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_pinecone_index(api_key=None):
    if api_key is None:
        api_key = os.environ.get('PINECONE_API_KEY')
        if api_key is None:
            raise ValueError("API key is required")

    try:
        # Initialize Pinecone client
        pc = Pinecone(
            api_key=api_key
        )

        index_name = "raptai-search"

        # Check if index exists, if not create it
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # or your preferred AWS region
                )
            )

        # Get the index
        return pc.Index(index_name)
    except Exception as e:
        print(f"An error occurred while initializing Pinecone index: {e}")
        raise


def get_openai_client(api_key=None):
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("API key is required")

    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        raise ValueError("Invalid API key") from e


# Export the functions
__all__ = ['get_pinecone_index', 'get_openai_client']
