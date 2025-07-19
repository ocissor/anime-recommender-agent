import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "anime-embeddings"

if not pc.has_index(index_name):
    logging.info(f"Creating index: {index_name}")
    # Create a new index with the specified name and configuration
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )


# Connect to the existing index
index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

