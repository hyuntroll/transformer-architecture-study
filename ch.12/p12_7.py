from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

pinecone_api_key =  os.getenv("PINECONE-API")

pc = Pinecone(api_key=pinecone_api_key)


pc.create_index('llm-book', spec=ServerlessSpec("aws", "us-east-1"), dimension=768)
index = pc.Index('llm-book')