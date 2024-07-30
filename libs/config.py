import os
from typing import Any, Dict


class Config:
    def __init__(self):
        self.COHERE_API_KEY: str = os.environ.get("COHERE_API_KEY", "")
        self.PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
        self.PINECONE_INDEX: str = os.environ.get("PINECONE_INDEX", "")
        self.LLAMA_CLOUD_API_KEY: str = os.environ.get("LLAMA_CLOUD_API_KEY", "")

    def get_secrets(self) -> Dict[str, Any]:
        return {
            "COHERE_API_KEY": self.COHERE_API_KEY,
            "PINECONE_API_KEY": self.PINECONE_API_KEY,
            "PINECONE_INDEX": self.PINECONE_INDEX,
            "LLAMA_CLOUD_API_KEY": self.LLAMA_CLOUD_API_KEY,
        }
