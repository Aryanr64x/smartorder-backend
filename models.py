from huggingface_hub import InferenceClient
import os

# Chat LLMs come from llm_provider.py and are switched via .env only.
from llm_provider import llm, slm

embedding_model = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    provider='hf-inference',
    api_key=os.getenv('HUGGINGFACE_TOKEN'))

