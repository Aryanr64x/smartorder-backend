from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import InferenceClient
import os

model = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_TOKEN'),
    temperature=0.5,
    max_new_tokens=100,
    provider='auto')

llm = ChatHuggingFace(llm = model)

embedding_model = InferenceClient(
    model="sentence-transformers/all-MiniLM-L6-v2",
    provider='hf-inference',
    api_key=os.getenv('HUGGINGFACE_TOKEN'))  
