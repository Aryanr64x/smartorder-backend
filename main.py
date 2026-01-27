from pipeline.graph import pipeline
from fastapi import FastAPI
from schemas import QueryRequest
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()


app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000", 'https://smartorder-frontend-nine.vercel.app'],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"])


@app.post('/menu')
def test(request: QueryRequest):
    res = pipeline.invoke({'input': request.query,  'refined_input':'' ,'output':'', 'milvus_rows':[], 'query_embedding':[], 'top_k_items':[], 'output_structured': '', 'prompt_top_k_items': '', 
                           'items': []})
    return {
        'response_text': res['output'],
        'items': res['items']
    }



