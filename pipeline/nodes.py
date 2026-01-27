from pipeline.prompts import refineQueryPrompt, mainPrompt, structuringPrompt
from models import llm, embedding_model
from pipeline.state import State
from milvus import collection
import json
def refine_query(state:State):
    res = llm.invoke(refineQueryPrompt.format(query = state['input']))
    state['refined_input'] = res.content
    print(res.content)
    return state



def generate_query_embedding(state: State):
    state['query_embedding'] = embedding_model.feature_extraction(state['refined_input'])
    return state


def get_nearest_menu_items(state:State):
    
    results = collection.search(
        data=[state['query_embedding']],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=10,
        output_fields=["text", "description"]
    )

    
    for hit in results[0]:
        state['top_k_items'].append({'text':hit.entity.get('text'), 'description': hit.entity.get('description')})  
        
    return state


def get_response(state):
    
    
    prompt_menu = "List of food items in the menu \n"
    for item in state['top_k_items']:
        prompt_menu+="item: "+item['text'] + ", description: "+item['description']+" \n"
        
    print(prompt_menu)
    state['prompt_top_k_items'] = prompt_menu
    res = llm.invoke(mainPrompt.format(menu=prompt_menu, query = state['input']))
    state['output'] = res.content
    return state







def make_structured_output(state: State):

    
    res = llm.invoke(structuringPrompt.format(items = state['output'], menu = state['prompt_top_k_items']))
    state['output_structured'] = res.content
    state['items'] = json.loads(state['output_structured'])['items']
    print(state['items'])
    return state