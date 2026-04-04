from pipeline.prompts import refineQueryPrompt, mainPrompt, structuringPrompt, constraintExtractorPrompt,responseTextPromptFromDBItems
from models import llm, embedding_model
from pipeline.state import State
from milvus import collection
import json
from supabase_client import supabase


def constraint_extractor(state: State):
    res = llm.invoke(constraintExtractorPrompt.format(query = state['input']))
    state['constraints'] = json.loads(res.content)
    print(state['constraints'])
    return state

def retrieval_strategy_decider(state: State):
    
    def is_present(val):
        return val is not None and val != "null"
    
    sql_fields = ['food_type', 'vegnonveg', 'max_price', 'min_price']
    sql_present = any(is_present(state['constraints'].get(f)) for f in sql_fields)
    other = state['constraints'].get('other', False)

    if sql_present and other:
        state['retriveal_strategy'] = "hybrid"
    elif sql_present:
        state['retriveal_strategy'] = "query"
    elif other:
        state['retriveal_strategy'] = "rag"
    else:
        state['retriveal_strategy'] = "fallback"

    print("RETRIEVAL STRATEGY IS: ", state['retriveal_strategy'])
    return state
        
def retriveal_strategy_router(state: State):
    
    return state['retriveal_strategy']


def query_constraints(state: State):
    constraints = state["constraints"]

    query = supabase.table("menu").select("*")

    # food_type
    if constraints.get("food_type") not in [None, "null"]:
        query = query.eq("food_type", constraints["food_type"])

    # veg/nonveg
    if constraints.get("vegnonveg") not in [None, "null"]:
        is_veg = True if constraints["vegnonveg"] == "veg" else False
        query = query.eq("vegnonveg", is_veg)


    max_price = constraints.get("max_price")
    min_price = constraints.get("min_price")

    
    if max_price not in [None, 0, "null"]:
        query = query.lte("price", max_price)

    if min_price not in [None, 10000, "null"]:
        query = query.gte("price", min_price)

    
    if max_price == 0:
        # "cheapest"
        query = query.order("price", desc=False)

    elif min_price == 10000:
        # "most expensive"
        query = query.order("price", desc=True)

    else:
        # default ranking (optional)
        query = query.order("price", desc=False)

    # limit
    response = query.limit(10).execute()
    data = response.data
    state['database_k_items'] = data
    print(state['database_k_items'])
    return state


def generate_response_for_dbonly(state: State):
    items = state['database_k_items']
    item_names = []
    for item in items:
        item_names.append(item['name'])
        
    
    res = llm.invoke(responseTextPromptFromDBItems.format(items = '\n'.join(item_names)))
    state['output'] = res.content
    
    state['items'] = state['database_k_items']
    
    return state




def refine_query(state: State):
    res = llm.invoke(refineQueryPrompt.format(query = state['input']))
    state['refined_input'] = res.content
    print(res.content)
    return state

def generate_query_embedding(state: State):
    state['query_embedding'] = embedding_model.feature_extraction(state['refined_input'])
    return state

def get_nearest_menu_items(state: State):
    ids = []
    if(state['database_k_items']):
        for item in state['database_k_items']:
            ids.append(item['id'])
    
    
    results = collection.search(
        data=[state['query_embedding']],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=5,
        output_fields=["sql_id"],
        expr=f"sql_id in {ids}" if ids else None
    )

    
    for hit in results[0]:
        state['top_k_items'].append(hit.entity.get('sql_id'))  
        
    return state


def database_used_router(state: State):
    res = "no"
    
    print("DATABASE HAS ITEMS? ", state['database_k_items'])
    if (len(state['database_k_items']) != 0):
        res = "yes"
    
    return res


def fetch_db_details(state: State):
    print("ALWAYS HITTING THIS?")
    ids = state['top_k_items']
    res = supabase.table('menu').select('*').in_('id', ids).execute()
    state['items'] = res.data
    return state

def fetch_loaded_db_details(state: State):
    fetched_ids = state['top_k_items']
    db_rows = state['database_k_items']
    print(db_rows)
    print(fetched_ids)
    print("-----------------------------------------")
    state['items'] = [row for row in db_rows if row['id'] in fetched_ids]
    return state

def get_response(state):  
    prompt_menu = "List of food items in the menu \n"
    for item in state['items']:
        prompt_menu+="item: "+item['name'] + ", description: "+item['description']+" \n"
        
    print(prompt_menu)
    state['prompt_top_k_items'] = prompt_menu
    res = llm.invoke(mainPrompt.format(menu=prompt_menu, query = state['input']))
    state['output'] = res.content
    return state

