from typing import TypedDict, List, Dict

class State(TypedDict):
    intent: str
    input: str
    refined_input: str
    output: str
    milvus_rows: list
    query_embedding:list
    top_k_items:list
    database_k_items: list
    prompt_top_k_items: str
    output_structured: str
    items: list
    constraints: Dict
    retriveal_strategy: str
    