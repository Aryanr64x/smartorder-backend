from typing import TypedDict, List
class State(TypedDict):
    input: str
    refined_input: str
    output: str
    milvus_rows: list
    query_embedding:list
    top_k_items:list
    prompt_top_k_items: str
    output_structured: str
    items: List[str]