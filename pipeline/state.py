from typing import TypedDict, List, Dict

class State(TypedDict):
    # Prior conversation turns as LangChain messages (HumanMessage / AIMessage).
    # Injected into every prompt's MessagesPlaceholder("history") so the model
    # sees the running conversation without the system prompt being duplicated.
    history: list
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
    # Which restaurant this guest is seated at. Every menu lookup — SQL and
    # vector — is filtered by it, so a guest only ever sees their own menu.
    restaurant_id: int
    table_id: int
    # From the `restaurants` row, so prompts speak as *this* restaurant's waiter
    # (name) and can answer FAQs about it (meta_data). See tenancy.restaurant_profile.
    restaurant_name: str
    restaurant_info: str
    