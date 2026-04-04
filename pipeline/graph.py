from langgraph.graph import StateGraph, END
from pipeline.state import State
from pipeline.agents.intent_detection_agent import detect_intent, intent_router
from pipeline.agents.greet_agent import greet
from pipeline.agents.faq_agent import faq
from pipeline.agents.guardrails_agent import reject
from pipeline.agents.menu_retrieval_agent import refine_query, generate_query_embedding, get_nearest_menu_items, get_response, constraint_extractor, retrieval_strategy_decider, retriveal_strategy_router, query_constraints, generate_response_for_dbonly, database_used_router, fetch_db_details, fetch_loaded_db_details



graph = StateGraph(State)
# technically constraint logic and all
# business pov ops and sassification check and all



graph.add_node('detect_intent', detect_intent)
graph.add_node('greet', greet)
graph.add_node('faq', faq)
graph.add_node('reject', reject)
graph.add_node('constraint_extractor', constraint_extractor)
graph.add_node('retrieval_strategy_decider', retrieval_strategy_decider)
graph.add_node('query_constraints', query_constraints)
graph.add_node('refine_query', refine_query)
graph.add_node('generate_query_embedding', generate_query_embedding)
graph.add_node('get_nearest_menu_items', get_nearest_menu_items)
graph.add_node('fetch_db_details', fetch_db_details)
graph.add_node('fetch_loaded_db_details', fetch_loaded_db_details)
graph.add_node('get_response', get_response)
graph.add_node('generate_response_for_dbonly', generate_response_for_dbonly)


graph.add_conditional_edges('detect_intent', intent_router , {'greet': 'greet', 'menu_retrieval': 'constraint_extractor', 'faq': 'faq', 'out_of_scope': 'reject'})
graph.add_edge('constraint_extractor', 'retrieval_strategy_decider')
graph.add_conditional_edges('retrieval_strategy_decider', retriveal_strategy_router , {'hybrid': 'query_constraints', 'query': 'query_constraints', 'rag': 'refine_query', 'fallback': 'reject'})
graph.add_conditional_edges('query_constraints', retriveal_strategy_router, {'hybrid': 'refine_query', 'query': 'generate_response_for_dbonly', 'rag': 'reject', 'fallback': 'reject'})
graph.add_edge('refine_query', 'generate_query_embedding')
graph.add_edge('generate_query_embedding', 'get_nearest_menu_items')
graph.add_conditional_edges('get_nearest_menu_items', database_used_router, {'yes': 'fetch_loaded_db_details', 'no': 'fetch_db_details'})
graph.add_edge('fetch_db_details', 'get_response')
graph.add_edge('fetch_loaded_db_details', 'get_response')



#end egdes 
graph.add_edge('get_response', END)
graph.add_edge('generate_response_for_dbonly', END)

graph.set_entry_point('detect_intent')
            
  
pipeline = graph.compile()                 