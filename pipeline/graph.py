from langgraph.graph import StateGraph
from pipeline.state import State
from pipeline.agents.intent_detection_agent import detect_intent, intent_router
from pipeline.agents.greet_agent import greet
from pipeline.agents.faq_agent import faq
from pipeline.agents.guardrails_agent import reject
from pipeline.agents.menu_retrieval_agent import refine_query, generate_query_embedding, get_nearest_menu_items, get_response, make_structured_output



graph = StateGraph(State)
# technically constraint logic and all
# business pov ops and sassification check and all



graph.add_node('detect_intent', detect_intent)
graph.add_node('greet', greet)
graph.add_node('faq', faq)
graph.add_node('reject', reject)
graph.add_node('refine_query', refine_query)
graph.add_node('generate_query_embedding', generate_query_embedding)
graph.add_node('get_nearest_menu_items', get_nearest_menu_items)
graph.add_node('get_response', get_response)
graph.add_node('make_structured_output', make_structured_output)


graph.add_conditional_edges('detect_intent', intent_router , {'greet': 'greet', 'menu_retrieval': 'refine_query', 'faq': 'faq', 'out_of_scope': 'reject'})
graph.add_edge('refine_query', 'generate_query_embedding')
graph.add_edge('generate_query_embedding', 'get_nearest_menu_items')
graph.add_edge('get_nearest_menu_items', 'get_response')
graph.add_edge('get_response', 'make_structured_output')
graph.set_entry_point('detect_intent')
            
graph.set_finish_point('make_structured_output')

pipeline = graph.compile()