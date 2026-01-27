from langgraph.graph import StateGraph
from pipeline.state import State
from pipeline.nodes import refine_query, generate_query_embedding, get_nearest_menu_items, get_response, make_structured_output



graph = StateGraph(State)



graph.add_node('refine_query', refine_query)
graph.add_node('generate_query_embedding', generate_query_embedding)
graph.add_node('get_nearest_menu_items', get_nearest_menu_items)
graph.add_node('get_response', get_response)
graph.add_node('make_structured_output', make_structured_output)



graph.add_edge('refine_query', 'generate_query_embedding')
graph.add_edge('generate_query_embedding', 'get_nearest_menu_items')
graph.add_edge('get_nearest_menu_items', 'get_response')
graph.add_edge('get_response', 'make_structured_output')
graph.set_entry_point('refine_query')

graph.set_finish_point('make_structured_output')

pipeline = graph.compile()