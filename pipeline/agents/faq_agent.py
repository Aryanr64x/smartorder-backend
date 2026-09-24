from pipeline.state import State
from models import slm 
from pipeline.prompts import faqPrompt

def faq(state: State):
    res = slm.invoke(faqPrompt.format_messages(
        query=state['input'], history=state.get('history', []),
        restaurant_name=state['restaurant_name'], restaurant_info=state['restaurant_info']))
    state['output'] = res.content
    return state


def faq_streaming(state: State):
    """Call outside graph for streaming path."""
    return slm.stream(faqPrompt.format_messages(
        query=state['input'], history=state.get('history', []),
        restaurant_name=state['restaurant_name'], restaurant_info=state['restaurant_info']))
