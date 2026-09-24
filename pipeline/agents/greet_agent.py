from models import slm
from pipeline.state import State
from pipeline.prompts import greetPrompt

def greet(state: State):
    res = slm.invoke(greetPrompt.format_messages(
        greet=state['input'], history=state.get('history', []),
        restaurant_name=state['restaurant_name']))
    state['output'] = res.content
    return state


def greet_streaming(state: State):
    """Call outside graph for streaming path."""
    return slm.stream(greetPrompt.format_messages(
        greet=state['input'], history=state.get('history', []),
        restaurant_name=state['restaurant_name']))