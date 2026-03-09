from models import slm
from pipeline.state import State
from pipeline.prompts import greetPrompt

def greet(state: State):
    res = slm.invoke(greetPrompt.format(greet = state['input']))
    state['output'] = res.content
    return state
