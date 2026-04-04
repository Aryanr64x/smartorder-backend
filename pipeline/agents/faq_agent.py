from pipeline.state import State
from models import slm 
from pipeline.prompts import faqPrompt

def faq(state: State):
    res = slm.invoke(faqPrompt.format(query = state['input']))
    state['output'] = res.content
    return state
