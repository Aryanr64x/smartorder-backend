from pipeline.prompts import intentDetectionPrompt
from models import llm
from pipeline.state import State

def detect_intent(state: State):
    res = llm.invoke(intentDetectionPrompt.format(query = state['input']))
    state['intent'] = res.content
    return state

def intent_router(state: State):
    return state['intent']
