from pipeline.prompts import intentDetectionPrompt
from models import llm
from pipeline.state import State


# to prevent hallucination and make sure only one result comes use langchain withstructuredoutput
def detect_intent(state: State):
    res = llm.invoke(intentDetectionPrompt.format_messages(
        query=state['input'], history=state.get('history', [])))
    state['intent'] = res.content
    print(state['intent'])
    return state

def intent_router(state: State):
    return state['intent']
