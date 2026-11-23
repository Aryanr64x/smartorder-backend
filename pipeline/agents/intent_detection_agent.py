from pipeline.prompts import intentDetectionPrompt
from models import llm
from pipeline.state import State

INTENTS = ('menu_retrieval', 'greet', 'faq', 'out_of_scope')


def _normalise_intent(text):
    """Map the model's reply onto one of INTENTS. Models differ in how they
    answer ("Greet.", "intent: faq", "**menu_retrieval**"), and the graph router
    crashes on anything that isn't an exact key — so match loosely and fall
    back to out_of_scope."""
    cleaned = (text or '').strip().lower()
    for intent in INTENTS:
        if intent in cleaned:
            return intent
    return 'out_of_scope'


# to prevent hallucination and make sure only one result comes use langchain withstructuredoutput
def detect_intent(state: State):
    res = llm.invoke(intentDetectionPrompt.format_messages(
        query=state['input'], history=state.get('history', [])))
    state['intent'] = _normalise_intent(res.content)
    print(repr(res.content), '->', state['intent'])
    return state

def intent_router(state: State):
    return state['intent']
