from pipeline.state import State

def reject(state: State):
    state['output'] = "I don't have information related to that. Please ask about the restaurant menu only !"
    return state
