"""Chat LLM clients, configured entirely from .env.

Every provider we use speaks the OpenAI chat-completions API (Ollama Cloud,
Groq, OpenAI, Gemini, DeepInfra, ...), so switching provider means changing
LLM_BASE_URL + LLM_API_KEY + LLM_MODEL in .env — no code change. See
.env.example for ready-made blocks per provider.
"""
from langchain_openai import ChatOpenAI
import os


def _required(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set - copy .env.example to .env and fill it in.")
    return value


def _chat_model(model_name):
    kwargs = dict(
        base_url=_required('LLM_BASE_URL'),
        api_key=_required('LLM_API_KEY'),
        model=model_name,
        timeout=float(os.getenv('LLM_TIMEOUT', '60')),
        max_retries=int(os.getenv('LLM_MAX_RETRIES', '2')),
    )

    temperature = os.getenv('LLM_TEMPERATURE')
    if temperature:
        kwargs['temperature'] = float(temperature)

    # Reasoning models (gpt-oss, gpt-5, ...) think before answering. "low" keeps
    # the waiter fast; leave unset for models that don't support it.
    reasoning_effort = os.getenv('LLM_REASONING_EFFORT')
    if reasoning_effort:
        kwargs['reasoning_effort'] = reasoning_effort

    # Reply-length cap. Must leave room for the model's hidden reasoning tokens,
    # or reasoning models return an empty reply. ChatOpenAI always sends it as
    # `max_completion_tokens`, which OpenAI requires but Ollama ignores — so for
    # every other provider we send the classic `max_tokens` field instead.
    max_tokens = int(os.getenv('LLM_MAX_TOKENS', '1024'))
    if os.getenv('LLM_MAX_TOKENS_FIELD', 'max_tokens') == 'max_completion_tokens':
        kwargs['max_tokens'] = max_tokens
    else:
        kwargs['extra_body'] = {'max_tokens': max_tokens}

    return ChatOpenAI(**kwargs)


LLM_MODEL = _required('LLM_MODEL')

# llm: intent, constraint extraction, menu replies. slm: greet / faq replies —
# defaults to the same model; set LLM_SMALL_MODEL to use a cheaper one.
llm = _chat_model(LLM_MODEL)
slm = _chat_model(os.getenv('LLM_SMALL_MODEL') or LLM_MODEL)
