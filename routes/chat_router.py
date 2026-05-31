from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pipeline.graph import pipeline
from pipeline.agents.greet_agent import greet_streaming
from pipeline.agents.faq_agent import faq_streaming
from pipeline.agents.menu_retrieval_agent import get_response_streaming, generate_response_for_dbonly_streaming
from schemas import QueryRequest
from sarvam import sarvam                        # sync client — STT only
from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse  # async client — streaming TTS
import asyncio, json, os
from concurrent.futures import ThreadPoolExecutor

chat_router = APIRouter()

SARVAM_API_KEY = os.getenv("sk_uceyn8vm_OZ0QSjH5doHDX6GouZSNiUsl")

INITIAL_STATE = lambda q: {
    'intent': '', 'input': q, 'refined_input': '', 'output': '',
    'milvus_rows': [], 'query_embedding': [], 'top_k_items': [],
    'output_structured': '', 'prompt_top_k_items': '', 'items': [],
    'constraints': {}, 'retriveal_strategy': '', 'database_k_items': []
}

def pick_stream(state: dict):
    intent   = state.get('intent', '')
    strategy = state.get('retriveal_strategy', '')

    if intent == 'greet':
        return greet_streaming(state)
    if intent == 'faq':
        return faq_streaming(state)
    if state.get('output'):        # reject — static string
        return None
    if strategy == 'query':
        return generate_response_for_dbonly_streaming(state)
    return get_response_streaming(state)   # hybrid / rag


# ── Text streaming endpoint (unchanged) ──────────────────────────────────────
@chat_router.post('/stream')
async def chat_stream(request: QueryRequest):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)

    state = await loop.run_in_executor(
        executor,
        lambda: pipeline.invoke(INITIAL_STATE(request.query))
    )

    print("INTENT:", state.get('intent'))
    print("STRATEGY:", state.get('retriveal_strategy'))
    print("OUTPUT:", state.get('output'))
    print("ITEMS:", state.get('items'))

    llm_stream = pick_stream(state)
    print("STREAM:", llm_stream)

    async def generate():
        if llm_stream is None:
            text = state.get('output', '')
            print("STATIC TEXT:", repr(text))
            words = text.split(' ')
            for i, word in enumerate(words):
                yield f"data: {json.dumps({'token': word + ('' if i == len(words)-1 else ' ')})}\n\n"
            yield f"data: {json.dumps({'items': state.get('items', []), 'done': True})}\n\n"
            return

        def _next(it):
            try:
                return next(it)
            except StopIteration:
                return None

        chunk_count = 0
        while True:
            chunk = await loop.run_in_executor(executor, _next, llm_stream)
            if chunk is None:
                break
            if chunk.content:
                chunk_count += 1
                print(f"CHUNK {chunk_count}:", repr(chunk.content))
                yield f"data: {json.dumps({'token': chunk.content})}\n\n"

        print("STREAM DONE, total chunks:", chunk_count)
        yield f"data: {json.dumps({'items': state.get('items', []), 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Voice streaming endpoint ──────────────────────────────────────────────────
@chat_router.post('/voice/stream')
async def voice_stream(audio: UploadFile = File(...)):
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Expected an audio file.")

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Received empty audio file.")

    print(f"[voice] Audio received | size={len(audio_bytes):,} bytes")

    # ── STT (full, unchanged) ─────────────────────────────────────────
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            stt_response = sarvam.speech_to_text.transcribe(
                file=f,
                model="saarika:v2.5",
                language_code="unknown",
            )
        transcript = stt_response.transcript
    finally:
        os.unlink(tmp_path)

    print(f"[voice] STT transcript: {transcript!r}")

    # ── Run retrieval graph ───────────────────────────────────────────
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)

    state = await loop.run_in_executor(
        executor,
        lambda: pipeline.invoke(INITIAL_STATE(transcript))
    )

    llm_stream = pick_stream(state)

    async def generate():
        # 1. Send transcript so frontend can update user bubble immediately
        yield f"data: {json.dumps({'type': 'transcript', 'content': transcript})}\n\n"

        # 2. Collect full LLM response (needed before TTS can start)
        if llm_stream is None:
            response_text = state.get('output', '')
        else:
            def _next(it):
                try: return next(it)
                except StopIteration: return None

            response_text = ""
            while True:
                chunk = await loop.run_in_executor(executor, _next, llm_stream)
                if chunk is None:
                    break
                if chunk.content:
                    response_text += chunk.content

        print(f"[voice] LLM response: {response_text!r}")

        # 3. Send text frame — frontend shows it immediately
        yield f"data: {json.dumps({'type': 'text', 'content': response_text, 'items': state.get('items', [])})}\n\n"

        # 4. Stream TTS via Sarvam AsyncSarvamAI WebSocket
        try:
            async_client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

            async with async_client.text_to_speech_streaming.connect(
                model="bulbul:v3",
                send_completion_event=True
            ) as ws:
                await ws.configure(
                    target_language_code="hi-IN",
                    speaker="priya",
                )

                await ws.convert(response_text)
                await ws.flush()

                async for message in ws:
                    if isinstance(message, AudioOutput):
                        # message.data.audio is already base64 encoded
                        yield f"data: {json.dumps({'type': 'audio_chunk', 'data': message.data.audio})}\n\n"
                    elif isinstance(message, EventResponse):
                        if message.data.event_type == "final":
                            break

        except Exception as e:
            print(f"[voice] TTS streaming error: {e}")
            # fallback to non-streaming TTS
            tts_response = sarvam.text_to_speech.convert(
                text=response_text,
                target_language_code="hi-IN",
                speaker="priya",
                model="bulbul:v3"
            )
            yield f"data: {json.dumps({'type': 'audio_chunk', 'data': tts_response.audios[0]})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ── Original non-streaming endpoints (unchanged) ──────────────────────────────
@chat_router.post('')
def test(request: QueryRequest):
    res = pipeline.invoke(INITIAL_STATE(request.query))
    return {
        'response_text': res['output'],
        'items': res['items']
    }


@chat_router.post('/voice')
async def voice(audio: UploadFile = File(...)):
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Expected an audio file.")

    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Received empty audio file.")
    

    print(f"[voice] Audio received | size={len(audio_bytes):,} bytes")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            stt_response = sarvam.speech_to_text.transcribe(
                file=f,
                model="saarika:v2.5",
                language_code="unknown",
            )
        transcript = stt_response.transcript
    finally:
        os.unlink(tmp_path)

    print(f"[voice] STT transcript: {transcript!r}")

    res = pipeline.invoke(INITIAL_STATE(transcript))
    response_text = res['output']
    print(f"[voice] pipeline output: {response_text!r}")

    tts_response = sarvam.text_to_speech.convert(
        text=response_text,
        target_language_code="hi-IN",
        speaker="priya",
        model="bulbul:v3"
    )
    audio_base64 = tts_response.audios[0]

    return {
        'response_text': response_text,
        'items': res['items'],
        'transcript': transcript,
        'audio_base64': audio_base64,
        'audio_mime': 'audio/wav',
    }