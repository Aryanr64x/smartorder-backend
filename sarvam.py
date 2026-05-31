from sarvamai import SarvamAI
import base64
sarvam = SarvamAI(
    api_subscription_key="sk_uceyn8vm_OZ0QSjH5doHDX6GouZSNiUsl",
)

response = sarvam.text_to_speech.convert(
    text="Welcome to  Moti Mahal. How are you?",
    target_language_code="hi-IN",
    speaker="priya",
    model="bulbul:v3"
)

audio_base64 = response.audios[0]

audio_bytes = base64.b64decode(audio_base64)

with open("output.wav", "wb") as f:
    f.write(audio_bytes)

print("saved")