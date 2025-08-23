import os
import io
import json
import sys
import tempfile
from typing import List
from dotenv import load_dotenv
import streamlit as st

# Optional mic component. If missing, fall back to file upload.
try:
    from streamlit_mic_recorder import mic_recorder  # pip install streamlit-mic-recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False

# -----------------------------
# Google AI Studio (Gemini)
# -----------------------------
# pip install google-generativeai
import google.generativeai as genai


# -----------------------------
# Whisper Speech-to-Text (local)
# -----------------------------
# pip install openai-whisper torch pydub
import whisper
from pydub import AudioSegment

# -----------------------------
# Text-to-Speech (simple & free)
# -----------------------------
from gtts import gTTS  # pip install gTTS

# Optional translation for playback
try:
    from googletrans import Translator  # pip install googletrans==4.0.0-rc1
    translator = Translator()
    HAS_TRANSLATE = True
except Exception:
    HAS_TRANSLATE = False

# =============================
# App Config
# =============================
st.set_page_config(page_title="Genie Voice Tutor (Gemini)", page_icon="🎙️", layout="centered")
APP_TITLE = "🎙️ Genie AI Voice Tutor (Gemini + Google STT)"

# Models / languages
load_dotenv()  # Ensure .env is loaded before reading GEMINI_MODEL
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # fast & generous free tier
SUPPORTED_LANGS = ["English", "Hindi", "Marathi", "Tamil"]
LANG_CODE = {"English": "en", "Hindi": "hi", "Marathi": "mr", "Tamil": "ta"}

# Roleplay content (embedded for single-file deploy)
ROLEPLAY_SCENARIOS = {
    "school": {
        "title": "At School",
        "steps": [
            {"ai": "Good morning! What's your name?", "hints": ["My name is Aarav", "My name is Anya"]},
            {"ai": "Do you like school?", "hints": ["Yes, I like school", "I like my school"]},
            {"ai": "What is your favorite subject?", "hints": ["Math", "English", "Science"]}
        ]
    },
    "store": {
        "title": "At the Store",
        "steps": [
            {"ai": "Welcome! What do you want to buy today?", "hints": ["I want a banana", "I want milk"]},
            {"ai": "Great! How many do you need?", "hints": ["One", "Two", "Three"]}
        ]
    },
    "home": {
        "title": "At Home",
        "steps": [
            {"ai": "Who do you live with?", "hints": ["I live with my parents", "I live with my mom and dad"]},
            {"ai": "That's nice! Do you help at home?", "hints": ["Yes, I help", "I help my parents"]}
        ]
    }
}

# =============================
# Setup Helpers
# =============================
@st.cache_resource(show_spinner=False)
  # pip install python-dotenv

def init_gemini():
    load_dotenv()  # Load variables from .env if present
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model
    except Exception:
        return None


# =============================
# Whisper Transcription Helper
# =============================
def transcribe_whisper(audio_bytes: bytes, lang_code: str = "en") -> str:
    """Transcribe audio using OpenAI Whisper. Accepts bytes, returns text."""
    # Save to temp WAV file
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            audio.export(tmp.name, format="wav")
            model = whisper.load_model("base")  # You can use "tiny", "base", "small", "medium", "large"
            result = model.transcribe(tmp.name, language=lang_code)
            return result.get("text", "").strip() or "(No speech detected)"
    except Exception as e:
        return f"(Transcription error: {e})"





def gemini_reply(model, user_text: str, persona: str = "tutor") -> str:
    """Call Gemini and return a short, kid-safe reply. Robust to SDK variations."""
    if not model:
        return "(AI unavailable: set GOOGLE_API_KEY)"

    if persona == "roleplay":
        sys = (
            "You are part of a child-friendly roleplay. Use warm, simple sentences (max 2). "
            "Stay on safe, kid-appropriate topics and never ask for personal data."
        )
    else:
        sys = (
            "You are Genie, a friendly tutor for children aged 6–12. Explain clearly in <= 2 short sentences "
            "and ask one gentle follow-up question. Avoid personal data and unsafe topics."
        )

    # Build prompt without f-string to avoid any accidental brace issues
    prompt = ("{sys}\n\nChild said: {user}\n\nYour reply:").format(sys=sys, user=user_text)


    try:
        resp = model.generate_content(prompt)
        # Preferred: .text
        if getattr(resp, "text", None):
            return resp.text.strip()
        # Fallback: stitch from candidates/parts if .text is empty
        candidates = getattr(resp, "candidates", []) or []
        parts_out = []
        for c in candidates:
            content = getattr(c, "content", None)
            if content and getattr(content, "parts", None):
                for p in content.parts:
                    t = getattr(p, "text", None)
                    if t:
                        parts_out.append(t)
        if parts_out:
            return " ".join(parts_out).strip()
        return "(No response)"
    except Exception as e:
        return f"(Model error: {e})"




def maybe_translate(text: str, target_lang: str) -> str:
    if target_lang == "English" or not text:
        return text
    if not HAS_TRANSLATE:
        return f"[Translation unavailable] {text}"
    try:
        code = LANG_CODE.get(target_lang, "en")
        return translator.translate(text, dest=code).text
    except Exception:
        return f"[Translation failed] {text}"


def speak_gtts(text: str, target_lang: str) -> io.BytesIO:
    lang = LANG_CODE.get(target_lang, "en")
    if not text or not text.strip():
        text = "I didn't hear anything. Please try again."
    tts = gTTS(text=text, lang=lang)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf


def emoji_feedback(text: str) -> str:
    if not text:
        return "🧩"
    n = len(text.split())
    if n <= 3:
        return "🙂"
    if "?" in text or "!" in text:
        return "🌟"
    if n >= 12:
        return "👍"
    return "🙂"

# =============================
# UI
# =============================
st.title(APP_TITLE)
with st.sidebar:
    st.header("Settings")
    playback_lang = st.selectbox("Playback Language", SUPPORTED_LANGS, index=0)
    mode = st.radio("Mode", ["Free Chat", "Roleplay"], index=0)
    st.markdown("---")
    st.caption("Set GOOGLE_API_KEY in env or st.secrets. Set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON for Speech-to-Text.")

model = init_gemini()

# Session state for roleplay
if "rp_step" not in st.session_state:
    st.session_state.rp_step = 0
if "rp_key" not in st.session_state:
    st.session_state.rp_key = "school"

recorded_bytes = None

# Microphone capture
if HAS_MIC:
    st.subheader("🎤 Press and speak")
    mic = mic_recorder(start_prompt="Tap to Speak", stop_prompt="Stop", key=f"mic_{mode}")
    if mic and isinstance(mic, dict) and mic.get("bytes"):
        recorded_bytes = mic["bytes"]
        st.audio(recorded_bytes, format="audio/wav")
else:
    st.info("Microphone component not available. Use the uploader below.")

# Fallback uploader
uploaded = st.file_uploader("Or upload a short WAV/MP3/M4A", type=["wav", "mp3", "m4a"])
if uploaded is not None and not recorded_bytes:
    recorded_bytes = uploaded.read()
    st.audio(recorded_bytes)

# Main logic
if mode == "Free Chat":
    st.subheader("Ask Genie anything!")
    if recorded_bytes:
        with st.spinner("Transcribing with Whisper..."):
            user_text = transcribe_whisper(recorded_bytes, lang_code="en")
        st.markdown(f"**You said:** {user_text}")
        st.write(f"Your speech: {emoji_feedback(user_text)}")

        with st.spinner("Thinking with Gemini..."):
            reply = gemini_reply(model, user_text, persona="tutor")
        st.markdown(f"**Genie:** {reply}")
        st.write(f"Genie feedback: {emoji_feedback(reply)}")

        final_text = maybe_translate(reply, playback_lang)
        with st.spinner("Speaking..."):
            mp3 = speak_gtts(final_text, playback_lang)
        st.audio(mp3, format="audio/mp3")
    else:
        st.caption("Tap the mic or upload audio to start.")

elif mode == "Roleplay":
    keys = list(ROLEPLAY_SCENARIOS.keys())
    current_key = st.selectbox(
        "Pick a scenario",
        options=keys,
        index=keys.index(st.session_state.rp_key) if st.session_state.rp_key in keys else 0,
    )
    if current_key != st.session_state.rp_key:
        st.session_state.rp_key = current_key
        st.session_state.rp_step = 0

    scenario = ROLEPLAY_SCENARIOS[current_key]
    steps = scenario["steps"]
    i = st.session_state.rp_step

    st.subheader(scenario["title"])
    if i < len(steps):
        node = steps[i]
        ai_line = node["ai"]
        hints: List[str] = node.get("hints", [])

        st.markdown(f"**AI:** {ai_line}")
        if hints:
            st.caption("Try saying: " + ", ".join(hints))

        if st.button("🔊 Play Prompt"):
            prompt_text = maybe_translate(ai_line, playback_lang)
            mp3 = speak_gtts(prompt_text, playback_lang)
            st.audio(mp3, format="audio/mp3")


        if recorded_bytes:
            with st.spinner("Transcribing with Whisper..."):
                child_text = transcribe_whisper(recorded_bytes, lang_code="en")
            st.markdown(f"**You said:** {child_text}")
            st.write(emoji_feedback(child_text))

            # Simple correctness: contains any hint phrase
            ok = any(h.lower() in child_text.lower() for h in hints) if child_text else False
            if ok:
                st.success("Great answer! ✅")
                st.session_state.rp_step += 1
                if st.session_state.rp_step >= len(steps):
                    st.balloons()
                    done_text = maybe_translate("Awesome job! You finished the roleplay!", playback_lang)
                    mp3 = speak_gtts(done_text, playback_lang)
                    st.audio(mp3, format="audio/mp3")
                    st.session_state.rp_step = 0
            else:
                st.warning("Almost there! Try one of the hints.")
                if st.toggle("Need a gentle hint?", value=False):
                    nudge_prompt = (
                        f"The child answered: '{child_text}'. Original target: '{ai_line}'. "
                        f"Give a very short, friendly nudge (<= 1 sentence) helping them say one of: {hints}."
                    )
                    hint = gemini_reply(model, nudge_prompt, persona="roleplay")
                    st.info(hint)
                    mp3 = speak_gtts(maybe_translate(hint, playback_lang), playback_lang)
                    st.audio(mp3, format="audio/mp3")
    else:
        st.balloons()
        st.write("🎉 Roleplay complete!")

st.markdown("---")
st.caption(
    "reserved-for-caption"
)
