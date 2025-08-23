import os
import io
import json
import time
import tempfile
from typing import Dict, List
from openai import OpenAI

import streamlit as st

# --- Optional mic recorder component (nice UX). Fallback to uploader if unavailable.
try:
    from streamlit_mic_recorder import mic_recorder  # pip install streamlit-mic-recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False

# OpenAI SDK v1
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Text-to-Speech (simple + free): gTTS
from gtts import gTTS  # pip install gTTS

# Optional translation for native-language playback
try:
    from googletrans import Translator  # pip install googletrans==4.0.0-rc1
    translator = Translator()
    HAS_TRANSLATE = True
except Exception:
    HAS_TRANSLATE = False

# ---------------------------
# Configuration
# ---------------------------
APP_TITLE = "🎙️ Genie AI Voice Tutor"
DEFAULT_MODEL = "gpt-4o-mini"
TRANSCRIBE_MODEL = "whisper-1"  # or "gpt-4o-transcribe" if available to your account
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

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_openai_client():
    """Initialize OpenAI client using env var or Streamlit secrets."""
    if not OPENAI_AVAILABLE:
        return None
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def safe_tts(text: str, target_lang: str) -> io.BytesIO:
    """Convert text to speech using gTTS; returns BytesIO (mp3)."""
    lang_code = LANG_CODE.get(target_lang, "en")
    # gTTS may fail on empty/very short text
    if not text or not text.strip():
        text = "I didn't hear anything. Please try again."
    tts = gTTS(text=text, lang=lang_code)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf


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


def emoji_feedback(text: str) -> str:
    """Tiny heuristic for fun feedback based on length and punctuation."""
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


def transcribe_audio(client: OpenAI, audio_bytes: bytes) -> str:
    """Send audio to Whisper for transcription."""
    if not client:
        return "(Transcription unavailable: missing OpenAI key)"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=f)
        # result.text is typical; some SDKs use result.segments, etc.
        text = getattr(result, "text", None)
        if not text:
            # Fallback for any SDK variation
            text = str(result)
        return text
    except Exception as e:
        return f"(Transcription error: {e})"


def gpt_reply(client: OpenAI, user_text: str, persona: str = "tutor") -> str:
    if not client:
        return "(AI unavailable: missing OpenAI key)"
    if persona == "roleplay":
        sys = (
            "You are part of a child-friendly roleplay. Respond in simple, short sentences (<= 2) "
            "and keep a warm, encouraging tone. Avoid personal data, and keep topics school-safe."
        )
    else:
        sys = (
            "You are Genie, a friendly tutor for kids aged 6-12. Explain simply using short sentences (<= 2). "
            "End with one gentle follow-up question to check understanding. Avoid personal data and unsafe topics."
        )
    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user_text},
            ],
            temperature=0.6,
            max_tokens=200,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(Model error: {e})"


def roleplay_next(step_text: str, hints: List[str]) -> bool:
    """Very simple correctness check: contains any hint phrase (case-insensitive)."""
    if not step_text:
        return False
    t = step_text.lower().strip()
    for h in hints:
        if h.lower() in t:
            return True
    return False


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Genie Voice Tutor", page_icon="🎙️", layout="centered")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Playback Language", SUPPORTED_LANGS, index=0)
    mode = st.radio("Mode", ["Free Chat", "Roleplay"], index=0)
    st.markdown("---")
    st.caption("Tip: If the mic button doesn't appear, use the audio file uploader below.")

client = get_openai_client()

# Session state for roleplay
if "rp_step" not in st.session_state:
    st.session_state.rp_step = 0
if "rp_key" not in st.session_state:
    st.session_state.rp_key = "school"

# Common widgets
recorded_bytes = None

# Microphone capture
if HAS_MIC:
    st.subheader("🎤 Press and speak")
    mic_data = mic_recorder(start_prompt="Tap to Speak", stop_prompt="Stop", key=f"mic_{mode}")
    if mic_data and isinstance(mic_data, dict) and mic_data.get("bytes"):
        recorded_bytes = mic_data["bytes"]
        st.audio(recorded_bytes, format="audio/wav")
else:
    st.info("Microphone component not available. Use the uploader below.")

# Fallback: file uploader
uploaded = st.file_uploader("Or upload a short WAV/MP3/M4A", type=["wav", "mp3", "m4a"])
if uploaded is not None and not recorded_bytes:
    recorded_bytes = uploaded.read()
    st.audio(recorded_bytes)

# Main logic
if mode == "Free Chat":
    st.subheader("Ask Genie anything!")

    if recorded_bytes:
        with st.spinner("Transcribing..."):
            user_text = transcribe_audio(client, recorded_bytes)
        st.markdown(f"**You said:** {user_text}")
        fb_user = emoji_feedback(user_text)
        st.write(f"Your speech: {fb_user}")

        with st.spinner("Thinking..."):
            reply = gpt_reply(client, user_text, persona="tutor")
        st.markdown(f"**Genie:** {reply}")
        fb_ai = emoji_feedback(reply)
        st.write(f"Genie feedback: {fb_ai}")

        # Native language playback (optional translate)
        final_text = maybe_translate(reply, language)
        with st.spinner("Speaking..."):
            mp3_buf = safe_tts(final_text, language)
        st.audio(mp3_buf, format="audio/mp3")

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
    step_i = st.session_state.rp_step

    st.subheader(scenario["title"])

    if step_i < len(steps):
        node = steps[step_i]
        ai_line = node["ai"]
        hints = node.get("hints", [])

        st.markdown(f"**AI:** {ai_line}")
        if hints:
            st.caption("Try saying: " + ", ".join(hints))

        # Speak AI line (in selected language)
        if st.button("🔊 Play Prompt"):
            prompt_text = maybe_translate(ai_line, language)
            mp3_buf = safe_tts(prompt_text, language)
            st.audio(mp3_buf, format="audio/mp3")

        if recorded_bytes:
            with st.spinner("Transcribing..."):
                child_text = transcribe_audio(client, recorded_bytes)
            st.markdown(f"**You said:** {child_text}")
            st.write(emoji_feedback(child_text))

            if roleplay_next(child_text, hints):
                st.success("Great answer! ✅")
                st.session_state.rp_step += 1
                if st.session_state.rp_step >= len(steps):
                    st.balloons()
                    congrats = maybe_translate("Awesome job! You finished the roleplay!", language)
                    mp3_buf = safe_tts(congrats, language)
                    st.audio(mp3_buf, format="audio/mp3")
                    st.session_state.rp_step = 0
            else:
                st.warning("Almost there! Try one of the hints.")
                # Offer a gentle, paraphrased re-prompt via LLM (optional)
                if client and st.toggle("Need a gentle hint?", value=False):
                    hint_prompt = (
                        f"The child answered: '{child_text}'. Original target: '{ai_line}'. "
                        f"Give a short, friendly nudge (<= 1 sentence) helping them say one of: {hints}."
                    )
                    hint = gpt_reply(client, hint_prompt, persona="roleplay")
                    st.info(hint)
                    # Speak the hint
                    hint_text = maybe_translate(hint, language)
                    mp3_buf = safe_tts(hint_text, language)
                    st.audio(mp3_buf, format="audio/mp3")
    else:
        st.balloons()
        st.write("🎉 Roleplay complete!")

# Footer / Setup help
st.markdown("---")
st.caption(
    "Setup: set OPENAI_API_KEY as an environment variable or in st.secrets. "
    "Install: pip install streamlit openai gTTS googletrans==4.0.0-rc1 streamlit-mic-recorder"
)

