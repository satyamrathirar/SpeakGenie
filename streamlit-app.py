import os
import io
import json
import sys
import tempfile
import time
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

# Initialize session state variables FIRST
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "stop_response" not in st.session_state:
    st.session_state.stop_response = False
if "last_audio_processed" not in st.session_state:
    st.session_state.last_audio_processed = None
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}
if "chat_just_cleared" not in st.session_state:
    st.session_state.chat_just_cleared = False
if "rp_step" not in st.session_state:
    st.session_state.rp_step = 0
if "rp_key" not in st.session_state:
    st.session_state.rp_key = "school"

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

    # Check if response was stopped
    if st.session_state.stop_response:
        st.session_state.stop_response = False
        return "(Response stopped by user)"

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
        # Check again after generation in case user clicked stop during generation
        if st.session_state.stop_response:
            st.session_state.stop_response = False
            return "(Response stopped by user)"
            
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


def speak_gtts(text: str, target_lang: str) -> bytes:
    lang = LANG_CODE.get(target_lang, "en")
    if not text or not text.strip():
        text = "I didn't hear anything. Please try again."
    
    # Create a cache key for this text/language combination
    cache_key = f"{text}_{lang}"
    
    # Check if we already have this audio cached
    if cache_key in st.session_state.audio_cache:
        return st.session_state.audio_cache[cache_key]
    
    try:
        # Use faster TTS settings
        tts = gTTS(text=text, lang=lang, slow=False, tld="com")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_bytes = buf.getvalue()
        
        # Cache the audio for future use
        if audio_bytes:
            st.session_state.audio_cache[cache_key] = audio_bytes
        
        return audio_bytes
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return b""


def play_audio_js(audio_bytes: bytes):
    """Play audio using JavaScript to bypass browser autoplay restrictions"""
    import base64
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio id="audio_player" style="display:none;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    <script>
        var audio = document.getElementById('audio_player');
        audio.play().catch(function(error) {{
            console.log('Autoplay failed:', error);
        }});
    </script>
    """
    st.components.v1.html(audio_html, height=0)


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

model = init_gemini()

recorded_bytes = None

# Main logic
if mode == "Free Chat":
    st.subheader("Ask Genie anything!")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
                    st.caption(f"Speech: {emoji_feedback(message['content'])}")
            else:  # genie
                with st.chat_message("assistant", avatar="🧞‍♂️"):
                    st.write(message["content"])
                    st.caption(f"Genie feedback: {emoji_feedback(message['content'])}")
                    # Display cached audio if available
                    if "audio" in message and message["audio"]:
                        st.audio(message["audio"], format="audio/mp3")
                        # Auto-play the audio using JavaScript
                        play_audio_js(message["audio"])
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.last_audio_processed = None  # Reset audio processing
            st.session_state.chat_just_cleared = True  # Mark that chat was just cleared
            st.rerun()
    
    # Add stop button in sidebar for emergency stop
    with st.sidebar:
        st.markdown("---")
        if st.button("🛑 Emergency Stop", help="Stop current AI response", type="secondary"):
            st.session_state.stop_response = True
            st.warning("AI response stopped!")
        
        # Audio cache management
        cache_size = len(st.session_state.audio_cache)
        st.caption(f"Audio cache: {cache_size} items")
        if cache_size > 0:
            if st.button("🗑️ Clear Audio Cache", help="Clear cached audio files"):
                st.session_state.audio_cache = {}
                st.success("Audio cache cleared!")
    
    # Text input chat box
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Type your question here:", placeholder="Ask Genie anything...")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            # File upload
            uploaded = st.file_uploader("📎 Upload", type=["wav", "mp3", "m4a"], key=f"upload_{mode}", label_visibility="collapsed")
            if uploaded is not None and not recorded_bytes:
                recorded_bytes = uploaded.read()
                st.audio(recorded_bytes)
            
        with col2:
            # Microphone capture
            if HAS_MIC:
                mic = mic_recorder(start_prompt="🎤 Speak", stop_prompt="Stop", key=f"mic_{mode}")
                if mic and isinstance(mic, dict) and mic.get("bytes"):
                    recorded_bytes = mic["bytes"]
                    st.audio(recorded_bytes, format="audio/wav")
            else:
                st.caption("Mic unavailable")
        with col3:
            submit_text = st.form_submit_button("Send 📤")
    
    # Show input method information
    st.caption("💬 Type, 🎤 speak, or 📎 upload audio to chat with Genie!")
    st.caption("📎 Supported audio formats: WAV, MP3, M4A")
    
    # Handle text input
    if submit_text and user_input.strip():
        # Reset stop flag
        st.session_state.stop_response = False
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

        # Generate response
        with st.spinner("🤔 Thinking with Gemini..."):
            reply = gemini_reply(model, user_input.strip(), persona="tutor")
        
        # Add Genie response to chat history only if not stopped
        if reply != "(Response stopped by user)":
            # Play audio
            final_text = maybe_translate(reply, playback_lang)
            
            # Quick check if audio is cached
            cache_key = f"{final_text}_{LANG_CODE.get(playback_lang, 'en')}"
            if cache_key in st.session_state.audio_cache:
                # Use cached audio immediately
                audio_bytes = st.session_state.audio_cache[cache_key]
                st.audio(audio_bytes, format="audio/mp3")
                play_audio_js(audio_bytes)  # Auto-play using JavaScript
                st.session_state.chat_history.append({
                    "role": "genie", 
                    "content": reply, 
                    "audio": audio_bytes
                })
            else:
                # Generate new audio with timeout
                with st.spinner("🔊 Generating voice..."):
                    try:
                        audio_bytes = speak_gtts(final_text, playback_lang)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                            play_audio_js(audio_bytes)  # Auto-play using JavaScript
                            # Store message with audio
                            st.session_state.chat_history.append({
                                "role": "genie", 
                                "content": reply, 
                                "audio": audio_bytes
                            })
                        else:
                            # Store message without audio if generation failed
                            st.session_state.chat_history.append({
                                "role": "genie", 
                                "content": reply
                            })
                            st.warning("Voice generation failed, but response saved.")
                    except Exception as e:
                        st.error(f"Audio generation failed: {e}")
                        # Store message without audio
                        st.session_state.chat_history.append({
                            "role": "genie", 
                            "content": reply
                        })
        
        # Rerun to refresh chat display
        st.rerun()
    
    if recorded_bytes:
        # Check if chat was just cleared - if so, ignore this audio processing
        if st.session_state.chat_just_cleared:
            st.session_state.chat_just_cleared = False  # Reset the flag
            st.caption("Chat cleared - audio input ignored.")
        else:
            # Create a unique identifier for this audio to prevent duplicate processing
            audio_id = hash(recorded_bytes)
            
            # Only process if this is new audio
            if st.session_state.last_audio_processed != audio_id:
                st.session_state.last_audio_processed = audio_id
                st.session_state.stop_response = False  # Reset stop flag
                
                with st.spinner("🎙️ Transcribing with Whisper..."):
                    user_text = transcribe_whisper(recorded_bytes, lang_code="en")
                
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_text})

                # Generate response
                with st.spinner("🤔 Thinking with Gemini..."):
                    reply = gemini_reply(model, user_text, persona="tutor")
                
                # Add Genie response to chat history only if not stopped
                if reply != "(Response stopped by user)":
                    # Play audio
                    final_text = maybe_translate(reply, playback_lang)
                    
                    # Quick check if audio is cached
                    cache_key = f"{final_text}_{LANG_CODE.get(playback_lang, 'en')}"
                    if cache_key in st.session_state.audio_cache:
                        # Use cached audio immediately
                        audio_bytes = st.session_state.audio_cache[cache_key]
                        st.audio(audio_bytes, format="audio/mp3")
                        play_audio_js(audio_bytes)  # Auto-play using JavaScript
                        st.session_state.chat_history.append({
                            "role": "genie", 
                            "content": reply, 
                            "audio": audio_bytes
                        })
                    else:
                        # Generate new audio with timeout
                        with st.spinner("🔊 Generating voice..."):
                            try:
                                audio_bytes = speak_gtts(final_text, playback_lang)
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                                    play_audio_js(audio_bytes)  # Auto-play using JavaScript
                                    # Store message with audio
                                    st.session_state.chat_history.append({
                                        "role": "genie", 
                                        "content": reply, 
                                        "audio": audio_bytes
                                    })
                                else:
                                    # Store message without audio if generation failed
                                    st.session_state.chat_history.append({
                                        "role": "genie", 
                                        "content": reply
                                    })
                                    st.warning("Voice generation failed, but response saved.")
                            except Exception as e:
                                st.error(f"Audio generation failed: {e}")
                                # Store message without audio
                                st.session_state.chat_history.append({
                                    "role": "genie", 
                                    "content": reply
                                })
                
                # Rerun to refresh chat display
                st.rerun()
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
            audio_bytes = speak_gtts(prompt_text, playback_lang)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
                play_audio_js(audio_bytes)  # Auto-play using JavaScript

        # Voice input section for roleplay
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.caption("Respond:")
        with col2:
            # Microphone capture
            if HAS_MIC:
                mic = mic_recorder(start_prompt="🎤 Speak", stop_prompt="Stop", key=f"rp_mic_{mode}_{i}")
                if mic and isinstance(mic, dict) and mic.get("bytes"):
                    recorded_bytes = mic["bytes"]
                    st.audio(recorded_bytes, format="audio/wav")
            else:
                st.caption("Mic unavailable")
        with col3:
            # File upload for roleplay
            uploaded = st.file_uploader("📎 Upload", type=["wav", "mp3", "m4a"], key=f"rp_upload_{mode}_{i}", label_visibility="collapsed")
            if uploaded is not None and not recorded_bytes:
                recorded_bytes = uploaded.read()
                st.audio(recorded_bytes)

        # Show roleplay input information
        st.caption("🎤 Speak or 📎 upload audio to respond")
        st.caption("📎 Supported audio formats: WAV, MP3, M4A")


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
                    audio_bytes = speak_gtts(done_text, playback_lang)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                        play_audio_js(audio_bytes)  # Auto-play using JavaScript
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
                    audio_bytes = speak_gtts(maybe_translate(hint, playback_lang), playback_lang)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")
                        play_audio_js(audio_bytes)  # Auto-play using JavaScript
    else:
        st.balloons()
        st.write("🎉 Roleplay complete!")

st.markdown("---")
st.caption(
    "reserved-for-caption"
)
