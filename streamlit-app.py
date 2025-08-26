import os
import io
import time
import tempfile
from typing import List
from dotenv import load_dotenv
import streamlit as st


try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC = True
except Exception:
    HAS_MIC = False


import google.generativeai as genai


import whisper
from pydub import AudioSegment


from gtts import gTTS

try:
    from googletrans import Translator
    translator = Translator()
    HAS_TRANSLATE = True
except Exception:
    HAS_TRANSLATE = False


# App Configs
st.set_page_config(page_title="AI Voice Tutor", page_icon="🎙️", layout="centered")
APP_TITLE = "🎙️ SpeakGenie: AI Voice Tutor"

# Initialize session state variables
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
if "rp_questions" not in st.session_state:
    st.session_state.rp_questions = []
if "rp_current_question" not in st.session_state:
    st.session_state.rp_current_question = ""
if "rp_conversation_history" not in st.session_state:
    st.session_state.rp_conversation_history = []
if "rp_stop_requested" not in st.session_state:
    st.session_state.rp_stop_requested = False
if "rp_conversation_turns" not in st.session_state:
    st.session_state.rp_conversation_turns = []
if "rp_waiting_for_response" not in st.session_state:
    st.session_state.rp_waiting_for_response = False

# Models / languages
load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # default is 1.5 flash, if not specified in .env
SUPPORTED_LANGS = ["English", "Hindi", "Marathi", "Tamil"]
LANG_CODE = {"English": "en", "Hindi": "hi", "Marathi": "mr", "Tamil": "ta"}

# Roleplay scenarios (simplified structure for dynamic questions)
ROLEPLAY_SCENARIOS = {
    "school": {
        "title": "🏫 At School",
        "description": "Practice conversations you might have at school with teachers and classmates",
        "context": "You are at school. You will have conversations with teachers, classmates, and other school staff about school activities, subjects, and daily routines."
    },
    "store": {
        "title": "🛒 At the Store", 
        "description": "Learn to shop and talk to store workers when buying things",
        "context": "You are at a store with your family. You will practice talking to store workers, asking for items, and learning about shopping."
    },
    "home": {
        "title": "👨‍👩‍👧 At Home",
        "description": "Practice family conversations and talking about home activities", 
        "context": "You are at home with your family. You will have conversations about daily activities, helping at home, and family time."
    }
}

# Setup Helpers

@st.cache_resource(show_spinner=False)

def init_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        return model
    except Exception:
        return None


def generate_roleplay_questions(model, scenario_key: str, scenario_context: str) -> List[str]:
    """Generate 10 dynamic questions for a roleplay scenario using Gemini"""
    if not model:
        # Fallback questions if AI is unavailable
        fallback_questions = {
            "school": [
                "Good morning! What's your name?",
                "What grade are you in?", 
                "What's your favorite subject?",
                "Do you like your teacher?",
                "What do you do during recess?",
                "Do you have homework today?",
                "What's your favorite school activity?",
                "Do you eat lunch at school?",
                "What do you learn in math class?",
                "Do you have any friends at school?"
            ],
            "store": [
                "Welcome! What would you like to buy?",
                "How many do you need?",
                "Do you have money to pay?",
                "Would you like a bag?",
                "Is this for you or your family?",
                "Do you need anything else?",
                "Have you been to this store before?",
                "Do you like shopping?",
                "What's your favorite thing to buy?",
                "Thank you for shopping! Have a nice day!"
            ],
            "home": [
                "Who do you live with at home?",
                "Do you help with chores?",
                "What's your favorite room in the house?",
                "What do you like to do at home?",
                "Do you have pets?",
                "What time do you go to bed?",
                "What's your favorite meal?",
                "Do you have your own room?",
                "What games do you play at home?",
                "Do you like spending time with your family?"
            ]
        }
        return fallback_questions.get(scenario_key, fallback_questions["school"])
    
    prompt = f"""
    Generate exactly 10 simple, child-friendly questions for a roleplay scenario about "{scenario_context}".
    
    Requirements:
    - Questions should be appropriate for children aged 6-16
    - Use simple vocabulary and short sentences
    - Questions should flow naturally in conversation
    - at school, questions should be from the perspective of a teacher, at the store, it should be from the perspective of a cashier and at home, it should be from the perspective of a parent or guardian
    - Avoid using slang or overly complex language
    - Avoid asking for personal information like full names, addresses, or phone numbers
    - Make questions engaging and fun
    - Start with a greeting question
    - End with a positive closing question
    
    Format: Return only the questions, one per line, numbered 1-10.
    
    Example for school scenario:
    1. Good morning! What's your name?
    2. What grade are you in?
    3. What's your favorite subject?
    
    Now generate for: {scenario_context}
    """
    
    try:
        response = model.generate_content(prompt)
        if hasattr(response, 'text') and response.text:
            # Parse the response to extract questions
            lines = response.text.strip().split('\n')
            questions = []
            for line in lines:
                # Remove numbering and clean up
                cleaned = line.strip()
                if cleaned and any(char.isalpha() for char in cleaned):
                    # Remove numbers and dots from start
                    import re
                    cleaned = re.sub(r'^\d+\.?\s*', '', cleaned)
                    if cleaned:
                        questions.append(cleaned)
            
            # Ensure we have exactly 10 questions
            if len(questions) >= 10:
                return questions[:10]
            else:
                # If not enough questions, use fallback
                fallback = {
                    "school": ["Good morning! What's your name?", "What grade are you in?", "What's your favorite subject?", "Do you like your teacher?", "What do you do during recess?", "Do you have homework today?", "What's your favorite school activity?", "Do you eat lunch at school?", "What do you learn in math class?", "Do you have any friends at school?"],
                    "store": ["Welcome! What would you like to buy?", "How many do you need?", "Do you have money to pay?", "Would you like a bag?", "Is this for you or your family?", "Do you need anything else?", "Have you been to this store before?", "Do you like shopping?", "What's your favorite thing to buy?", "Thank you for shopping! Have a nice day!"],
                    "home": ["Who do you live with at home?", "Do you help with chores?", "What's your favorite room in the house?", "What do you like to do at home?", "Do you have pets?", "What time do you go to bed?", "What's your favorite meal?", "Do you have your own room?", "What games do you play at home?", "Do you like spending time with your family?"]
                }
                return fallback.get(scenario_key, fallback["school"])
        else:
            # Fallback if response is empty
            fallback = {
                "school": ["Good morning! What's your name?", "What grade are you in?", "What's your favorite subject?", "Do you like your teacher?", "What do you do during recess?", "Do you have homework today?", "What's your favorite school activity?", "Do you eat lunch at school?", "What do you learn in math class?", "Do you have any friends at school?"],
                "store": ["Welcome! What would you like to buy?", "How many do you need?", "Do you have money to pay?", "Would you like a bag?", "Is this for you or your family?", "Do you need anything else?", "Have you been to this store before?", "Do you like shopping?", "What's your favorite thing to buy?", "Thank you for shopping! Have a nice day!"],
                "home": ["Who do you live with at home?", "Do you help with chores?", "What's your favorite room in the house?", "What do you like to do at home?", "Do you have pets?", "What time do you go to bed?", "What's your favorite meal?", "Do you have your own room?", "What games do you play at home?", "Do you like spending time with your family?"]
            }
            return fallback.get(scenario_key, fallback["school"])
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        # Return fallback questions
        fallback = {
            "school": ["Good morning! What's your name?", "What grade are you in?", "What's your favorite subject?", "Do you like your teacher?", "What do you do during recess?", "Do you have homework today?", "What's your favorite school activity?", "Do you eat lunch at school?", "What do you learn in math class?", "Do you have any friends at school?"],
            "store": ["Welcome! What would you like to buy?", "How many do you need?", "Do you have money to pay?", "Would you like a bag?", "Is this for you or your family?", "Do you need anything else?", "Have you been to this store before?", "Do you like shopping?", "What's your favorite thing to buy?", "Thank you for shopping! Have a nice day!"],
            "home": ["Who do you live with at home?", "Do you help with chores?", "What's your favorite room in the house?", "What do you like to do at home?", "Do you have pets?", "What time do you go to bed?", "What's your favorite meal?", "Do you have your own room?", "What games do you play at home?", "Do you like spending time with your family?"]
        }
        return fallback.get(scenario_key, fallback["school"])


# Whisper Transcription Helper
def transcribe_whisper(audio_bytes: bytes, lang_code: str = "en") -> str:
    """Transcribe audio using OpenAI Whisper. Accepts bytes, returns text."""
    # Save to temp WAV file
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            audio.export(tmp.name, format="wav")
            model = whisper.load_model("base")  # available models: "tiny", "base", "small", "medium", "large"
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

# UI
st.title(APP_TITLE)
with st.sidebar:
    st.header("Settings")
    playback_lang = st.selectbox("Playback Language", SUPPORTED_LANGS, index=0)
    mode = st.radio("Mode", ["Free Chat", "Roleplay"], index=0)

model = init_gemini()

recorded_bytes = None

# Main logic
if mode == "Free Chat":
    st.subheader("I will solve all your doubts!")
    
    status_container = st.empty()

    # Chat history
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
                        # Auto-play the most recent message
                        if i == len(st.session_state.chat_history) - 1:
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
            if uploaded is not None:
                recorded_bytes = uploaded.read()
            
        with col2:
            # Microphone capture
            if HAS_MIC:
                mic = mic_recorder(start_prompt="🎤 Speak", stop_prompt="Stop", key=f"mic_{mode}")
                if mic and isinstance(mic, dict) and mic.get("bytes"):
                    recorded_bytes = mic["bytes"]
            else:
                st.caption("Mic unavailable")
        with col3:
            submit_text = st.form_submit_button("Send 📤")
    
    # Show input method information
    st.caption("💬 Type, 🎤 speak, or 📎 upload audio to chat with Genie!")
    st.caption("📎 Supported audio formats: WAV, MP3, M4A")
    st.caption("🎙️ After recording with microphone, click Send to process your audio")
    
    # Handle both text input and audio input
    if submit_text and (user_input.strip() or recorded_bytes):
        # Reset stop flag
        st.session_state.stop_response = False
        
        # Process audio input if available
        if recorded_bytes and not user_input.strip():
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
                    
                    with status_container:
                        with st.spinner("🎙️ Transcribing with Whisper..."):
                            user_text = transcribe_whisper(recorded_bytes, lang_code="en")
                    
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_text})

                    # Generate response
                    with status_container:
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
                            st.session_state.chat_history.append({
                                "role": "genie", 
                                "content": reply, 
                                "audio": audio_bytes
                            })
                        else:
                            # Generate new audio with timeout
                            with status_container:
                                with st.spinner("🔊 Generating voice..."):
                                    try:
                                        audio_bytes = speak_gtts(final_text, playback_lang)
                                        if audio_bytes:
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
                    
                    # Clear status and rerun to refresh chat display
                    status_container.empty()
                    st.rerun()
        
        # Process text input if available
        elif user_input.strip():
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

            # Generate response
            with status_container:
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
                    st.session_state.chat_history.append({
                        "role": "genie", 
                        "content": reply, 
                        "audio": audio_bytes
                    })
                else:
                    # Generate new audio with timeout
                    with status_container:
                        with st.spinner("🔊 Generating voice..."):
                            try:
                                audio_bytes = speak_gtts(final_text, playback_lang)
                                if audio_bytes:
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
            
            # Clear status and rerun to refresh chat display
            status_container.empty()
            st.rerun()
    
    if recorded_bytes:
        st.caption("Audio recorded - click Send to process")
    else:
        st.caption("Tap the mic or upload audio to start.")

elif mode == "Roleplay":
    st.subheader("Let's imagine a scenario!")
    st.caption("Students practice speaking through guided, interactive scenarios. These help build confidence and everyday vocabulary.")
    
    # Emergency stop button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🛑 Stop Roleplay", help="Emergency stop roleplay", type="secondary"):
            st.session_state.rp_stop_requested = True
            st.session_state.rp_step = 0
            st.session_state.rp_questions = []
            st.session_state.rp_current_question = ""
            st.session_state.rp_conversation_history = []
            st.session_state.rp_conversation_turns = []
            st.session_state.rp_waiting_for_response = False
            st.warning("Roleplay stopped!")
            st.rerun()
    
    # Scenario selection
    keys = list(ROLEPLAY_SCENARIOS.keys())
    current_key = st.selectbox(
        "Pick a scenario:",
        options=keys,
        format_func=lambda x: ROLEPLAY_SCENARIOS[x]["title"],
        index=keys.index(st.session_state.rp_key) if st.session_state.rp_key in keys else 0,
    )
    
    # Reset roleplay if scenario changed
    if current_key != st.session_state.rp_key:
        st.session_state.rp_key = current_key
        st.session_state.rp_step = 0
        st.session_state.rp_questions = []
        st.session_state.rp_current_question = ""
        st.session_state.rp_conversation_history = []
        st.session_state.rp_stop_requested = False
        st.session_state.rp_conversation_turns = []
        st.session_state.rp_waiting_for_response = False

    scenario = ROLEPLAY_SCENARIOS[current_key]
    
    # Show scenario description
    st.markdown(f"### {scenario['title']}")
    st.caption(scenario['description'])
    
    # Generate questions if not already generated
    if not st.session_state.rp_questions and not st.session_state.rp_stop_requested:
        with st.spinner("🤖 Generating roleplay questions..."):
            st.session_state.rp_questions = generate_roleplay_questions(model, current_key, scenario['context'])
            st.session_state.rp_current_question = st.session_state.rp_questions[0] if st.session_state.rp_questions else ""
    
    # Progress indicator
    if st.session_state.rp_questions:
        progress = st.session_state.rp_step / 10
        st.progress(progress, text=f"Question {st.session_state.rp_step + 1} of 10")
    
    # Main roleplay interaction
    if st.session_state.rp_questions and st.session_state.rp_step < 10 and not st.session_state.rp_stop_requested:
        current_question = st.session_state.rp_questions[st.session_state.rp_step]
        
        # Initialize conversation turns for this question if not exists
        if not st.session_state.rp_conversation_turns:
            st.session_state.rp_conversation_turns = [{"role": "ai", "content": current_question}]
        
        # Display conversation history for current question
        st.markdown("### Conversation:")
        for i, turn in enumerate(st.session_state.rp_conversation_turns):
            if turn["role"] == "ai":
                st.markdown(f"**AI:** {turn['content']}")
                # Add play button for AI messages
                if st.button(f"🔊 Play", key=f"play_turn_{st.session_state.rp_step}_{i}"):
                    prompt_text = maybe_translate(turn['content'], playback_lang)
                    audio_bytes = speak_gtts(prompt_text, playback_lang)
                    if audio_bytes:
                        play_audio_js(audio_bytes)
            else:  # user
                st.markdown(f"**You:** {turn['content']}")
                st.caption(f"Speech quality: {emoji_feedback(turn['content'])}")
        
        # Status container for processing indicators
        status_container = st.empty()
        
        # Voice input section
        st.markdown("**Your turn to speak:**")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        recorded_bytes = None
        
        with col1:
            # Microphone capture
            if HAS_MIC:
                mic = mic_recorder(
                    start_prompt="🎤 Record", 
                    stop_prompt="Stop", 
                    key=f"rp_mic_{current_key}_{st.session_state.rp_step}_{len(st.session_state.rp_conversation_turns)}"
                )
                if mic and isinstance(mic, dict) and mic.get("bytes"):
                    recorded_bytes = mic["bytes"]
            else:
                st.caption("Mic unavailable")
        
        with col2:
            # File upload
            uploaded = st.file_uploader(
                "📎 Upload", 
                type=["wav", "mp3", "m4a"], 
                key=f"rp_upload_{current_key}_{st.session_state.rp_step}_{len(st.session_state.rp_conversation_turns)}",
                label_visibility="collapsed"
            )
            if uploaded is not None:
                recorded_bytes = uploaded.read()
        
        with col3:
            # Next Question button
            if st.button("➡️ Next Question", key=f"next_q_{st.session_state.rp_step}", type="primary"):
                # Save current conversation to history
                if st.session_state.rp_conversation_turns:
                    st.session_state.rp_conversation_history.append({
                        "question_number": st.session_state.rp_step + 1,
                        "conversation": st.session_state.rp_conversation_turns.copy()
                    })
                
                # Move to next question
                st.session_state.rp_step += 1
                st.session_state.rp_conversation_turns = []  # Reset conversation for next question
                
                # Check if roleplay is complete
                if st.session_state.rp_step >= 10:
                    st.balloons()
                    completion_text = "Fantastic! You completed all 10 questions in this roleplay scenario! 🎉"
                    st.success(completion_text)
                    
                    # Play completion audio
                    final_audio = speak_gtts(maybe_translate(completion_text, playback_lang), playback_lang)
                    if final_audio:
                        play_audio_js(final_audio)
                    
                    # Reset for next roleplay
                    st.session_state.rp_step = 0
                    st.session_state.rp_questions = []
                    st.session_state.rp_current_question = ""
                    st.session_state.rp_conversation_history = []
                    st.session_state.rp_conversation_turns = []
                    st.session_state.rp_waiting_for_response = False
                
                st.rerun()
        
        with col4:
            # Skip button for difficult questions
            if st.button("⏭️ Skip Question", help="Skip this question if too difficult"):
                # Save current conversation to history even if skipped
                if st.session_state.rp_conversation_turns:
                    st.session_state.rp_conversation_history.append({
                        "question_number": st.session_state.rp_step + 1,
                        "conversation": st.session_state.rp_conversation_turns.copy(),
                        "skipped": True
                    })
                
                st.session_state.rp_step += 1
                st.session_state.rp_conversation_turns = []  # Reset conversation for next question
                
                if st.session_state.rp_step >= 10:
                    st.balloons()
                    completion_text = "Great job! You completed the roleplay scenario!"
                    st.success(completion_text)
                    
                    # Play completion audio
                    final_audio = speak_gtts(maybe_translate(completion_text, playback_lang), playback_lang)
                    if final_audio:
                        play_audio_js(final_audio)
                    
                    # Reset for next roleplay
                    st.session_state.rp_step = 0
                    st.session_state.rp_questions = []
                    st.session_state.rp_current_question = ""
                    st.session_state.rp_conversation_history = []
                    st.session_state.rp_conversation_turns = []
                    st.session_state.rp_waiting_for_response = False
                else:
                    st.info("Question skipped! Let's try the next one.")
                st.rerun()
        
        # Input guidance
        st.caption("🎤 Record your response, 📎 upload audio, ➡️ next question, or ⏭️ skip")
        st.caption("📎 Supported formats: WAV, MP3, M4A")
        
        # Process audio input
        if recorded_bytes:
            # Transcribe audio
            with status_container:
                with st.spinner("🎙️ Transcribing your response..."):
                    user_text = transcribe_whisper(recorded_bytes, lang_code="en")
            
            if user_text and user_text != "(No speech detected)":
                # Add user response to conversation turns
                st.session_state.rp_conversation_turns.append({
                    "role": "user",
                    "content": user_text
                })
                
                # Generate AI follow-up response
                with status_container:
                    with st.spinner("🤖 Generating AI response..."):
                        # Create context from conversation history
                        conversation_context = ""
                        for turn in st.session_state.rp_conversation_turns:
                            if turn["role"] == "ai":
                                conversation_context += f"AI: {turn['content']}\n"
                            else:
                                conversation_context += f"Student: {turn['content']}\n"
                        
                        follow_up_prompt = f"""
                        You are having a conversation with a child in a roleplay scenario. Here's the conversation so far:
                        
                        {conversation_context}
                        
                        Generate a natural follow-up response that:
                        - Acknowledges their answer positively
                        - Asks a related follow-up question to keep the conversation going
                        - Is warm, supportive, and encouraging
                        - Uses simple language appropriate for children
                        - Stays within the roleplay context
                        - Keeps the conversation natural and engaging
                        
                        If this feels like a natural place to end this part of the conversation, you can give a positive closing statement instead of asking another question.
                        
                        Respond in 1-2 sentences maximum.
                        """
                        
                        ai_response = gemini_reply(model, follow_up_prompt, persona="roleplay")
                
                # Add AI response to conversation turns
                if ai_response and ai_response != "(Response stopped by user)":
                    st.session_state.rp_conversation_turns.append({
                        "role": "ai",
                        "content": ai_response
                    })
                    
                    # Play AI response audio
                    ai_audio = speak_gtts(maybe_translate(ai_response, playback_lang), playback_lang)
                    if ai_audio:
                        play_audio_js(ai_audio)
                
                # Clear status and rerun to show updated conversation
                status_container.empty()
                st.rerun()
                
            else:
                st.warning("I couldn't hear you clearly. Please try recording again.")
    
    elif st.session_state.rp_step >= 10:
        # Roleplay completed
        st.success("🎉 Roleplay Complete!")
        st.markdown("### Great job finishing the roleplay!")
        
        # Show conversation summary
        if st.session_state.rp_conversation_history:
            with st.expander("📝 View Conversation Summary"):
                for item in st.session_state.rp_conversation_history:
                    st.markdown(f"### Question {item['question_number']}")
                    if item.get('skipped', False):
                        st.markdown("*(This question was skipped)*")
                    
                    # Display the full conversation for this question
                    for turn in item['conversation']:
                        if turn['role'] == 'ai':
                            st.markdown(f"**AI:** {turn['content']}")
                        else:
                            st.markdown(f"**You:** {turn['content']}")
                    st.markdown("---")
        
        # Restart button
        if st.button("🔄 Start New Roleplay"):
            st.session_state.rp_step = 0
            st.session_state.rp_questions = []
            st.session_state.rp_current_question = ""
            st.session_state.rp_conversation_history = []
            st.session_state.rp_stop_requested = False
            st.session_state.rp_conversation_turns = []
            st.session_state.rp_waiting_for_response = False
            st.rerun()
    
    elif st.session_state.rp_stop_requested:
        # Roleplay was stopped
        st.info("Roleplay was stopped. Choose a scenario above to start again!")
        if st.button("🔄 Reset"):
            st.session_state.rp_stop_requested = False
            st.session_state.rp_conversation_turns = []
            st.session_state.rp_waiting_for_response = False
            st.rerun()
    
    else:
        # Initial state - show example scenarios
        st.markdown("### Example Roleplays:")
        
        st.markdown("""
        **🏫 At School**  
        AI: "Good morning! What's your name?"  
        Student: "My name is Aarav."  
        AI: "Hi Aarav! Do you like school?"
        
        **🛒 At the Store**  
        AI: "Welcome! What do you want to buy today?"  
        Student: "I want a banana."  
        AI: "One banana coming right up!"
        
        **👨‍👩‍👧 At Home**  
        AI: "Who do you live with?"  
        Student: "I live with my parents."  
        AI: "Nice! Do you help them at home?"
        """)
        
        st.info("👆 Choose a scenario above to start your interactive roleplay with 10 questions!")

st.markdown("---")
st.caption(
    "reserved-for-caption"
)
