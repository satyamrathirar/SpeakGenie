# 🎙️ SpeakGenie: AI Voice Tutor

An interactive AI-powered voice tutoring application designed to help children (ages 6-16) practice speaking skills through conversational AI and roleplay scenarios. Built with Streamlit, OpenAI Whisper, and Google Gemini AI.



## ✨ Features

### 🗣️ Free Chat Mode
- **Voice-to-Voice Conversations**: Speak naturally with an AI tutor using microphone input
- **Multi-language Support**: English, Hindi, Marathi, and Tamil with real-time translation
- **Speech Recognition**: Powered by OpenAI Whisper with GPU acceleration
- **Text-to-Speech**: Natural voice responses using Google Text-to-Speech (gTTS)
- **Audio Caching**: Intelligent caching system using session state for faster response times
- **Multiple Input Methods**: Voice, text, or file upload

### 🎭 Interactive Roleplay Mode
- **Guided Scenarios**: Practice conversations in real-world situations
  - 🏫 **At School**: Teacher-student interactions
  - 🛒 **At the Store**: Shopping conversations with cashiers
  - 👨‍👩‍👧 **At Home**: Family conversations and daily activities
- **Dynamic Question Generation**: AI generates contextual questions for each scenario
- **Progress Tracking**: 10-question sessions with conversation history
- **Speech Quality Feedback**: Real-time feedback on speaking performance

## 🛡️ Safety & Privacy

### 🔒 **Child Safety First**
**SpeakGenie prioritizes child safety through carefully designed AI prompts and interactions:**

- **🛡️ Safe AI Prompts**: All prompts sent to Google Gemini AI are specifically crafted to ensure child-appropriate responses
- **🚫 Personal Information Protection**: The system is designed to avoid requesting or storing personal information like full names, addresses, or phone numbers
- **👶 Age-Appropriate Content**: AI responses are limited to educational, supportive, and encouraging content suitable for children aged 6-16
- **🚨 Content Filtering**: Built-in safeguards prevent inappropriate topics and maintain a safe learning environment
- **💬 Conversation Boundaries**: AI is programmed to redirect conversations back to educational topics if they drift off-course
- **🔐 Local Processing**: Voice recognition happens locally using OpenAI Whisper, ensuring audio privacy

**Note**: While we implement multiple safety measures, we recommend adult supervision during use, especially for younger children.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU (optional, for faster speech recognition)
- Google AI Studio API key (Gemini)
- Google Cloud service account with Speech API enabled

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/satyamrathirar/SpeakGenie.git
cd SpeakGenie
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env file and add your Google AI Studio API key
```

Required environment variables:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
GEMINI_MODEL=gemini-1.5-flash  # Optional, defaults to gemini-1.5-flash
```

**Note**: The `GOOGLE_APPLICATION_CREDENTIALS` should point to your Google Cloud service account JSON file for speech API access.

### Google Cloud Setup
1. **Create a Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one

2. **Enable Speech API**
   - Navigate to "APIs & Services" → "Library"
   - Search for "Cloud Speech-to-Text API"
   - Click "Enable"

3. **Create Service Account**
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Add roles: "Speech Client" or "Editor"
   - Download the JSON key file

5. **Run the application**
```bash
streamlit run streamlit-app.py
```

## 🛠️ Configuration

### Supported Languages
- English (default)
- Hindi (हिंदी)
- Marathi (मराठी)
- Tamil (தமிழ்)

### Roleplay Scenarios
Customize scenarios by editing `roleplay_scenarios.json`:
```json
{
  "scenario_key": {
    "title": "🎯 Scenario Title",
    "description": "Brief description",
    "context": "Detailed context for AI generation"
  }
}
```

### Fallback Questions
Edit `roleplay_questions.json` to provide backup questions when AI generation fails:
```json
{
  "scenario_key": [
    "Question 1",
    "Question 2",
    "..."
  ]
}
```

## 📋 Requirements

### Core Dependencies
- **streamlit**: Web application framework
- **google-generativeai**: Google Gemini AI integration
- **openai-whisper**: Speech-to-text conversion
- **gTTS**: Text-to-speech synthesis
- **deep-translator**: Multi-language translation
- **streamlit-mic-recorder**: Microphone input component
- **pydub**: Audio processing
- **torch**: PyTorch for Whisper GPU acceleration

### Optional Dependencies
- **CUDA**: For GPU-accelerated speech recognition
- **googletrans**: Alternative translation service

## 🎯 Usage Guide

### Free Chat Mode
1. Select "Free Chat" from the sidebar
2. Choose your preferred playback language
3. Use any input method:
   - 🎤 **Microphone**: Click to record, speak naturally, dont forget to click send !
   - 💬 **Text**: Type questions directly
   - 📎 **File Upload**: Upload audio files (WAV, MP3, M4A)
4. Receive AI responses with automatic voice playback

### Roleplay Mode
1. Select "Roleplay" from the sidebar
2. Choose a scenario (School, Store, or Home)
3. Follow the guided conversation:
   - Listen to AI prompts
   - Respond using the microphone
   - You may continue the conversation for each question as long as you wish
   - Progress through 10 interactive questions
4. View conversation summary upon completion

## 🔧 Advanced Features

### Audio Caching
- Automatically caches generated speech for faster playback
- Reduces API calls and improves performance
- Clear cache option available in sidebar

### GPU Acceleration
- Automatic GPU detection for Whisper
- Fallback to CPU if GPU unavailable
- Monitor usage with `nvidia-smi`

### Translation System
- Lazy loading for optimal startup performance
- Cached translation tests per language
- Fallback to English if translation fails


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- **OpenAI Whisper** for speech recognition
- **Google Gemini AI** for conversational AI
- **Streamlit** for the web framework
- **Deep Translator** for multi-language support


Made with ❤️ for young learners everywhere. Happy speaking! 🗣️✨


