# BlindAssit - Navigation Assistant with AI

A real-time navigation assistance system for visually impaired users, powered by SeaFormer segmentation and OpenRouter AI.

## 🎯 Features

- **Live Camera Processing** - Real-time semantic segmentation
- **Server-Side TTS** - No browser dependency for voice output
- **OpenRouter AI Integration** - Intelligent query responses
- **Vibration Feedback** - Haptic feedback for navigation
- **Dark Monochrome UI** - Simple, accessible 4-button interface
- **70% Segmentation Visibility** - Clear colored overlays

## 🚀 Quick Start

### 1. Watch the Demo

**Demo Video:** [amd1.mp4](./amd1.mp4)

See the BlindAssit navigation system in action with real-time semantic segmentation and AI-powered guidance.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up OpenRouter API Key

**Get your API key:** https://openrouter.ai/keys

**Set environment variable:**
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

**Or add to `start.sh`:**
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

### 4. Run the Server

```bash
./start.sh
```

### 5. Access the Web Interface

Open: http://localhost:5000

## 🎮 How to Use

### Buttons

1. **LIVE** - Confirms live navigation is active
2. **QUERY** - Ask OpenRouter AI about your surroundings
3. **SOS** - Emergency alert with strong vibration
4. **VOICE** - Toggle voice command recording

### Keyboard Shortcuts

- `1` - LIVE button
- `2` - QUERY button
- `3` - SOS button
- `4` - VOICE button
- `Arrow keys` - Speak navigation directions

## 🔧 Server-Side TTS

The app uses **Google Text-to-Speech (gTTS)** on the server, so there's no browser dependency:

- All audio generated server-side
- Sent to client as base64-encoded MP3
- Works in any browser with audio support
- No SpeechSynthesis API required

## 🤖 OpenRouter AI Integration

The QUERY button sends context to OpenRouter AI:

**Context includes:**
- Current navigation instruction
- Path analysis (left/center/right percentages)
- Hazard detection status
- Recent obstacles detected

**Example queries:**
- "What's ahead of me?"
- "Is it safe to cross?"
- "What should I do next?"

**Response:**
- AI analyzes the context
- Provides helpful guidance
- Response spoken via server-side TTS

## 📋 API Endpoints

### `/` (GET)
- Returns the main web interface

### `/segment` (POST)
- Accepts: JSON with `image` (base64)
- Returns: JSON with `segmented_image` (base64)
- Processes single frame with SeaFormer

### `/tts` (POST)
- Accepts: JSON with `text` (string)
- Returns: JSON with `audio` (base64 MP3)
- Generates speech using gTTS

### `/query` (POST)
- Accepts: JSON with `query` (string) and `context` (object)
- Returns: JSON with `response` (string) and `audio` (base64 MP3)
- Queries OpenRouter AI with navigation context

### `/health` (GET)
- Returns: JSON with server status
- Shows TTS and OpenRouter availability

## 🎨 UI Design

**Dark Monochrome:**
- Background: Pure black (#000000)
- Text: Pure white (#ffffff)
- Buttons: Dark gray (#1a1a1a)
- No colors or gradients

**4 Large Buttons:**
- LIVE (green indicator when active)
- QUERY (blue indicator when processing)
- SOS (red indicator when active)
- VOICE (yellow indicator when recording)

## 🔊 Audio System

### Navigation Instructions
Server-generated TTS speaks:
- "Path clear. Continue straight."
- "Obstacle ahead. Veer slight left."
- "Obstacle ahead. Veer slight right."
- "Caution: Path obstructed. Slow down."

### AI Query Responses
Server-generated TTS speaks AI responses:
- Context-aware guidance
- Safety recommendations
- Detailed explanations

### Vibration Feedback
- Subtle: Normal instruction changes
- Strong: SOS activation
- Pattern: Emergency alerts

## 📁 Project Structure

```
BlindAssit/
├── server/
│   ├── client_camera_server.py      # Flask server with TTS & OpenRouter
│   └── templates/
│       └── client_camera.html       # Dark monochrome UI
├── start_client_camera_demo.sh      # Launcher script
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── [SeaFormer model files...]
```

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Yes |
| `TTS_ENABLED` | Enable server-side TTS | No (default: true) |
| `OPENROUTER_ENABLED` | Enable OpenRouter AI | No (default: true) |
| `OPENROUTER_MODEL` | Model to use | No (default: openai/gpt-3.5-turbo) |

## 🐛 Troubleshooting

### TTS Not Working

**Check gTTS is installed:**
```bash
pip install gtts
```

**Check server logs for errors:**
```bash
tail -f server.log
```

### OpenRouter Not Working

**Check API key is set:**
```bash
echo $OPENROUTER_API_KEY
```

**Test API key:**
```bash
curl https://openrouter.ai/api/v1/auth/key \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

**Check server logs:**
```bash
tail -f server.log
```

### Camera Not Starting

**Check browser permissions:**
- Allow camera access
- Check if another app is using the camera

**Check server is running:**
```bash
curl http://localhost:5000/health
```

### Segments Not Visible

**Server automatically uses 70% visibility.** If still not visible:

1. Check server logs for errors
2. Verify SeaFormer model is loaded
3. Check `/health` endpoint status

## 📊 Server Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "tts_available": true,
  "openrouter_available": true,
  "model_loaded": true
}
```

## 🔄 Updates History

### Latest (3791b3a)
- ✅ Server-side TTS (gTTS)
- ✅ OpenRouter AI integration
- ✅ QUERY button with AI context
- ✅ No browser SpeechSynthesis dependency

### Previous (cf4b871)
- ✅ Dark monochrome UI
- ✅ 4 large buttons
- ✅ Vibration API
- ✅ 70% segmentation visibility

## 📝 License

This project uses SeaFormer for semantic segmentation. See original SeaFormer license for model usage terms.

## 🤝 Contributing

Contributions welcome! Please submit pull requests to:
https://github.com/manthanabc/BlindAssit
