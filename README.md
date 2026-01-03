# Voice AI Pipeline

> A real-time voice conversation system built from scratch to explore the world of Voice AI, STT, TTS, and conversational agents. Targeting sub-700ms end-to-end latency.

## Architecture

```
Audio In → VAD → STT → LLM → TTS → Audio Out
   │        │     │     │     │
   │     Silero  Deepgram Groq  Cartesia
   │     (~5ms)  Nova-2  Llama4 Sonic-Turbo
   │                     (~150ms)(~40ms TTFB)
   └─────────────────────────────────────────→ ~500-700ms total
```

## Key Features

### Turn Detection (Hybrid Approach)

- **Silero VAD** - Voice Activity Detection with hysteresis thresholds to avoid rapid switching
- **Smart Turn v3** - [Pipecat's ML model](https://github.com/pipecat-ai/smart-turn) using Whisper encoder for prosody-based end-of-turn detection
- **EOU Detector** - Spanish linguistic patterns (prepositions, conjunctions, incomplete verb forms)

### Streaming Pipeline

- **Sentence-by-sentence TTS** - Audio starts playing before LLM finishes generating
- **Parallel processing** - STT/LLM/TTS run concurrently via asyncio
- **Interruption handling** - Real-time barge-in with context cancellation

### Multi-Provider Support

| Component | Providers |
|-----------|-----------|
| STT | Deepgram Nova-2/Nova-3 |
| LLM | Groq, Cerebras, OpenAI, Gemini |
| TTS | Cartesia Sonic, ElevenLabs |

## Tech Stack

**Backend:** FastAPI, Python 3.11+, WebSockets, asyncio
**Frontend:** Next.js 14, TypeScript, Web Audio API
**ML Models:** Silero VAD (ONNX), Smart Turn v3 (ONNX)

## Latency Optimizations

| Technique | Impact |
|-----------|--------|
| Streaming STT (interim results) | -200ms |
| Sentence chunking for TTS | -300ms |
| Persistent WebSocket connections | -100ms |
| VAD-based endpointing (500ms) | Faster turn detection |
| TTS context management | Clean interruptions |

## Project Structure

```
backend/
├── main.py              # FastAPI server + WebSocket handler
├── pipeline/
│   ├── session.py       # Core conversation logic
│   ├── vad.py           # Silero VAD wrapper
│   ├── stt.py           # Deepgram streaming client
│   ├── llm.py           # Multi-provider LLM (Groq/Cerebras/OpenAI/Gemini)
│   ├── tts.py           # Cartesia Sonic client
│   ├── turn_detector.py # Hybrid turn detection
│   ├── smart_turn.py    # Pipecat Smart Turn v3
│   └── eou_detector.py  # Spanish linguistic patterns

frontend/
├── app/page.tsx         # Main UI + audio handling
└── components/
    ├── VoiceInterface.tsx    # Audio visualizer
    └── MetricsDashboard.tsx  # Real-time latency metrics
```

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python main.py

# Frontend
cd frontend
npm install
npm run dev
```

## References

- [Deepgram: Understanding STT Latency](https://deepgram.com/learn/understanding-and-reducing-latency-in-speech-to-text-apis)
- [Pipecat Smart Turn](https://github.com/pipecat-ai/smart-turn) - End-of-turn detection model
- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice Activity Detection
- [Cartesia Sonic](https://docs.cartesia.ai) - Low-latency TTS
- [LiveKit: Reducing Voice Agent Latency](https://kb.livekit.io/articles/4490830410)

## Metrics Dashboard

Real-time visualization of pipeline latency:
- P50/P95/Avg for each component (VAD, STT, LLM, TTS)
- Time to First Audio (TTFA)
- Session statistics (turns, interruptions, cache hits)

---

Built as a learning project to understand voice AI pipelines end-to-end.
