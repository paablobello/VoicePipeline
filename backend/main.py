"""
Voice AI Pipeline - FastAPI Backend
State of the art implementation with streaming, VAD, turn detection.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
from typing import Dict
import os
from dotenv import load_dotenv

from pipeline.vad import SileroVAD
from pipeline.turn_detector import TurnDetector
from pipeline.session import VoiceSession
from pipeline.tts_cache import TTSCache

load_dotenv()

# Global state
vad_model: SileroVAD = None
turn_detector: TurnDetector = None
tts_cache: TTSCache = None
active_sessions: Dict[str, VoiceSession] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle - load models on startup."""
    global vad_model, turn_detector, tts_cache
    import numpy as np

    print("=" * 50)
    print("🔄 INITIALIZING VOICE AI PIPELINE")
    print("=" * 50)

    # 1. Load VAD model
    print("\n🔄 Loading VAD model (Silero with hysteresis)...")
    vad_model = SileroVAD()
    # Warm up VAD with a test audio chunk (512 samples of silence)
    test_audio = np.zeros(512, dtype=np.float32)
    _ = vad_model.process(test_audio)
    print("✅ VAD loaded and warmed up")

    # 2. Initialize Turn Detector
    print("\n🔄 Initializing Turn Detector...")
    turn_detector = TurnDetector()
    print("✅ Turn Detector ready")
    print(f"   Base silence threshold: {turn_detector.base_silence_threshold_ms}ms")
    print(f"   Min speech duration: {turn_detector.min_speech_duration_ms}ms")

    # 3. Pre-load Smart Turn v3 model for faster first connection
    print("\n🔄 Pre-loading Smart Turn v3 model...")
    try:
        await turn_detector.initialize_smart_turn()
        print("✅ Smart Turn v3 pre-loaded")
    except Exception as e:
        print(f"⚠️ Smart Turn pre-load failed (will use fallback): {e}")

    # 4. Initialize and warm up EOU Detector
    print("\n🔄 Initializing EOU Detector for Spanish...")
    from pipeline.eou_detector import get_eou_detector
    eou = get_eou_detector()
    # Test with sample text to compile regex patterns
    _ = eou.analyze("Hola, ¿qué tal?", silence_duration_ms=500, speech_duration_ms=1000)
    print("✅ EOU Detector ready")
    print(f"   Base silence: {eou.base_silence_ms}ms")
    print(f"   Max extension: {eou.max_extension_ms}ms")

    # 5. Initialize TTS Cache
    print("\n🔄 Initializing TTS Cache...")
    tts_cache = TTSCache()
    print("✅ TTS Cache ready")

    print("\n" + "=" * 50)
    print("🚀 VOICE AI PIPELINE READY!")
    print("=" * 50)
    print("\nConfiguration Summary:")
    print(f"  • VAD: Silero ONNX with hysteresis (0.4/0.45)")
    print(f"  • Turn Detection: {turn_detector.base_silence_threshold_ms}ms base silence")
    print(f"  • Fast path: {turn_detector.short_response_threshold_ms}ms for 'sí/ok/perfecto'")
    print(f"  • Min speech: {turn_detector.min_speech_duration_ms}ms")
    print(f"  • STT: Deepgram Nova-2 (Spanish, 1000ms utterance)")
    print(f"  • EOU: Spanish patterns ({eou.base_silence_ms}ms base)")
    print(f"  • Interruption: 50ms speech detection")
    print("=" * 50 + "\n")

    yield

    # Cleanup
    print("\n👋 Shutting down...")
    for session in active_sessions.values():
        await session.cleanup()
    print("✅ All sessions cleaned up")


app = FastAPI(
    title="Voice AI Pipeline",
    description="Real-time voice conversation with AI - State of the Art",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/providers")
async def list_providers():
    """
    List available LLM and TTS providers.
    Use these values in the WebSocket URL query parameters.
    """
    return {
        "llm": {
            "cerebras": {"latency": "~100-150ms", "model": "llama-3.3-70b", "description": "FASTEST - Cerebras Wafer-Scale"},
            "groq": {"latency": "~100-150ms", "model": "llama-4-scout (750 T/s)", "description": "Llama 4 - Groq LPU"},
            "gemini": {"latency": "~200-400ms", "model": "gemini-2.5-flash (3-flash disponible)", "description": "Gemini 3 disponible - Free tier"},
            "openai": {"latency": "~600-800ms", "model": "gpt-4o-mini", "description": "High quality - OpenAI"},
        },
        "tts": {
            "cartesia": {"latency": "~50-100ms", "quality": "Very Good", "description": "Fastest TTS - Sonic 3"},
            "elevenlabs": {"latency": "~75-150ms", "quality": "Excellent", "description": "Flash v2.5 - Most natural"},
        }
    }


@app.websocket("/ws/voice/{session_id}")
async def voice_websocket(
    websocket: WebSocket, 
    session_id: str,
    llm: str = "groq",  # Query param: ?llm=cerebras
    tts: str = "cartesia",  # Query param: ?tts=elevenlabs
):
    """
    Main WebSocket endpoint for voice communication.
    
    Query Parameters:
    - llm: LLM provider (cerebras, groq, gemini, openai)
    - tts: TTS provider (cartesia, elevenlabs)
    
    Protocol:
    - Client → Server: Binary audio chunks (PCM16 @ 16kHz mono)
    - Server → Client: 
        - JSON: {"type": "transcript", "role": "user"|"assistant", "text": "..."}
        - JSON: {"type": "llm_token", "token": "..."}
        - JSON: {"type": "metrics", "data": {...}}
        - JSON: {"type": "control", "action": "..."}
        - Binary: TTS audio chunks (PCM16 @ 24kHz)
    """
    await websocket.accept()
    print(f"✅ Client connected: {session_id} (LLM: {llm}, TTS: {tts})")
    
    # Create session with selected providers
    session = VoiceSession(
        session_id=session_id,
        websocket=websocket,
        vad=vad_model,
        turn_detector=turn_detector,
        tts_cache=tts_cache,
        llm_provider=llm,
        tts_provider=tts,
    )
    active_sessions[session_id] = session
    
    # Background tasks
    tasks = []
    
    try:
        # Initialize session (connect to APIs)
        await session.start()
        
        # Task 1: Receive audio from client
        async def receive_audio():
            while True:
                try:
                    data = await websocket.receive()
                    if "bytes" in data:
                        await session.audio_input_queue.put(data["bytes"])
                    elif "text" in data:
                        # Handle JSON control messages
                        msg = json.loads(data["text"])
                        if msg.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Receive error: {e}")
                    break
        
        # Task 2: Process audio pipeline
        async def process_pipeline():
            async for event in session.process():
                try:
                    if event["type"] == "audio":
                        await websocket.send_bytes(event["data"])
                    else:
                        await websocket.send_json(event)
                except Exception as e:
                    print(f"Send error: {e}")
                    break
        
        # Run tasks concurrently
        tasks = [
            asyncio.create_task(receive_audio()),
            asyncio.create_task(process_pipeline()),
        ]
        
        # Wait for any task to complete (usually due to disconnect)
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
    except WebSocketDisconnect:
        print(f"❌ Client disconnected: {session_id}")
    except Exception as e:
        print(f"❌ Error in session {session_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        for task in tasks:
            if not task.done():
                task.cancel()
        await session.cleanup()
        if session_id in active_sessions:
            del active_sessions[session_id]
        print(f"🧹 Session cleaned up: {session_id}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "models_loaded": {
            "vad": vad_model is not None,
            "turn_detector": turn_detector is not None,
            "tts_cache": tts_cache is not None,
        }
    }


@app.get("/metrics")
async def get_metrics():
    """Get aggregated metrics from all sessions."""
    all_metrics = []
    for session_id, session in active_sessions.items():
        all_metrics.append({
            "session_id": session_id,
            "metrics": session.metrics.get_summary()
        })
    return {"sessions": all_metrics}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

