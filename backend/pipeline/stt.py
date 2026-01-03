"""
Deepgram STT - Raw WebSocket (websockets 13.1 compatible)
Best practices from: https://developers.deepgram.com/docs/audio-keep-alive
"""
import asyncio
import json
import os
import time
import random
from typing import AsyncGenerator, Optional, List
import websockets
from websockets.exceptions import ConnectionClosed


class DeepgramSTT:
    """
    Deepgram Nova-2 Speech-to-Text streaming client.
    Using websockets 13.1 which is fully compatible.

    Implements Deepgram best practices:
    - KeepAlive every 5 seconds (timeout is 10s)
    - Audio buffering during disconnects
    - Exponential backoff with jitter for reconnection
    """

    def __init__(
        self,
        model: str = "nova-2",  # nova-2 for Spanish (nova-3 has compatibility issues with some params)
        language: str = "es",
        sample_rate: int = 16000,
    ):
        self.api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not set")

        self.model = model
        self.language = language
        self.sample_rate = sample_rate

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.receive_task: Optional[asyncio.Task] = None
        self.keepalive_task: Optional[asyncio.Task] = None

        # Results queue (bounded to prevent memory bloat)
        self.results_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Metrics
        self.last_audio_time: Optional[float] = None

        # Audio buffer for reconnection (store up to 5 seconds of audio)
        self._audio_buffer: List[bytes] = []
        self._max_buffer_chunks = 156  # ~5 seconds at 32ms chunks
        self._is_reconnecting = False

        # Exponential backoff state
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._base_delay = 1.0  # Start at 1 second
        self._max_delay = 30.0  # Cap at 30 seconds
        
    async def connect(self):
        """Establish WebSocket connection to Deepgram."""
        # Working configuration (verified)
        # Optimized parameters for lower latency (Phase 1)
        # - endpointing: 500ms (balanced - 300ms was too aggressive, caused interruptions)
        # - utterance_end_ms: 1000ms (minimum allowed by Deepgram API)
        url = (
            f"wss://api.deepgram.com/v1/listen?"
            f"model={self.model}&"
            f"language={self.language}&"
            f"punctuate=true&"
            f"interim_results=true&"
            f"utterance_end_ms=1000&"
            f"vad_events=true&"
            f"endpointing=500&"
            f"smart_format=true&"
            f"encoding=linear16&"
            f"sample_rate={self.sample_rate}&"
            f"channels=1"
        )

        try:
            # websockets 13.1 uses extra_headers
            self.ws = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Token {self.api_key}"},
                ping_interval=20,
                ping_timeout=10,
            )
            self.is_connected = True
            self._reconnect_attempts = 0  # Reset on successful connection
            print("  ✅ Deepgram connected")

            # Start receiver task
            self.receive_task = asyncio.create_task(self._receive_loop())

            # Start dedicated KeepAlive task (every 5 seconds per Deepgram docs)
            self.keepalive_task = asyncio.create_task(self._keepalive_loop())

        except Exception as e:
            print(f"  ❌ Deepgram connection failed: {e}")
            raise

    async def _keepalive_loop(self):
        """
        Dedicated KeepAlive task - sends every 5 seconds.
        Per Deepgram docs: timeout is 10s, so send every 3-5s.
        https://developers.deepgram.com/docs/audio-keep-alive
        """
        while self.is_connected and self.ws:
            try:
                await asyncio.sleep(5.0)  # Send every 5 seconds
                if self.ws and self.is_connected:
                    # Must be sent as TEXT frame, not binary
                    await self.ws.send(json.dumps({"type": "KeepAlive"}))
            except ConnectionClosed:
                self.is_connected = False
                break
            except Exception as e:
                print(f"  ⚠️ KeepAlive error: {e}")
                self.is_connected = False
                break
    
    async def send_audio(self, audio_bytes: bytes):
        """
        Send audio chunk to Deepgram with auto-reconnection.
        Implements buffering and exponential backoff per Deepgram best practices.
        https://developers.deepgram.com/docs/recovering-from-connection-errors-and-timeouts-when-live-streaming-audio
        """
        # Always buffer audio (for potential reconnection)
        self._audio_buffer.append(audio_bytes)
        if len(self._audio_buffer) > self._max_buffer_chunks:
            self._audio_buffer.pop(0)  # Keep rolling buffer

        # Auto-reconnect if connection lost
        if not self.is_connected or not self.ws:
            if self._is_reconnecting:
                return  # Another task is handling reconnection

            self._is_reconnecting = True
            try:
                await self._reconnect_with_backoff()
            finally:
                self._is_reconnecting = False

            if not self.is_connected:
                return  # Still not connected after retries

        try:
            self.last_audio_time = time.time()
            await self.ws.send(audio_bytes)
        except ConnectionClosed:
            self.is_connected = False
            print("  ⚠️ Deepgram connection closed, will reconnect on next audio")

    async def _reconnect_with_backoff(self):
        """
        Reconnect with exponential backoff and jitter.
        Per Deepgram docs: start at 1s, double up to 30s cap, add jitter.
        """
        while self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1

            # Calculate delay with exponential backoff + jitter
            delay = min(
                self._base_delay * (2 ** (self._reconnect_attempts - 1)),
                self._max_delay
            )
            # Add jitter (0-25% of delay) to prevent thundering herd
            jitter = random.uniform(0, delay * 0.25)
            actual_delay = delay + jitter

            print(f"  🔄 Deepgram reconnecting (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}, delay: {actual_delay:.1f}s)...")

            await asyncio.sleep(actual_delay)

            try:
                await self.connect()
                if self.is_connected:
                    # Replay buffered audio (at max 1.25x realtime per Deepgram docs)
                    if self._audio_buffer:
                        print(f"  📤 Replaying {len(self._audio_buffer)} buffered audio chunks...")
                        for chunk in self._audio_buffer:
                            if self.ws and self.is_connected:
                                await self.ws.send(chunk)
                                await asyncio.sleep(0.025)  # ~1.25x realtime
                        self._audio_buffer.clear()
                    return
            except Exception as e:
                print(f"  ❌ Reconnection attempt {self._reconnect_attempts} failed: {e}")

        print(f"  ❌ Deepgram reconnection failed after {self._max_reconnect_attempts} attempts")
    
    async def _receive_loop(self):
        """Background task to receive transcription results."""
        while self.is_connected and self.ws:
            try:
                # Longer timeout since KeepAlive is handled by dedicated task
                message = await asyncio.wait_for(self.ws.recv(), timeout=15.0)
                data = json.loads(message)

                if data.get("type") == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])

                    if alternatives:
                        transcript = alternatives[0].get("transcript", "")
                        confidence = alternatives[0].get("confidence", 0)
                        is_final = data.get("is_final", False)

                        latency_ms = 0
                        if self.last_audio_time:
                            latency_ms = (time.time() - self.last_audio_time) * 1000

                        if transcript.strip():
                            result = {
                                "type": "transcript",
                                "text": transcript,
                                "is_final": is_final,
                                "confidence": confidence,
                                "latency_ms": latency_ms,
                            }
                            await self.results_queue.put(result)

                            # Log both interim and final for debugging
                            if is_final:
                                print(f"  📝 STT final: \"{transcript}\" ({latency_ms:.0f}ms)")
                            else:
                                # Log interim every ~500ms to avoid spam
                                if latency_ms < 200 or int(latency_ms) % 500 < 100:
                                    print(f"  📝 STT interim: \"{transcript[:40]}...\" ({latency_ms:.0f}ms)")

                elif data.get("type") == "UtteranceEnd":
                    await self.results_queue.put({"type": "utterance_end"})

                elif data.get("type") == "SpeechStarted":
                    await self.results_queue.put({"type": "speech_started"})

            except asyncio.TimeoutError:
                # KeepAlive is handled by dedicated task, just continue waiting
                continue
            except ConnectionClosed:
                self.is_connected = False
                break
            except Exception as e:
                print(f"  ⚠️ Deepgram error: {e}")
                continue
    
    async def receive_transcripts(self) -> AsyncGenerator[dict, None]:
        """Receive transcription results."""
        while self.is_connected:
            try:
                result = await asyncio.wait_for(self.results_queue.get(), timeout=0.5)
                yield result
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
                break
    
    async def flush(self):
        """Flush buffer."""
        if self.ws and self.is_connected:
            try:
                await self.ws.send(json.dumps({"type": "Finalize"}))
            except:
                pass
    
    async def close(self):
        """Close connection."""
        self.is_connected = False

        # Cancel keepalive task
        if self.keepalive_task:
            self.keepalive_task.cancel()
            try:
                await self.keepalive_task
            except asyncio.CancelledError:
                pass

        # Cancel receive task
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except:
                pass

        # Clear buffer
        self._audio_buffer.clear()

        print("  ✅ Deepgram disconnected")
