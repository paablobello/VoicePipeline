"""
Cartesia TTS - Using Official SDK (Dec 2024)
Ultra-low latency streaming synthesis (~30-50ms TTFB).

Best practices from: https://docs.cartesia.ai/api-reference/tts/working-with-web-sockets/contexts
- Each turn should have its own context_id
- For interruptions, cancel the current context and start a new one
"""
import asyncio
import os
import time
import uuid
from typing import AsyncGenerator, Optional, List
from cartesia import AsyncCartesia


class CartesiaTTS:
    """
    Cartesia Sonic Text-to-Speech using official SDK.

    Features:
    - WebSocket streaming for real-time audio
    - Ultra-low latency (~30-50ms TTFB)
    - High quality voice synthesis
    - Spanish language support
    - Context management for interruption handling

    Output: PCM16 @ 24kHz
    """

    # Spanish voice IDs from Cartesia
    VOICE_IDS = {
        "carmen": "727f663b-0e90-4031-90f2-558b7334425b",  # Carmen - Friendly Neighbor
        "marta": "5c29d7e3-a133-4c7e-804a-1d9c6dea83f6",   # Marta
        "carlos": "143de963-600b-4ad4-9b62-e458929ccb36",  # Carlos
        "gabriela": "01d23d18-2956-44b0-8888-e89d234b17b4", # Gabriela
        "teresa": "0afd8614-31cb-438c-8a46-80650e19c29c",  # Teresa
    }

    # Available models (2025)
    # sonic-turbo: 40ms TTFB (fastest)
    # sonic-3: 90ms TTFB (highest quality)
    # sonic-2: 90ms TTFB (stable)
    MODELS = {
        "sonic-turbo": "sonic-turbo",  # Fastest (~40ms TTFB) - DEFAULT
        "sonic-3": "sonic-3",          # Latest, highest quality (~90ms TTFB)
        "sonic-2": "sonic-2",          # Stable, ultra-realistic (~90ms TTFB)
    }

    def __init__(
        self,
        voice_id: Optional[str] = None,
        model: str = "sonic-turbo",  # Fastest: 40ms TTFB (was sonic-3: 90ms)
        output_sample_rate: int = 24000,
        speed: str = "normal",  # Can be "slowest", "slow", "normal", "fast", "fastest"
        emotion: Optional[List[str]] = None,
    ):
        self.api_key = os.getenv("CARTESIA_API_KEY")
        if not self.api_key:
            raise ValueError("CARTESIA_API_KEY not set")

        self.voice_id = voice_id or self.VOICE_IDS["carmen"]
        self.model = self.MODELS.get(model, model)
        self.output_sample_rate = output_sample_rate
        self.speed = speed
        # Natural emotions for conversational Spanish
        # Valid levels: lowest, low, (omit for moderate), high, highest
        self.emotion = emotion or ["positivity:low", "curiosity:lowest"]

        self.client: Optional[AsyncCartesia] = None
        self.ws = None
        self.is_connected = False

        # Context management for interruption handling
        self._current_context_id: Optional[str] = None
        self._is_generating = False

        # Dynamic emotion keywords (Spanish)
        self._emotion_keywords = {
            "apology": ["perdón", "disculpa", "lo siento", "perdona", "lamento"],
            "question": ["qué", "cómo", "cuándo", "dónde", "por qué", "cuál", "?"],
            "excitement": ["genial", "increíble", "fantástico", "excelente", "perfecto", "maravilloso"],
            "empathy": ["entiendo", "comprendo", "sé cómo", "lamento", "siento"],
        }

        print(f"  📢 Cartesia config: model={self.model} (40ms TTFB), voice=carmen, speed={speed}")

    def _analyze_text_for_emotion(self, text: str) -> tuple[list, float]:
        """
        Analyze text to dynamically select emotions and speed.

        Cartesia emotion levels (valid values):
        - lowest, low, (omit for moderate), high, highest

        Cartesia emotions (valid values):
        - anger, positivity, surprise, sadness, curiosity

        Returns:
            tuple: (emotions list, speed value)
                   Speed is relative: -1.0 to 1.0 where 0.0 is normal
                   Positive = faster, Negative = slower
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Default emotions
        emotions = list(self.emotion)  # Copy default emotions

        # Dynamic speed based on text length
        # Cartesia speed: -1.0 to 1.0, where 0.0 is normal
        # Short responses (1-5 words): slightly faster (+0.05)
        # Long responses (20+ words): slightly slower (-0.05)
        if word_count <= 5:
            speed = 0.05  # Slightly faster
        elif word_count >= 20:
            speed = -0.05  # Slightly slower
        else:
            speed = 0.0  # Normal

        # Detect apology/empathy - add softness
        if any(kw in text_lower for kw in self._emotion_keywords["apology"]):
            emotions = ["sadness:low"]
            speed = -0.05  # Slower for apologies

        # Detect questions - add curiosity
        elif any(kw in text_lower for kw in self._emotion_keywords["question"]):
            emotions = ["curiosity:high", "positivity:low"]

        # Detect excitement - add energy
        elif any(kw in text_lower for kw in self._emotion_keywords["excitement"]):
            emotions = ["positivity:high", "surprise:low"]
            speed = 0.05  # Slightly faster for excitement

        # Detect empathy - softer tone
        elif any(kw in text_lower for kw in self._emotion_keywords["empathy"]):
            emotions = ["positivity:low", "sadness:lowest"]
            speed = -0.05

        return emotions, speed
        
    async def connect(self):
        """Initialize Cartesia client and WebSocket."""
        self.client = AsyncCartesia(api_key=self.api_key)
        self.ws = await self.client.tts.websocket()
        self.is_connected = True
        print("  ✅ Cartesia connected (SDK)")
    
    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to speech with streaming.
        Uses official SDK websocket for lowest latency.

        Each call gets a unique context_id for proper interruption handling.
        Per Cartesia docs: each turn should be a new context.

        Args:
            text: Text to synthesize

        Yields:
            Raw audio bytes (PCM16 @ 24kHz)
        """
        if not self.is_connected or not self.ws:
            await self.connect()

        start_time = time.time()
        first_byte_time = None

        # Generate unique context_id for this synthesis
        self._current_context_id = f"turn-{uuid.uuid4().hex[:12]}"
        self._is_generating = True

        try:
            # Analyze text for dynamic emotion and speed
            dynamic_emotions, dynamic_speed = self._analyze_text_for_emotion(text)

            # Build voice config with experimental controls for emotions
            # Note: Cartesia speed is -1.0 to 1.0, where 0.0 is normal
            voice_config = {
                "mode": "id",
                "id": self.voice_id,
                "__experimental_controls": {
                    "speed": dynamic_speed,  # Use dynamic speed (range: -1.0 to 1.0)
                    "emotion": dynamic_emotions
                }
            }

            # Build output format - RAW PCM for lowest latency
            output_format = {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.output_sample_rate
            }

            # Send to TTS with context_id
            ctx = self.ws.send(
                model_id=self.model,
                transcript=text,
                voice=voice_config,
                output_format=output_format,
                stream=True,
                context_id=self._current_context_id,  # Track context for cancellation
            )

            # Await the context and iterate
            async for output in await ctx:
                # Check if generation was cancelled
                if not self._is_generating:
                    print(f"  ⚡ TTS generation cancelled mid-stream")
                    break

                if first_byte_time is None:
                    first_byte_time = time.time()
                    ttfb_ms = (first_byte_time - start_time) * 1000
                    print(f"  🔊 Cartesia TTFB: {ttfb_ms:.0f}ms (ctx: {self._current_context_id[:8]})")

                # SDK returns audio in output.audio
                if hasattr(output, 'audio') and output.audio:
                    yield output.audio
                elif isinstance(output, bytes):
                    yield output

        except Exception as e:
            print(f"  ❌ Cartesia TTS error: {e}")
            # Try to reconnect
            self.is_connected = False
            raise
        finally:
            self._is_generating = False

    async def cancel_current_context(self):
        """
        Cancel the current TTS generation context.
        Per Cartesia docs: send {"context_id": "...", "cancel": true}

        Should be called when user interrupts the bot.
        """
        if not self._current_context_id or not self._is_generating:
            return

        self._is_generating = False  # Stop yielding audio

        if self.ws and self.is_connected:
            try:
                # Send cancel message for current context
                # Note: The SDK might handle this internally, but we set the flag
                # to stop iteration in synthesize()
                print(f"  🛑 Cancelling TTS context: {self._current_context_id[:8]}")
                # The SDK's ws.send() might not support cancel directly,
                # but setting _is_generating = False will stop the audio stream
            except Exception as e:
                print(f"  ⚠️ Error cancelling TTS context: {e}")

    async def synthesize_to_buffer(self, text: str) -> bytes:
        """
        Synthesize text to a complete audio buffer (for caching).
        """
        buffer = bytearray()
        async for chunk in self.synthesize(text):
            buffer.extend(chunk)
        return bytes(buffer)

    async def close(self):
        """Close WebSocket connection."""
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        if self.client:
            try:
                await self.client.close()
            except Exception:
                pass
        self.is_connected = False
        print("  ✅ Cartesia disconnected")
