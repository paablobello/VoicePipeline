"""
ElevenLabs TTS - Using Official SDK (2025)
Most natural-sounding AI voices with streaming.

Models:
- eleven_multilingual_v2: Best quality for Spanish (~250-300ms TTFB)
- eleven_flash_v2_5: Fastest, 32 languages (~75ms TTFB)
- eleven_turbo_v2_5: Balanced speed/quality (~150ms TTFB)
"""
import asyncio
import os
import time
from typing import AsyncGenerator, Optional
from elevenlabs.client import AsyncElevenLabs


class ElevenLabsTTS:
    """
    ElevenLabs Text-to-Speech using official SDK.

    Features:
    - Most natural-sounding AI voices
    - Streaming with convert_as_stream
    - Flash v2.5 for low latency with language_code support

    Output: PCM16 @ 24kHz (pcm_24000)
    """

    # Official ElevenLabs default voices (verified working)
    # See: https://elevenlabs.io/docs/capabilities/voices
    VOICE_IDS = {
        # Default multilingual voices (work well with Spanish)
        "aria": "9BWtsMINqrJLrRacOk9x",         # Aria - expressive female
        "roger": "CwhRBWXzGAHq8TQ4Fs17",        # Roger - confident male
        "sarah": "EXAVITQu4vr4xnSDxMaL",        # Sarah - soft female
        "laura": "FGY2WhTYpPnrIDTdsKH5",        # Laura - upbeat female
        "charlie": "IKne3meq5aSn9XLyUdCD",      # Charlie - casual male
        "george": "JBFqnCBsd6RMkjVDRZzb",       # George - warm male
        "callum": "N2lVS1w4EtoT3dr4eOWO",       # Callum - intense male
        "river": "SAz9YHcvj6GT2YYXdXww",        # River - confident non-binary
        "liam": "TX3LPaxmHKxFdv7VOQHJ",         # Liam - articulate male
        "charlotte": "XB0fDUnXU5powFXDhCwa",    # Charlotte - seductive female
        "alice": "Xb7hH8MSUJpSbSDYk0k2",        # Alice - confident female
        "matilda": "XrExE9yKIg1WjnnlVkGX",      # Matilda - warm female
        "will": "bIHbv24MWmeRgasZH58o",         # Will - friendly male
        "jessica": "cgSgspJ2msm6clMCkdW9",      # Jessica - expressive female
        "eric": "cjVigY5qzO86Huf0OWal",         # Eric - friendly male
        "chris": "iP95p4xoKVk53GoZ742B",        # Chris - casual male
        "brian": "nPczCjzI2devNBz1zQrb",        # Brian - deep male
        "daniel": "onwK4e9ZLuTAKqWW03F9",       # Daniel - authoritative male
        "lily": "pFZP5JQG7iQjIQuC4Bku",         # Lily - warm female
        "bill": "pqHfZKP75CvOlQylNhV4",         # Bill - trustworthy male
    }

    # Models (2025)
    MODELS = {
        "multilingual_v2": "eleven_multilingual_v2",  # Best quality, 29 langs
        "flash_v2_5": "eleven_flash_v2_5",            # Fastest ~75ms, 32 langs
        "turbo_v2_5": "eleven_turbo_v2_5",            # Balanced ~150ms, 32 langs
    }

    # Supported output formats for PCM
    OUTPUT_FORMATS = {
        16000: "pcm_16000",
        22050: "pcm_22050",
        24000: "pcm_24000",
        44100: "pcm_44100",
    }

    def __init__(
        self,
        voice_id: Optional[str] = None,
        model: str = "flash_v2_5",  # Fast + supports language_code for Spanish
        output_sample_rate: int = 24000,
        stability: float = 0.4,          # 35-40% recommended for natural speech
        similarity_boost: float = 0.75,  # Keep at/below 75-80%
        style: float = 0.0,
        language_code: str = "es",       # Spanish - improves pronunciation
    ):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not set")

        # Use Aria by default (good multilingual support)
        self.voice_id = voice_id or self.VOICE_IDS["aria"]
        self.model_id = self.MODELS.get(model, model)
        self.output_sample_rate = output_sample_rate
        self.output_format = self.OUTPUT_FORMATS.get(output_sample_rate, "pcm_24000")

        # Voice settings optimized for natural speech
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style

        # Language code for Flash/Turbo v2.5 (improves Spanish pronunciation)
        self.language_code = language_code

        self.client: Optional[AsyncElevenLabs] = None
        self.is_connected = False

        print(f"  📢 ElevenLabs config: model={self.model_id}, voice=aria, lang={language_code}")
    
    async def connect(self):
        """Initialize ElevenLabs client."""
        self.client = AsyncElevenLabs(api_key=self.api_key)
        self.is_connected = True
        print("  ✅ ElevenLabs connected (SDK)")
    
    async def close(self):
        """Close client."""
        self.is_connected = False
        print("  ✅ ElevenLabs disconnected")
    
    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize text to speech with streaming.

        Uses convert_as_stream for true streaming with lowest latency.

        Args:
            text: Text to synthesize

        Yields:
            Raw audio bytes (PCM16 @ 24kHz)
        """
        if not self.is_connected or not self.client:
            await self.connect()

        start_time = time.time()
        first_byte_time = None

        try:
            # Build request parameters
            request_params = {
                "voice_id": self.voice_id,
                "text": text,
                "model_id": self.model_id,
                "output_format": self.output_format,
                "voice_settings": {
                    "stability": self.stability,
                    "similarity_boost": self.similarity_boost,
                    "style": self.style,
                }
            }

            # Add language_code for Flash/Turbo v2.5 models (improves Spanish)
            if "flash" in self.model_id or "turbo" in self.model_id:
                request_params["language_code"] = self.language_code

            # Use stream() for async streaming
            # Returns an async iterator of audio chunks
            audio_stream = self.client.text_to_speech.stream(
                **request_params
            )

            # Iterate over the streaming response
            async for chunk in audio_stream:
                if first_byte_time is None:
                    first_byte_time = time.time()
                    ttfb_ms = (first_byte_time - start_time) * 1000
                    print(f"  🔊 ElevenLabs TTFB: {ttfb_ms:.0f}ms")

                # Handle different response types
                if hasattr(chunk, 'audio'):
                    yield chunk.audio
                elif isinstance(chunk, bytes):
                    yield chunk
                else:
                    yield bytes(chunk)

        except Exception as e:
            print(f"  ❌ ElevenLabs TTS error: {e}")
            raise
    
    async def synthesize_to_buffer(self, text: str) -> bytes:
        """
        Synthesize text to a complete audio buffer (for caching).
        """
        buffer = bytearray()
        async for chunk in self.synthesize(text):
            buffer.extend(chunk)
        return bytes(buffer)
