"""
TTS Cache - Pre-synthesized common phrases
Zero latency for cached responses (Sierra AI technique).
"""
from typing import Dict, Optional


class TTSCache:
    """
    Cache for pre-synthesized TTS audio.
    
    Common phrases like greetings, acknowledgments, etc.
    are pre-synthesized for zero-latency playback.
    
    TTFB for cached phrases: 0ms
    """
    
    # Phrases to pre-cache
    CACHED_PHRASES = {
        "greeting": "¡Hola! ¿En qué puedo ayudarte hoy?",
        "greeting_short": "¡Hola!",
        "thinking": "Dame un momento.",
        "clarify": "¿Podrías repetir eso?",
        "goodbye": "¡Hasta luego!",
        "acknowledge": "Entendido.",
        "ok": "Ok.",
        "yes": "Sí.",
        "no_entiendo": "No entendí bien, ¿puedes repetir?",
    }
    
    # Backchannels (short acknowledgments)
    BACKCHANNELS = {
        "mmhmm": "Mm-hmm.",
        "aja": "Ajá.",
        "ya": "Ya.",
        "claro": "Claro.",
        "entiendo": "Entiendo.",
        "ok_short": "Ok.",
        "si_short": "Sí.",
    }
    
    def __init__(self):
        self.cache: Dict[str, bytes] = {}
        self.is_warmed = False
    
    async def warm_cache(self, tts_client):
        """
        Pre-synthesize all cached phrases.
        Call this on startup for zero-latency responses.
        """
        print("  🔥 Warming TTS cache...")
        
        all_phrases = {**self.CACHED_PHRASES, **self.BACKCHANNELS}
        
        for key, phrase in all_phrases.items():
            try:
                audio = await tts_client.synthesize_to_buffer(phrase)
                self.cache[key] = audio
                print(f"    ✅ Cached: {key}")
            except Exception as e:
                print(f"    ⚠️ Failed to cache {key}: {e}")
        
        self.is_warmed = True
        print(f"  ✅ TTS cache warmed ({len(self.cache)} phrases)")
    
    def get(self, key: str) -> Optional[bytes]:
        """Get cached audio by key."""
        return self.cache.get(key)
    
    def get_greeting(self) -> Optional[bytes]:
        """Get greeting audio."""
        return self.cache.get("greeting")
    
    def get_backchannel(self, backchannel_type: str = "mmhmm") -> Optional[bytes]:
        """Get backchannel audio."""
        return self.cache.get(backchannel_type)
    
    def has(self, key: str) -> bool:
        """Check if key is cached."""
        return key in self.cache
    
    def add(self, key: str, audio: bytes):
        """Add audio to cache."""
        self.cache[key] = audio
    
    def clear(self):
        """Clear all cached audio."""
        self.cache.clear()
        self.is_warmed = False

