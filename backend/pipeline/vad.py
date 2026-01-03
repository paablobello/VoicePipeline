"""
Silero VAD - Voice Activity Detection
Ultra-low latency (~5ms) speech detection.
"""
import numpy as np
import torch
import time
from typing import Optional


class SileroVAD:
    """
    Silero Voice Activity Detection wrapper.

    Optimized for real-time voice AI:
    - ONNX runtime for fast inference
    - 512 samples per chunk (32ms @ 16kHz)
    - Hysteresis to avoid rapid on/off switching
    - Adaptive threshold support
    """

    def __init__(
        self,
        threshold: float = 0.5,  # Increased: less sensitive to background noise
        threshold_on: float = 0.55,  # Higher: require clearer speech to start
        threshold_off: float = 0.40,  # Higher: detect silence faster
        sample_rate: int = 16000,
        use_onnx: bool = True,
        min_speech_chunks: int = 2,  # Require N consecutive chunks to confirm speech
        min_silence_chunks: int = 2,  # Reduced from 3: detect silence faster (~64ms)
    ):
        self.threshold = threshold
        self.threshold_on = threshold_on
        self.threshold_off = threshold_off
        self.sample_rate = sample_rate
        self.use_onnx = use_onnx
        self.min_speech_chunks = min_speech_chunks
        self.min_silence_chunks = min_silence_chunks
        
        # Load Silero VAD model
        print("  Loading Silero VAD...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=use_onnx,
            trust_repo=True
        )
        
        # Extract utility functions
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks
        ) = utils
        
        # State tracking
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self.cumulative_speech_ms: float = 0
        self.cumulative_silence_ms: float = 0

        # Hysteresis state
        self.is_currently_speaking: bool = False  # Current stable state
        self.consecutive_speech_chunks: int = 0
        self.consecutive_silence_chunks: int = 0
        self.recent_probabilities: list = []  # Rolling window for smoothing
        self.smoothing_window: int = 2  # Reduced from 3 for 32ms faster response

        self.reset_states()
    
    def reset_states(self):
        """Reset model hidden states for new session."""
        self.model.reset_states()
        self.speech_start_time = None
        self.last_speech_time = None
        self.cumulative_speech_ms = 0
        self.cumulative_silence_ms = 0
        # Reset hysteresis state
        self.is_currently_speaking = False
        self.consecutive_speech_chunks = 0
        self.consecutive_silence_chunks = 0
        self.recent_probabilities = []
    
    def process(self, audio_chunk: np.ndarray) -> dict:
        """
        Process audio chunk and return VAD result.
        
        Args:
            audio_chunk: numpy array of float32 audio samples [-1.0, 1.0]
                        or int16 samples [-32768, 32767]
        
        Returns:
            {
                "is_speech": bool,
                "probability": float,
                "speech_duration_ms": float,
                "silence_duration_ms": float,
                "latency_ms": float
            }
        """
        start_time = time.time()
        
        # Convert to float32 if needed
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Ensure values are in [-1, 1]
        if np.abs(audio_chunk).max() > 1.0:
            audio_chunk = audio_chunk / np.abs(audio_chunk).max()
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk)
        
        # Get speech probability
        raw_prob = self.model(audio_tensor, self.sample_rate).item()

        # Smooth probability with weighted rolling window (newer samples weighted higher)
        self.recent_probabilities.append(raw_prob)
        if len(self.recent_probabilities) > self.smoothing_window:
            self.recent_probabilities.pop(0)

        # Weighted average: newer samples have more weight
        # For 2 samples: weights = [0.4, 0.6] (40% old, 60% new)
        if len(self.recent_probabilities) == 2:
            speech_prob = 0.4 * self.recent_probabilities[0] + 0.6 * self.recent_probabilities[1]
        else:
            speech_prob = sum(self.recent_probabilities) / len(self.recent_probabilities)

        # Apply hysteresis for stable speech detection
        # This prevents rapid on/off switching during natural speech variations
        if self.is_currently_speaking:
            # Currently speaking - need lower threshold to stop
            raw_is_speech = speech_prob >= self.threshold_off
            if not raw_is_speech:
                self.consecutive_silence_chunks += 1
                self.consecutive_speech_chunks = 0
                # Only transition to silence after N consecutive silent chunks
                if self.consecutive_silence_chunks >= self.min_silence_chunks:
                    self.is_currently_speaking = False
            else:
                self.consecutive_silence_chunks = 0
                self.consecutive_speech_chunks += 1
        else:
            # Currently silent - need higher threshold to start
            raw_is_speech = speech_prob >= self.threshold_on
            if raw_is_speech:
                self.consecutive_speech_chunks += 1
                self.consecutive_silence_chunks = 0
                # Only transition to speaking after N consecutive speech chunks
                if self.consecutive_speech_chunks >= self.min_speech_chunks:
                    self.is_currently_speaking = True
            else:
                self.consecutive_speech_chunks = 0
                self.consecutive_silence_chunks += 1

        # Use the stable hysteresis state for output
        is_speech = self.is_currently_speaking

        # Calculate chunk duration
        chunk_duration_ms = (len(audio_chunk) / self.sample_rate) * 1000

        # Update timing state
        current_time = time.time()

        if is_speech:
            if self.speech_start_time is None:
                self.speech_start_time = current_time
            self.last_speech_time = current_time
            self.cumulative_speech_ms += chunk_duration_ms
            self.cumulative_silence_ms = 0
        else:
            if self.last_speech_time is not None:
                self.cumulative_silence_ms += chunk_duration_ms

        latency_ms = (time.time() - start_time) * 1000

        return {
            "is_speech": is_speech,
            "probability": speech_prob,
            "raw_probability": raw_prob,  # Raw unsmoothed probability
            "speech_duration_ms": self.cumulative_speech_ms,
            "silence_duration_ms": self.cumulative_silence_ms,
            "latency_ms": latency_ms,
        }
    
    def get_speech_duration_ms(self) -> float:
        """Get total speech duration in current utterance."""
        return self.cumulative_speech_ms
    
    def get_silence_duration_ms(self) -> float:
        """Get silence duration since last speech."""
        return self.cumulative_silence_ms

