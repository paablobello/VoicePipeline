"""
Turn Detection - Hybrid approach
Combines VAD, silence duration, and semantic signals.

State-of-the-art implementation using:
1. EOU (End of Utterance) - Text-based analysis
2. Smart Turn v3 - Audio-based ML model (Whisper encoder + classifier)

Based on research from LiveKit, Pipecat, and Deepgram.
"""
import time
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from .eou_detector import get_eou_detector, EOUResult
from .backchannel import get_backchannel_detector


@dataclass
class TurnState:
    """State for turn detection."""
    silence_duration_ms: float = 0
    speech_duration_ms: float = 0
    is_speaking: bool = False
    turn_start_time: Optional[float] = None
    last_eou_result: Optional[EOUResult] = None
    audio_buffer: List[bytes] = field(default_factory=list)  # For Smart Turn
    last_smart_turn_time: float = 0  # Debounce Smart Turn calls
    last_smart_turn_result: Optional[bool] = None  # Cache result


class TurnDetector:
    """
    State-of-the-art turn-taking detection.
    
    Combines multiple signals:
    1. VAD (Voice Activity Detection) - silence duration
    2. EOU (End of Utterance) - text-based semantic analysis
    3. Smart Turn v3 - audio-based ML model (optional, most accurate)
    4. Adaptive timeout based on speech patterns
    
    Smart Turn v3 uses Whisper encoder to analyze prosody, intonation,
    and speech patterns - much more accurate than text-only analysis.
    """
    
    def __init__(
        self,
        base_silence_threshold_ms: float = 700,   # Lowered from 900: faster response
        min_speech_duration_ms: float = 150,      # Lowered from 300: capture short phrases
        max_silence_threshold_ms: float = 1500,   # Lowered from 1800: don't wait too long
        min_silence_threshold_ms: float = 400,    # Lowered from 500: faster response
        absolute_max_silence_ms: float = 2500,    # Lowered from 3000: force turn end sooner
        # NEW: Fast path for short complete responses
        short_response_threshold_ms: float = 350, # Lowered from 500: faster for "sí", "ok"
        use_smart_turn: bool = True,
        sample_rate: int = 16000,
    ):
        self.base_silence_threshold_ms = base_silence_threshold_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_silence_threshold_ms = max_silence_threshold_ms
        self.min_silence_threshold_ms = min_silence_threshold_ms
        self.absolute_max_silence_ms = absolute_max_silence_ms
        self.short_response_threshold_ms = short_response_threshold_ms
        self.use_smart_turn = use_smart_turn
        self.sample_rate = sample_rate

        # Short complete responses - process FAST (500ms)
        self.short_complete_responses = {
            'sí', 'si', 'no', 'ok', 'okay', 'vale', 'bueno', 'bien',
            'claro', 'perfecto', 'genial', 'exacto', 'correcto',
            'gracias', 'adiós', 'chao', 'hola', 'ajá', 'aja',
            'entendido', 'de acuerdo', 'ya', 'listo', 'dale',
            'sí perfecto', 'ok perfecto', 'vale perfecto',
        }

        # Adaptive parameters
        self.speech_rate_factor = 1.0
        self.recent_turn_durations: list = []
        
        # Smart Turn detector (lazy loaded)
        self._smart_turn = None
        self._smart_turn_available = use_smart_turn
        
        self.state = TurnState()
    
    async def initialize_smart_turn(self):
        """Initialize Smart Turn v3 model (lazy loading)."""
        if not self.use_smart_turn or self._smart_turn is not None:
            return
        
        try:
            from .smart_turn import SmartTurnDetector
            self._smart_turn = SmartTurnDetector(sample_rate=self.sample_rate)
            await self._smart_turn.initialize()
            print("  ✅ Smart Turn v3 enabled for turn detection")
        except Exception as e:
            print(f"  ⚠️ Smart Turn unavailable, using fallback: {e}")
            self._smart_turn_available = False
    
    def add_audio(self, audio_bytes: bytes):
        """Add audio to buffer for Smart Turn analysis."""
        self.state.audio_buffer.append(audio_bytes)
        # Keep last 10 seconds max
        max_chunks = int(10 * self.sample_rate / 320)  # 320 samples per 20ms chunk
        if len(self.state.audio_buffer) > max_chunks:
            self.state.audio_buffer = self.state.audio_buffer[-max_chunks:]
    
    def reset(self):
        """Reset state for new turn."""
        if self.state.speech_duration_ms > 0:
            self.recent_turn_durations.append(self.state.speech_duration_ms)
            # Keep only last 5 turns for adaptation
            self.recent_turn_durations = self.recent_turn_durations[-5:]
            self._update_speech_rate_factor()
        
        self.state = TurnState()
        self.state.audio_buffer = []
    
    def _update_speech_rate_factor(self):
        """
        Adapt threshold based on recent turn patterns.
        Fast talkers (short turns) → shorter threshold
        Slow talkers (long turns) → longer threshold
        """
        if len(self.recent_turn_durations) < 2:
            return
        
        avg_turn_duration = sum(self.recent_turn_durations) / len(self.recent_turn_durations)
        
        # Normalize: 2000ms is "average" turn duration
        self.speech_rate_factor = min(1.5, max(0.7, avg_turn_duration / 2000))
    
    def get_adaptive_threshold(self) -> float:
        """Calculate adaptive silence threshold."""
        threshold = self.base_silence_threshold_ms * self.speech_rate_factor
        # Clamp between min and max, but never exceed absolute max
        threshold = max(self.min_silence_threshold_ms, threshold)
        threshold = min(self.max_silence_threshold_ms, threshold)
        threshold = min(self.absolute_max_silence_ms - 100, threshold)  # Leave margin
        return threshold
    
    async def process(self, vad_result: dict, partial_transcript: str = "") -> dict:
        """
        Process VAD result and determine if turn is complete.
        Uses Smart Turn v3 (audio) + EOU (text) for maximum accuracy.
        
        Args:
            vad_result: Output from VAD.process()
            partial_transcript: Current partial transcript (for semantic analysis)
        
        Returns:
            {
                "is_turn_complete": bool,
                "confidence": float,
                "reason": str,
                "adaptive_threshold_ms": float,
                "latency_ms": float
            }
        """
        start_time = time.time()
        
        is_speech = vad_result["is_speech"]
        speech_duration_ms = vad_result["speech_duration_ms"]
        silence_duration_ms = vad_result["silence_duration_ms"]

        # Update state
        if is_speech:
            if not self.state.is_speaking:
                self.state.is_speaking = True
                self.state.turn_start_time = time.time()
            self.state.speech_duration_ms = speech_duration_ms
            self.state.silence_duration_ms = 0
        else:
            if self.state.is_speaking:
                self.state.silence_duration_ms = silence_duration_ms
        
        # Decision logic
        is_turn_complete = False
        confidence = 0.0
        reason = "listening"
        adaptive_threshold = self.get_adaptive_threshold()
        smart_turn_used = False
        
        if self.state.is_speaking and self.state.silence_duration_ms > 0:
            # User was speaking and now silent
            
            # SAFETY: Force turn end if absolute max silence exceeded
            if self.state.silence_duration_ms >= self.absolute_max_silence_ms:
                if self.state.speech_duration_ms >= self.min_speech_duration_ms:
                    is_turn_complete = True
                    confidence = 0.95
                    reason = "absolute_timeout"
            
            # === FAST PATH: Short complete responses ===
            # For "sí", "ok", "perfecto" etc. - respond in 500ms
            if not is_turn_complete and partial_transcript:
                normalized = partial_transcript.strip().lower()
                # Remove punctuation for matching
                normalized_clean = ''.join(c for c in normalized if c.isalnum() or c.isspace())

                if normalized_clean in self.short_complete_responses:
                    if self.state.silence_duration_ms >= self.short_response_threshold_ms:
                        is_turn_complete = True
                        confidence = 0.95
                        reason = f"short_response_fast:{normalized_clean}"
                        print(f"  ⚡ FAST PATH: '{normalized_clean}' → complete in {self.state.silence_duration_ms:.0f}ms")

            # === STANDARD PATH: Longer utterances ===
            # Check minimum silence before running expensive models
            min_silence_for_analysis = 300  # ms - lowered for faster response

            if not is_turn_complete and self.state.silence_duration_ms >= min_silence_for_analysis:
                # Try Smart Turn v3 first (audio-based, most accurate)
                # DEBOUNCE: Only run Smart Turn every 250ms to balance speed vs CPU
                current_time = time.time()
                smart_turn_debounce_ms = 250
                time_since_last = (current_time - self.state.last_smart_turn_time) * 1000
                
                if self._smart_turn_available and self._smart_turn and self.state.audio_buffer:
                    # Use cached result if within debounce window
                    if time_since_last < smart_turn_debounce_ms and self.state.last_smart_turn_result is not None:
                        smart_turn_used = True
                        if self.state.last_smart_turn_result:
                            if self.state.speech_duration_ms >= self.min_speech_duration_ms:
                                is_turn_complete = True
                                confidence = 0.8
                                reason = "smart_turn:cached_complete"
                        else:
                            reason = "smart_turn:cached_incomplete"
                    else:
                        # Run Smart Turn analysis
                        smart_result = await self._analyze_with_smart_turn()
                        if smart_result:
                            self.state.last_smart_turn_time = current_time
                            self.state.last_smart_turn_result = smart_result.is_complete
                            smart_turn_used = True
                            
                            if smart_result.is_complete:
                                if self.state.speech_duration_ms >= self.min_speech_duration_ms:
                                    is_turn_complete = True
                                    confidence = smart_result.probability
                                    reason = f"smart_turn:complete({smart_result.probability:.2f})"
                                else:
                                    reason = "insufficient_speech"
                            else:
                                reason = f"smart_turn:incomplete({smart_result.probability:.2f})"
                                # Smart Turn says wait - extend threshold (but respect absolute max)
                                adaptive_threshold = min(
                                    self.absolute_max_silence_ms - 100,
                                    max(adaptive_threshold, self.base_silence_threshold_ms * 1.5)
                                )
                
                # Fallback: EOU text-based analysis
                if not smart_turn_used:
                    eou = get_eou_detector()
                    eou_result = eou.analyze(
                        transcript=partial_transcript,
                        silence_duration_ms=self.state.silence_duration_ms,
                        speech_duration_ms=self.state.speech_duration_ms,
                    )
                    self.state.last_eou_result = eou_result
                    
                    if eou_result.is_complete:
                        if self.state.speech_duration_ms >= self.min_speech_duration_ms:
                            is_turn_complete = True
                            confidence = eou_result.confidence
                            reason = f"eou:{eou_result.reason}"
                        else:
                            reason = "insufficient_speech"
                    else:
                        reason = f"eou_waiting:{eou_result.reason}"
                        # Extend threshold but never exceed absolute max
                        suggested_threshold = self.state.silence_duration_ms + eou_result.suggested_wait_ms
                        adaptive_threshold = min(
                            self.absolute_max_silence_ms - 100,  # Leave margin
                            max(adaptive_threshold, suggested_threshold)
                        )
            else:
                reason = "waiting_min_silence"
                
        elif self.state.is_speaking:
            reason = "speaking"
        
        # Check if this is a backchannel (should not complete turn)
        if is_turn_complete and partial_transcript:
            backchannel = get_backchannel_detector()
            if backchannel.is_likely_backchannel(partial_transcript):
                is_turn_complete = False
                reason = f"backchannel_detected:{partial_transcript}"
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Debug logging - only log when silence is detected (key diagnostic)
        if self.state.silence_duration_ms > 0 and self.state.speech_duration_ms > 0:
            # Log every 200ms during silence
            if int(self.state.silence_duration_ms) % 200 < 32:
                print(f"  🔍 Turn: silence={self.state.silence_duration_ms:.0f}ms, speech={self.state.speech_duration_ms:.0f}ms, reason={reason}, complete={is_turn_complete}")
        
        return {
            "is_turn_complete": is_turn_complete,
            "confidence": confidence,
            "reason": reason,
            "adaptive_threshold_ms": adaptive_threshold,
            "speech_duration_ms": self.state.speech_duration_ms,
            "silence_duration_ms": self.state.silence_duration_ms,
            "smart_turn_used": smart_turn_used,
            "latency_ms": latency_ms,
        }
    
    async def _analyze_with_smart_turn(self):
        """Run Smart Turn v3 analysis on buffered audio."""
        try:
            # Convert audio bytes to numpy array
            audio_data = b''.join(self.state.audio_buffer)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Need at least 500ms of audio
            min_samples = int(0.5 * self.sample_rate)
            if len(audio_array) < min_samples:
                return None
            
            result = await self._smart_turn.analyze(audio_array)
            return result
            
        except Exception as e:
            print(f"  ⚠️ Smart Turn analysis failed: {e}")
            return None
    
    def _check_semantic_completeness(self, transcript: str) -> float:
        """
        Check if transcript appears semantically complete.
        Returns confidence score [0.0 - 1.0]
        
        Simple heuristics - in production could use NLP model.
        """
        if not transcript or not transcript.strip():
            return 0.0
        
        transcript = transcript.strip()
        
        # Ends with sentence-ending punctuation
        if transcript[-1] in '.?!¿¡':
            return 1.0
        
        # Has minimum word count
        words = transcript.split()
        if len(words) < 2:
            return 0.2
        elif len(words) < 4:
            return 0.5
        else:
            return 0.7

