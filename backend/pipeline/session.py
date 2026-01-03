"""
Voice Session Manager
Orchestrates the full voice AI pipeline with all optimizations.
"""
import asyncio
import numpy as np
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional

from .vad import SileroVAD
from .turn_detector import TurnDetector
from .stt import DeepgramSTT
from .llm import LLMProvider
from .tts import CartesiaTTS
from .tts_elevenlabs import ElevenLabsTTS
from .tts_cache import TTSCache
from .backchannel import get_backchannel_detector


@dataclass
class SessionMetrics:
    """Detailed metrics per session."""
    vad_latencies: List[float] = field(default_factory=list)
    turn_latencies: List[float] = field(default_factory=list)
    stt_latencies: List[float] = field(default_factory=list)
    llm_ttft: List[float] = field(default_factory=list)
    tts_ttfb: List[float] = field(default_factory=list)
    ttfa: List[float] = field(default_factory=list)  # Time to First Audio
    interruptions: int = 0
    turns: int = 0
    cache_hits: int = 0
    
    def get_summary(self) -> dict:
        """Get metrics summary with percentiles."""
        def percentile(data: List[float], p: int) -> float:
            if not data:
                return 0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        
        def avg(data: List[float]) -> float:
            return sum(data) / len(data) if data else 0
        
        return {
            "vad": {
                "p50": percentile(self.vad_latencies, 50),
                "p95": percentile(self.vad_latencies, 95),
                "avg": avg(self.vad_latencies),
            },
            "stt": {
                "p50": percentile(self.stt_latencies, 50),
                "p95": percentile(self.stt_latencies, 95),
                "avg": avg(self.stt_latencies),
            },
            "llm_ttft": {
                "p50": percentile(self.llm_ttft, 50),
                "p95": percentile(self.llm_ttft, 95),
                "avg": avg(self.llm_ttft),
            },
            "tts_ttfb": {
                "p50": percentile(self.tts_ttfb, 50),
                "p95": percentile(self.tts_ttfb, 95),
                "avg": avg(self.tts_ttfb),
            },
            "ttfa": {
                "p50": percentile(self.ttfa, 50),
                "p95": percentile(self.ttfa, 95),
                "avg": avg(self.ttfa),
            },
            "turns": self.turns,
            "interruptions": self.interruptions,
            "cache_hits": self.cache_hits,
        }


class VoiceSession:
    """
    Voice conversation session manager.
    
    Orchestrates the full pipeline:
    Audio → VAD → Turn Detection → STT → LLM → TTS → Audio
    
    Implements state-of-the-art optimizations:
    - Streaming at every layer
    - Sentence chunking for TTS
    - Interruption handling
    - TTS caching
    """
    
    def __init__(
        self,
        session_id: str,
        websocket,
        vad: SileroVAD,
        turn_detector: TurnDetector,
        tts_cache: TTSCache,
        # Provider configuration
        llm_provider: str = "groq",
        tts_provider: str = "cartesia",
    ):
        self.session_id = session_id
        self.websocket = websocket
        self.vad = vad
        self.turn_detector = turn_detector
        self.tts_cache = tts_cache
        
        # Provider settings
        self.llm_provider = llm_provider
        self.tts_provider = tts_provider
        
        # Create new VAD instance for this session (isolated state)
        self.session_vad = SileroVAD()
        
        # Create new turn detector for this session
        # It will share the Smart Turn model with the global detector
        self.session_turn_detector = TurnDetector()
        
        # Copy Smart Turn model reference from global detector (if pre-loaded)
        if turn_detector and turn_detector._smart_turn:
            self.session_turn_detector._smart_turn = turn_detector._smart_turn
            self.session_turn_detector._smart_turn_available = True
        
        # API clients (created per session with selected providers)
        self.stt: Optional[DeepgramSTT] = None
        self.llm: Optional[LLMProvider] = None
        self.tts = None  # Can be CartesiaTTS or ElevenLabsTTS
        
        # Queues
        self.audio_input_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.audio_output_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        # State
        self.conversation_history: List[Dict[str, str]] = []
        self.is_bot_speaking = False
        self.current_tasks: List[asyncio.Task] = []
        self._max_tasks = 20  # Max tasks before forced cleanup
        
        # False interruption recovery (like LiveKit)
        self.paused_response: Optional[str] = None  # Response that was interrupted
        self.false_interruption_timeout: float = 2.0  # Seconds to wait for user to continue

        # Interruption flag - checked frequently during TTS generation
        self._interrupt_flag: bool = False

        # Greeting phase flag - don't treat speech during greeting as interruption
        self._is_greeting: bool = False

        # Post-interruption cooldown - give user time to finish speaking after interrupting
        self._last_interruption_time: Optional[float] = None
        self._interruption_cooldown_ms: float = 800  # Wait 800ms after interruption before responding

        # False interruption recovery - prompt user if they're silent after interruption
        self._empty_turns_after_interruption: int = 0
        self._max_empty_turns_before_prompt: int = 2  # After 2 empty turns, ask if user is there

        # Transcript state - ACCUMULATE finals, don't overwrite
        self.partial_transcript = ""  # Current interim (overwrites)
        self.final_transcript = ""    # ACCUMULATED finals for current turn
        self.transcript_lock = asyncio.Lock()

        # Transcript accumulation bounds (Issue #3: prevent ghost text)
        self._max_transcript_chars = 500  # Max chars before truncating
        self._max_accumulation_time_sec = 15  # Max seconds to accumulate
        self._accumulation_start_time: Optional[float] = None

        # UtteranceEnd tracking - wait for Deepgram to signal it's done
        self._utterance_end_received = False
        self._last_stt_final_time: Optional[float] = None  # Track when last STT final arrived

        # Event-based transcript signaling (eliminates polling latency)
        self._transcript_event = asyncio.Event()

        # Audio pre-roll buffer (protected by lock to prevent race conditions)
        self._preroll_lock = asyncio.Lock()
        self._audio_preroll_buffer: List[bytes] = []
        self._preroll_sent: bool = False

        # Metrics
        self.metrics = SessionMetrics()
        self.turn_start_time: Optional[float] = None


    def _cleanup_completed_tasks(self):
        """Remove completed tasks from the list to prevent memory leaks."""
        if len(self.current_tasks) > self._max_tasks:
            # Remove all completed tasks
            self.current_tasks = [t for t in self.current_tasks if not t.done()]

    def _add_task(self, task: asyncio.Task):
        """Add a task to the list with automatic cleanup."""
        self._cleanup_completed_tasks()
        self.current_tasks.append(task)

    async def start(self):
        """Initialize session and connect to APIs."""
        print(f"  🎙️ Starting session {self.session_id}")
        print(f"     LLM: {self.llm_provider} | TTS: {self.tts_provider}")
        
        # Create API clients with selected providers
        self.stt = DeepgramSTT()
        self.llm = LLMProvider(provider=self.llm_provider)
        
        # Create TTS with selected provider
        if self.tts_provider == "elevenlabs":
            try:
                self.tts = ElevenLabsTTS(
                    model="flash_v2_5",       # Fast + supports language_code
                    stability=0.4,            # 35-40% for natural variation
                    similarity_boost=0.75,    # Keep at/below 75-80%
                    style=0.0,
                    language_code="es",       # Spanish pronunciation
                )
            except ValueError:
                print("  ⚠️ ElevenLabs not configured, falling back to Cartesia")
                self.tts_provider = "cartesia"
        
        if self.tts_provider == "cartesia":
            self.tts = CartesiaTTS(
                model="sonic-turbo",  # Fastest: 40ms TTFB (Phase 1 optimization)
                speed="normal",
                emotion=["positivity:low", "curiosity:lowest"]  # Valid levels: lowest, low, high, highest
            )
        
        # Initialize Smart Turn v3 only if not already loaded from global
        if not self.session_turn_detector._smart_turn:
            try:
                await self.session_turn_detector.initialize_smart_turn()
            except Exception as e:
                print(f"  ⚠️ Smart Turn init failed, using text-based EOU: {e}")
        else:
            print("  ✅ Smart Turn v3 (using pre-loaded model)")
        
        # Connect to APIs
        await self.stt.connect()
        await self.tts.connect()

        # Start STT receiver early so we don't miss any speech
        self._stt_task = asyncio.create_task(self._stt_receiver())
        self._add_task(self._stt_task)

        # Send greeting in background - don't block listening
        greeting_task = asyncio.create_task(self._send_greeting())
        self._add_task(greeting_task)

        print(f"  ✅ Session {self.session_id} ready - listening immediately")

    async def _send_greeting(self):
        """Send greeting in background without blocking the main loop."""
        self.is_bot_speaking = True
        self._is_greeting = True  # Don't treat user speech as interruption during greeting

        greeting_text = "¡Hola! ¿En qué puedo ayudarte?"
        print(f"  🎤 Generating greeting with {self.tts_provider}...")

        try:
            audio_chunks = 0
            async for chunk in self.tts.synthesize(greeting_text):
                # Check if user interrupted
                if self._interrupt_flag:
                    print("  ⚡ Greeting interrupted by user")
                    break
                await self.websocket.send_bytes(chunk)
                audio_chunks += 1

            print(f"  ✅ Greeting sent ({audio_chunks} chunks)")

            # Very brief pause, then ready for user
            await asyncio.sleep(0.1)

        except Exception as e:
            print(f"  ⚠️ Greeting error: {e}")
        finally:
            # CRITICAL: Clear any garbage transcripts that might have accumulated
            # from echo during the greeting
            async with self.transcript_lock:
                self.partial_transcript = ""
                self.final_transcript = ""
                self._accumulation_start_time = None  # Reset accumulation timer
            
            # Reset VAD and turn detector state
            self.session_vad.reset_states()
            self.session_turn_detector.reset()
            
            self._is_greeting = False  # Greeting done, now user speech IS interruption
            self.is_bot_speaking = False
            print("  🎧 Ready to listen - greeting complete (buffers cleared)")
    
    async def process(self) -> AsyncGenerator[dict, None]:
        """
        Main processing loop.
        
        Yields events:
        - {"type": "transcript", "role": "user"|"assistant", "text": "..."}
        - {"type": "llm_token", "token": "..."}
        - {"type": "audio", "data": bytes}
        - {"type": "metrics", "data": {...}}
        - {"type": "control", "action": "..."}
        """
        # STT receiver is already started in start() - don't duplicate

        try:
            while True:
                # Get audio from queue with timeout
                try:
                    audio_bytes = await asyncio.wait_for(
                        self.audio_input_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Convert to numpy
                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                
                # === STAGE 1: VAD ===
                start = time.time()
                vad_result = self.session_vad.process(audio_float)
                vad_latency = (time.time() - start) * 1000
                self.metrics.vad_latencies.append(vad_latency)
                
                # === SEND AUDIO TO STT WITH ROLLING PRE-ROLL BUFFER ===
                # The VAD hysteresis takes ~2 chunks (~64ms) to confirm speech.
                # Words spoken before raw_prob reaches threshold are lost.
                # Solution: ALWAYS keep a rolling buffer of last N chunks, send when speech confirmed.

                raw_prob = vad_result.get("raw_probability", vad_result["probability"])

                # Use lock to prevent race conditions on pre-roll buffer
                async with self._preroll_lock:
                    # ALWAYS add to rolling buffer when not speech
                    if not vad_result["is_speech"]:
                        self._audio_preroll_buffer.append(audio_bytes)
                        # Keep max 8 chunks (~256ms) - enough to capture a short word
                        if len(self._audio_preroll_buffer) > 8:
                            self._audio_preroll_buffer.pop(0)

                    if vad_result["is_speech"]:
                        # Only send rolling buffer when starting a NEW turn
                        is_new_turn = self.turn_start_time is None

                        if self._audio_preroll_buffer and not self._preroll_sent and is_new_turn:
                            for buffered_chunk in self._audio_preroll_buffer:
                                await self.stt.send_audio(buffered_chunk)
                            self._audio_preroll_buffer.clear()
                            self._preroll_sent = True

                        # Send current audio
                        await self.stt.send_audio(audio_bytes)
                    else:
                        # Reset preroll flag when silence returns (low probability)
                        if self._preroll_sent and raw_prob < 0.1:
                            self._preroll_sent = False
                            self._audio_preroll_buffer.clear()
                
                # === CHECK INTERRUPTION ===7
                # CRITICAL: If bot is speaking and user starts speaking, STOP immediately
                # EXCEPT during greeting - we want to capture the first message without clearing it
                if self.is_bot_speaking and vad_result["is_speech"]:
                    # During greeting, just let the greeting finish and capture speech
                    # Don't treat it as an interruption that clears transcript
                    if self._is_greeting:
                        # User is speaking during greeting - just capture it, don't interrupt
                        # The greeting will finish naturally and we'll process their message
                        pass
                    else:
                        # Normal interruption handling for non-greeting speech
                        raw_prob = vad_result.get("raw_probability", vad_result["probability"])

                        # Check if this is a real interruption vs backchannel
                        backchannel_detector = get_backchannel_detector()

                        # Get current transcript to check for backchannels
                        async with self.transcript_lock:
                            current_text = self.partial_transcript.strip()

                        is_backchannel = backchannel_detector.is_likely_backchannel(current_text)

                        if is_backchannel:
                            # User is just acknowledging ("ok", "ajá") - NOT an interruption
                            # IMPORTANT: Clear accumulated transcript to prevent "ghost text"
                            print(f"  💬 Backchannel detected: '{current_text}' - not interrupting, clearing transcript")
                            async with self.transcript_lock:
                                self.partial_transcript = ""
                                self.final_transcript = ""
                        elif vad_result["speech_duration_ms"] > 150 and raw_prob > 0.7:
                            # BALANCED interruption detection:
                            # Require BOTH sustained speech (150ms+) AND high probability
                            # to avoid false positives from coughs/throat noises
                            print(f"  🛑 INTERRUPTION! Speech: {vad_result['speech_duration_ms']:.0f}ms, prob: {raw_prob:.2f}")
                            await self._handle_interruption()
                            yield {"type": "interruption"}
                            continue
                
                # === STAGE 2: Turn Detection ===
                # Add audio to turn detector buffer for Smart Turn v3
                self.session_turn_detector.add_audio(audio_bytes)
                
                start = time.time()
                turn_result = await self.session_turn_detector.process(
                    vad_result,
                    self.partial_transcript
                )
                turn_latency = (time.time() - start) * 1000
                self.metrics.turn_latencies.append(turn_latency)
                
                # Mark turn start when speech detected and bot is NOT speaking
                if vad_result["is_speech"] and not self.is_bot_speaking:
                    if self.turn_start_time is None:
                        self.turn_start_time = time.time()
                        # Reset UtteranceEnd tracking for new turn
                        self._utterance_end_received = False
                        self._last_stt_final_time = None
                        # Reset transcript event for new turn
                        self._transcript_event.clear()
                        # Clear transcripts at the START of a new turn
                        # This prevents old transcripts from leaking into new turns
                        async with self.transcript_lock:
                            self.partial_transcript = ""
                            self.final_transcript = ""
                            self._accumulation_start_time = None  # Reset accumulation timer
                
                # === PROCESS COMPLETE TURN ===
                if turn_result["is_turn_complete"] and not self.is_bot_speaking:
                    # Check if we're in post-interruption cooldown
                    # This gives the user time to finish speaking after interrupting
                    if self._last_interruption_time is not None:
                        elapsed_since_interruption = (time.time() - self._last_interruption_time) * 1000
                        if elapsed_since_interruption < self._interruption_cooldown_ms:
                            # Still in cooldown - don't process this turn yet
                            print(f"  ⏸️ Post-interruption cooldown: {elapsed_since_interruption:.0f}ms / {self._interruption_cooldown_ms}ms - waiting")
                            continue
                        else:
                            # Cooldown expired - clear the flag
                            self._last_interruption_time = None

                    # Smart Turn completes BEFORE Deepgram sends transcript
                    # We must WAIT for transcript instead of discarding

                    # First, flush STT to get any pending transcript
                    await self._flush_stt()

                    # Check if we already have transcript
                    async with self.transcript_lock:
                        has_transcript = bool(self.partial_transcript.strip() or self.final_transcript.strip())

                    # If no transcript yet, wait for event signal (much faster than polling!)
                    if not has_transcript:
                        self._transcript_event.clear()
                        try:
                            # Wait up to 1 second for transcript event
                            await asyncio.wait_for(self._transcript_event.wait(), timeout=1.0)
                            print(f"  ⚡ Transcript arrived via event signal")
                        except asyncio.TimeoutError:
                            print(f"  ⚠️ No transcript after 1s event wait")

                    # Check again for transcript
                    async with self.transcript_lock:
                        has_transcript = bool(self.partial_transcript.strip() or self.final_transcript.strip())

                    if has_transcript:
                        self.metrics.turns += 1

                        # OPTIMIZED: Check transcript completion to minimize wait time
                        async with self.transcript_lock:
                            current_text = (self.final_transcript or self.partial_transcript).strip()

                        # Smart completion detection using Deepgram punctuation
                        has_final_punctuation = current_text.rstrip().endswith(('.', '?', '!'))
                        has_trailing_comma = current_text.rstrip().endswith(',')
                        is_very_short = len(current_text.split()) <= 3
                        word_count = len(current_text.split())

                        # FAST PATH: If transcript has final punctuation and enough words,
                        # start LLM immediately - no need to wait for UtteranceEnd
                        if has_final_punctuation and word_count >= 3:
                            # Give Deepgram just 50ms to send any trailing text
                            await asyncio.sleep(0.05)
                            async with self.transcript_lock:
                                current_text = (self.final_transcript or self.partial_transcript).strip()
                            print(f"  ⚡ FAST PATH: Complete sentence detected, starting LLM immediately")
                            max_utterance_wait_ms = 0  # Skip the wait loop
                            waited_for_utterance = 0
                        else:
                            # Need to wait for more transcript
                            max_utterance_wait_ms = 400  # Max wait for UtteranceEnd
                            stt_settle_time_ms = 150  # Reduced from 200ms
                            waited_for_utterance = 0

                            # Use EOUDetector for intelligent phrase completeness analysis
                            # It uses regex patterns to detect prepositions, conjunctions,
                            # thinking words, and other incomplete markers
                            from .eou_detector import get_eou_detector
                            eou = get_eou_detector()
                            eou_result = eou.analyze(
                                transcript=current_text,
                                silence_duration_ms=0,  # We're just checking text patterns
                                speech_duration_ms=1000,  # Assume decent speech duration
                            )

                            # EOUDetector returns incompleteness score 0-1
                            # High incompletion score = clearly incomplete phrase
                            incompletion_score = eou._check_incompletion(current_text.lower().strip())

                            is_clearly_incomplete = (
                                has_trailing_comma or
                                (not has_final_punctuation and is_very_short) or
                                incompletion_score >= 0.5  # Any incomplete marker detected (preposition, conjunction, etc.)
                            )
                            is_possibly_incomplete = (
                                not has_final_punctuation and
                                not is_clearly_incomplete and
                                incompletion_score >= 0.2  # Very weak incomplete signal
                            )

                            if is_clearly_incomplete:
                                # User is clearly mid-thought - wait longer for continuation
                                max_utterance_wait_ms = 1500
                                print(f"  🤔 Incomplete phrase (score={incompletion_score:.2f}): '{current_text[:50]}...' - waiting {max_utterance_wait_ms}ms")
                            elif is_possibly_incomplete:
                                # Weak incomplete signal - medium wait
                                max_utterance_wait_ms = 800
                                print(f"  ⏳ Possibly incomplete (score={incompletion_score:.2f}): '{current_text[:40]}...' - waiting {max_utterance_wait_ms}ms")
                            else:
                                # Has final punctuation - but still wait enough for STT to settle
                                # 300ms was too short - STT finals can arrive 400-600ms after audio
                                max_utterance_wait_ms = 500
                                print(f"  ✓ Phrase detected: '{current_text[:40]}...' - waiting {max_utterance_wait_ms}ms")

                        while waited_for_utterance < max_utterance_wait_ms:
                            # Check if STT is still sending finals (wait for it to settle)
                            # This takes priority over UtteranceEnd because Deepgram can send
                            # UtteranceEnd on brief pauses while user is still speaking
                            if self._last_stt_final_time:
                                time_since_last_final = (time.time() - self._last_stt_final_time) * 1000
                                if time_since_last_final < stt_settle_time_ms:
                                    # STT recently sent a final - wait for more
                                    await asyncio.sleep(0.05)
                                    waited_for_utterance += 50
                                    continue

                            # Check if UtteranceEnd received AND no recent STT activity
                            if self._utterance_end_received:
                                # Wait after UtteranceEnd to catch late transcripts
                                # Longer wait if phrase doesn't end with punctuation
                                post_utterance_wait = 200 if has_final_punctuation else 350
                                await asyncio.sleep(post_utterance_wait / 1000)
                                waited_for_utterance += post_utterance_wait

                                # Check if new transcript arrived during the wait
                                async with self.transcript_lock:
                                    new_text = (self.final_transcript or self.partial_transcript).strip()

                                if new_text != current_text:
                                    # More speech came after UtteranceEnd - keep waiting
                                    current_text = new_text
                                    self._utterance_end_received = False  # Reset, wait for next one
                                    # Re-evaluate punctuation for next iteration
                                    has_final_punctuation = current_text.rstrip().endswith(('.', '?', '!'))
                                    print(f"  📝 Speech continued after UtteranceEnd: '{current_text[:40]}...'")
                                    continue
                                else:
                                    print(f"  ✅ UtteranceEnd confirmed after {waited_for_utterance}ms")
                                    break

                            # Check if new transcript arrived and re-evaluate completeness
                            if waited_for_utterance > 150:  # Reduced from 300ms
                                async with self.transcript_lock:
                                    new_text = (self.final_transcript or self.partial_transcript).strip()
                                if new_text != current_text:
                                    # User continued speaking! Update and re-evaluate
                                    current_text = new_text
                                    print(f"  📝 User continued: '{current_text[:40]}...'")

                                    # Re-check if now complete
                                    has_final_punctuation = current_text.rstrip().endswith(('.', '?', '!'))
                                    has_trailing_comma = current_text.rstrip().endswith(',')
                                    is_very_short = len(current_text.split()) <= 3

                                    is_clearly_incomplete = (
                                        has_trailing_comma or
                                        (not has_final_punctuation and is_very_short)
                                    )

                                    # Only reduce wait if we have FINAL punctuation
                                    # Phrases without punctuation should keep waiting
                                    if has_final_punctuation and not is_clearly_incomplete:
                                        # Phrase is now complete with punctuation - reduce remaining wait
                                        max_utterance_wait_ms = min(max_utterance_wait_ms, waited_for_utterance + 300)
                                        print(f"  ✓ Phrase now complete with punctuation - reducing wait")

                            await asyncio.sleep(0.05)
                            waited_for_utterance += 50

                        # Log if we timed out without UtteranceEnd
                        if not self._utterance_end_received:
                            print(f"  ⚠️ No UtteranceEnd after {waited_for_utterance}ms - proceeding anyway")

                        # CRITICAL: Capture transcript atomically BEFORE processing
                        async with self.transcript_lock:
                            captured_final = self.final_transcript.strip()
                            captured_partial = self.partial_transcript.strip()
                            captured_transcript = captured_final if captured_final else captured_partial
                            self.partial_transcript = ""
                            self.final_transcript = ""
                            self._accumulation_start_time = None  # Reset accumulation timer
                            # Reset utterance tracking for next turn
                            self._utterance_end_received = False
                            self._last_stt_final_time = None

                        if captured_transcript:
                            # Reset empty turn counter - we got real speech
                            self._empty_turns_after_interruption = 0
                            async for event in self._process_turn(captured_transcript):
                                yield event
                        else:
                            print("  ⚠️ Turn complete but no transcript captured after wait")

                        # Reset state AFTER processing
                        self.session_turn_detector.reset()
                        self.session_vad.reset_states()
                        self.turn_start_time = None
                        # Reset preroll flag so next turn can use buffer
                        async with self._preroll_lock:
                            self._preroll_sent = False
                            self._audio_preroll_buffer.clear()
                    else:
                        # Still no transcript after waiting - likely false positive from Smart Turn
                        print(f"  ⚠️ Turn complete but no transcript after 1000ms event wait - ignoring")

                        # Track empty turns after interruption - might need to prompt user
                        if self._last_interruption_time is not None:
                            self._empty_turns_after_interruption += 1
                            print(f"  📊 Empty turns after interruption: {self._empty_turns_after_interruption}")

                            if self._empty_turns_after_interruption >= self._max_empty_turns_before_prompt:
                                # User seems to be silent after interruption - prompt them
                                print(f"  🔄 Prompting user after {self._empty_turns_after_interruption} empty turns")
                                self._empty_turns_after_interruption = 0
                                self._last_interruption_time = None

                                # Send a short prompt
                                prompt_text = "¿Sigues ahí? ¿En qué puedo ayudarte?"
                                self.is_bot_speaking = True
                                async for audio_chunk in self.tts.synthesize(prompt_text):
                                    await self.websocket.send_bytes(audio_chunk)
                                self.is_bot_speaking = False

                        self.session_turn_detector.reset()
                        self.session_vad.reset_states()
                        self.turn_start_time = None
                        # Reset preroll flag so next turn can use buffer
                        async with self._preroll_lock:
                            self._preroll_sent = False
                            self._audio_preroll_buffer.clear()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"  ❌ Process error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cancel background tasks
            for task in self.current_tasks:
                if not task.done():
                    task.cancel()
    
    async def _stt_receiver(self):
        """Background task to receive STT results."""
        try:
            async for result in self.stt.receive_transcripts():
                # Handle UtteranceEnd signal from Deepgram
                if result["type"] == "utterance_end":
                    self._utterance_end_received = True
                    print("  📍 Deepgram UtteranceEnd received")
                    continue

                # Handle SpeechStarted - reset utterance tracking
                if result["type"] == "speech_started":
                    self._utterance_end_received = False
                    continue

                if result["type"] == "transcript":
                    # CRITICAL: Only ignore transcriptions during greeting
                    # The microphone picks up the greeting audio (echo) and STT
                    # transcribes it as garbage like "Tal, Alcidha."
                    if self._is_greeting:
                        # Silently discard - don't accumulate echo/garbage
                        continue
                    
                    # NOTE: Do NOT discard during is_bot_speaking!
                    # The user might start talking just as the bot finishes,
                    # and we need to capture that. Interruption handling
                    # is done separately in the main loop.
                    
                    async with self.transcript_lock:
                        text = result["text"].strip()
                        if not text:
                            continue
                        
                        # Filter out very short garbage transcripts (likely echo/noise)
                        if len(text) < 2:
                            continue

                        if result.get("is_final"):
                            # Track when this STT final arrived (for late-chunk detection)
                            self._last_stt_final_time = time.time()

                            # Signal that transcript is available (eliminates polling latency)
                            self._transcript_event.set()

                            # Issue #3: Check accumulation bounds before adding
                            current_time = time.time()

                            # Start accumulation timer if not started
                            if self._accumulation_start_time is None:
                                self._accumulation_start_time = current_time

                            # Check if we've been accumulating too long (stale data)
                            time_elapsed = current_time - self._accumulation_start_time
                            if time_elapsed > self._max_accumulation_time_sec:
                                # Reset - this is likely ghost text from a previous turn
                                print(f"  🧹 Resetting accumulation (exceeded {self._max_accumulation_time_sec}s)")
                                self.final_transcript = text
                                self._accumulation_start_time = current_time
                            # Check character limit
                            elif len(self.final_transcript) + len(text) > self._max_transcript_chars:
                                # Truncate old content to make room
                                max_keep = self._max_transcript_chars - len(text) - 50
                                if max_keep > 0:
                                    self.final_transcript = "..." + self.final_transcript[-max_keep:] + " " + text
                                else:
                                    self.final_transcript = text
                                print(f"  ⚠️ Transcript truncated (exceeded {self._max_transcript_chars} chars)")
                            else:
                                # ACCUMULATE final transcripts (don't overwrite!)
                                # This handles cases where user pauses mid-sentence
                                # and Deepgram sends multiple "final" chunks
                                if self.final_transcript:
                                    # Add space between accumulated parts
                                    self.final_transcript = f"{self.final_transcript} {text}"
                                else:
                                    self.final_transcript = text

                            # Also update partial to show accumulated
                            self.partial_transcript = self.final_transcript
                            self.metrics.stt_latencies.append(result.get("latency_ms", 0))
                            print(f"  📝 STT final: \"{text}\" → accumulated: \"{self.final_transcript[:50]}...\"")
                        else:
                            # Interim results: show current + accumulated
                            if self.final_transcript:
                                self.partial_transcript = f"{self.final_transcript} {text}"
                            else:
                                self.partial_transcript = text
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"  ⚠️ STT receiver error: {e}")
    
    async def _process_turn(self, transcript: str = None) -> AsyncGenerator[dict, None]:
        """
        Process complete user turn: STT → LLM → TTS

        Args:
            transcript: Pre-captured transcript text. If None, will try to capture from state.
        """
        turn_start = time.time()

        # Use provided transcript or fallback to capturing from state
        if transcript is None:
            async with self.transcript_lock:
                transcript = self.final_transcript or self.partial_transcript
                transcript = transcript.strip() if transcript else ""

        if not transcript:
            print("  ⚠️ _process_turn called with empty transcript")
            return
        
        # Yield user transcript
        yield {
            "type": "transcript",
            "role": "user",
            "text": transcript,
        }
        
        # === LLM + TTS STREAMING ===
        self.is_bot_speaking = True
        self._interrupt_flag = False  # Reset interrupt flag at start of new turn
        full_response = ""
        first_audio_sent = False
        tts_start_time = None
        total_audio_bytes = 0  # Track audio for duration estimation

        sentence_count = 0

        # OPTIMIZED Sentence batching for minimal latency:
        # - Sentence 1: send immediately (lowest latency)
        # - Sentence 2: send immediately (no waiting)
        # - Sentence 3+: batch with timeout (smoother prosody)
        sentence_buffer = []
        last_sentence_time = None
        BATCH_TIMEOUT_MS = 150  # Max wait before sending buffered sentences
        BATCH_START_FROM = 3   # Start batching from sentence 3 onwards

        async def synthesize_text(text_to_speak: str) -> AsyncGenerator[dict, None]:
            """Helper to synthesize text and yield audio chunks."""
            nonlocal first_audio_sent, total_audio_bytes, tts_start_time

            sentence_tts_start = time.time()
            if tts_start_time is None:
                tts_start_time = sentence_tts_start

            first_tts_chunk = True
            async for audio_chunk in self.tts.synthesize(text_to_speak):
                # Check interruption on EVERY chunk
                if not self.is_bot_speaking or self._interrupt_flag:
                    print("  ⚡ TTS stopped mid-synthesis due to interruption")
                    break

                # Track TTS TTFB for first chunk
                if first_tts_chunk:
                    tts_ttfb = (time.time() - sentence_tts_start) * 1000
                    self.metrics.tts_ttfb.append(tts_ttfb)
                    first_tts_chunk = False

                total_audio_bytes += len(audio_chunk)

                if not first_audio_sent:
                    ttfa = (time.time() - turn_start) * 1000
                    self.metrics.ttfa.append(ttfa)
                    first_audio_sent = True
                    yield {"type": "ttfa", "ms": ttfa}

                yield {"type": "audio", "data": audio_chunk}

        async def flush_sentence_buffer() -> AsyncGenerator[dict, None]:
            """Flush any buffered sentences."""
            nonlocal sentence_buffer, last_sentence_time
            if sentence_buffer and self.is_bot_speaking and not self._interrupt_flag:
                combined_text = " ".join(sentence_buffer)
                sentence_buffer.clear()
                last_sentence_time = None
                async for event in synthesize_text(combined_text):
                    yield event

        try:
            async for llm_event in self.llm.generate(
                transcript,
                self.conversation_history
            ):
                # Check for interruption - use both flags for redundancy
                if not self.is_bot_speaking or self._interrupt_flag:
                    print("  ⚡ LLM generation stopped due to interruption")
                    break

                if llm_event["type"] == "token":
                    full_response += llm_event["token"]
                    yield {"type": "llm_token", "token": llm_event["token"]}

                    # Check if buffer timeout expired (flush early)
                    if sentence_buffer and last_sentence_time:
                        elapsed_ms = (time.time() - last_sentence_time) * 1000
                        if elapsed_ms >= BATCH_TIMEOUT_MS:
                            async for event in flush_sentence_buffer():
                                yield event

                elif llm_event["type"] == "ttft":
                    self.metrics.llm_ttft.append(llm_event["ttft_ms"])

                elif llm_event["type"] == "sentence":
                    # Check interruption before starting TTS
                    if self._interrupt_flag:
                        print("  ⚡ TTS skipped due to interruption")
                        break

                    sentence_count += 1
                    sentence_text = llm_event["text"]

                    # Sentences 1 & 2: send immediately for minimal latency
                    if sentence_count < BATCH_START_FROM:
                        # First flush any buffered content
                        async for event in flush_sentence_buffer():
                            yield event
                        # Then send this sentence
                        async for event in synthesize_text(sentence_text):
                            yield event
                    else:
                        # Sentence 3+: buffer with timeout for smoother prosody
                        sentence_buffer.append(sentence_text)
                        if last_sentence_time is None:
                            last_sentence_time = time.time()

                        # If buffer has 2 sentences, flush immediately
                        if len(sentence_buffer) >= 2:
                            async for event in flush_sentence_buffer():
                                yield event

                elif llm_event["type"] == "error":
                    yield {"type": "error", "message": llm_event["error"]}
                    break

            # Process any remaining sentences in the buffer
            async for event in flush_sentence_buffer():
                yield event

            # CRITICAL: Save conversation history BEFORE playback wait
            # This ensures context is preserved even if user interrupts during playback
            if full_response:
                self.conversation_history.append({
                    "role": "user",
                    "content": transcript
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                # Keep history limited
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]

            # Estimate audio playback duration and wait
            # Cartesia outputs 24kHz 16-bit mono PCM = 48000 bytes/second
            if total_audio_bytes > 0 and self.is_bot_speaking:
                estimated_duration_sec = total_audio_bytes / 48000
                # Wait for most of the audio to play (but not all, to allow early interaction)
                # We wait ~70% of the duration to allow interruptions near the end
                wait_time = min(estimated_duration_sec * 0.7, 5.0)  # Cap at 5 seconds
                if wait_time > 0.5:
                    print(f"  ⏳ Audio duration: {estimated_duration_sec:.1f}s, waiting {wait_time:.1f}s")
                    # Wait in small chunks while ACTIVELY checking for interruptions
                    elapsed = 0
                    chunks_processed = 0
                    consecutive_speech_chunks = 0  # Track consecutive speech for more reliable detection

                    while elapsed < wait_time and self.is_bot_speaking:
                        # Process ALL available audio chunks (drain the queue)
                        while not self.audio_input_queue.empty():
                            try:
                                audio_bytes = self.audio_input_queue.get_nowait()
                                chunks_processed += 1

                                # Process with VAD to detect user speech
                                audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                                audio_float = audio_chunk.astype(np.float32) / 32768.0
                                vad_result = self.session_vad.process(audio_float)
                                raw_prob = vad_result.get("raw_probability", vad_result["probability"])

                                # Track consecutive speech chunks (more reliable than is_speech with hysteresis)
                                # Use higher threshold to avoid false positives from throat noises
                                if raw_prob > 0.6:
                                    consecutive_speech_chunks += 1
                                else:
                                    consecutive_speech_chunks = 0

                                # ALWAYS send audio to STT during playback wait
                                await self.stt.send_audio(audio_bytes)

                                # BALANCED interruption detection:
                                # Require sustained speech to avoid false positives from coughs/sneezes
                                # A sneeze is loud (high prob) but very short (~100-200ms)
                                # Real speech interruption should be at least 400ms (12+ chunks at 32ms each)
                                is_real_interruption = (
                                    raw_prob > 0.75 and consecutive_speech_chunks >= 12  # ~400ms of speech
                                ) or consecutive_speech_chunks >= 15  # ~500ms very sustained

                                if not self._is_greeting and is_real_interruption:
                                    print(f"  🛑 INTERRUPTION during playback! prob: {raw_prob:.2f}, consecutive: {consecutive_speech_chunks}")
                                    await self._handle_interruption()
                                    yield {"type": "interruption"}
                                    return  # Exit _process_turn immediately
                            except Exception:
                                break  # Queue empty or error

                        await asyncio.sleep(0.03)  # 30ms sleep between checks
                        elapsed += 0.03

                    if chunks_processed > 0:
                        print(f"  📊 Processed {chunks_processed} audio chunks during playback wait")
            
            self.is_bot_speaking = False

            # CRITICAL: Clear any "ghost" transcripts that accumulated while bot was speaking
            # This prevents text spoken during bot speech from appearing as a new turn
            async with self.transcript_lock:
                if self.partial_transcript or self.final_transcript:
                    print(f"  🧹 Clearing ghost transcript after bot finished: '{self.partial_transcript[:30] if self.partial_transcript else self.final_transcript[:30]}...'")
                    self.partial_transcript = ""
                    self.final_transcript = ""
                    self._accumulation_start_time = None  # Reset accumulation timer

            # Note: Conversation history is saved earlier (before playback wait)
            # to ensure context is preserved even if interrupted

            # Yield assistant transcript
            yield {
                "type": "transcript",
                "role": "assistant",
                "text": full_response,
            }
            
            # Yield metrics
            yield {
                "type": "metrics",
                "data": self.metrics.get_summary()
            }
            
        except Exception as e:
            print(f"  ❌ Turn processing error: {e}")
            self.is_bot_speaking = False
            yield {"type": "error", "message": str(e)}
    
    async def _flush_stt(self):
        """
        Flush STT buffer to get final transcript.
        This is CRITICAL to prevent transcript mixing between turns
        and to reset Deepgram's state after interruptions.
        """
        if self.stt and self.stt.is_connected:
            try:
                # Send Finalize command to Deepgram to flush its buffer
                import json
                if self.stt.ws:
                    await self.stt.ws.send(json.dumps({"type": "Finalize"}))
                    print("  🔄 STT flushed (Finalize sent)")
                    # Wait for Deepgram to process the finalize (150ms is sufficient)
                    await asyncio.sleep(0.15)
                    # Drain any pending results from the queue to clear old data
                    drained = 0
                    while not self.stt.results_queue.empty():
                        try:
                            self.stt.results_queue.get_nowait()
                            drained += 1
                        except:
                            break
                    if drained > 0:
                        print(f"  🧹 Drained {drained} pending STT results")
            except Exception as e:
                print(f"  ⚠️ STT flush error: {e}")
    
    async def _handle_interruption(self):
        """
        Handle user interruption during bot speech.
        IMMEDIATELY stops all TTS output and notifies frontend.
        """
        print(f"  🛑 INTERRUPTION in session {self.session_id} - STOPPING ALL OUTPUT")

        # Set interrupt flag FIRST - this is checked in TTS loops
        self._interrupt_flag = True
        self.is_bot_speaking = False
        self.metrics.interruptions += 1

        # Cancel TTS context immediately (per Cartesia best practices)
        # This stops server-side generation and prevents wasted resources
        if self.tts and hasattr(self.tts, 'cancel_current_context'):
            await self.tts.cancel_current_context()

        # Notify frontend to stop playback immediately - send MULTIPLE times
        # to ensure it gets through even if there's buffered data
        for _ in range(3):
            try:
                await self.websocket.send_json({
                    "type": "control",
                    "action": "stop_playback"
                })
            except:
                pass

        # Clear any queued audio
        while not self.audio_output_queue.empty():
            try:
                self.audio_output_queue.get_nowait()
            except:
                break

        # Reset turn detector for new turn
        self.session_turn_detector.reset()
        self.session_vad.reset_states()
        self.turn_start_time = None  # Reset turn timing

        # Reset UtteranceEnd tracking
        self._utterance_end_received = False
        self._last_stt_final_time = None
        # Reset transcript event
        self._transcript_event.clear()

        # Clear transcripts to start fresh
        async with self.transcript_lock:
            self.partial_transcript = ""
            self.final_transcript = ""
            self._accumulation_start_time = None  # Reset accumulation timer

        # CRITICAL: Flush Deepgram's buffer to ensure clean state
        # Without this, Deepgram can get stuck and stop sending transcripts
        await self._flush_stt()

        # Set cooldown timestamp - don't process turns immediately after interruption
        # This gives the user time to finish their thought
        self._last_interruption_time = time.time()

        print(f"  ✅ Interruption handled - {self._interruption_cooldown_ms}ms cooldown started")
    
    async def cleanup(self):
        """Cleanup session resources."""
        print(f"  🧹 Cleaning up session {self.session_id}")
        
        # Cancel tasks
        for task in self.current_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close API connections
        if self.stt:
            await self.stt.close()
        if self.tts:
            await self.tts.close()
        
        # Clear state
        self.conversation_history.clear()

