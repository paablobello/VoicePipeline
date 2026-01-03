"""
Smart Turn v3 - End of Turn Detection
Based on Pipecat's Smart Turn model (Whisper Tiny encoder + classifier)

This is the state-of-the-art approach for detecting when a user has finished speaking.
Uses audio features directly (not text) to detect prosody, intonation, and speech patterns.

Model: pipecat-ai/smart-turn-v3 (ONNX version)
Supports 23 languages including Spanish.
"""
import os
import asyncio
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import time


@dataclass  
class SmartTurnResult:
    """Result from Smart Turn analysis."""
    is_complete: bool
    probability: float  # 0-1, higher = more likely complete
    inference_time_ms: float


class SmartTurnDetector:
    """
    Smart Turn v3 detector using Whisper-based audio analysis.
    
    This model analyzes audio features to detect:
    - Prosody patterns (falling vs rising intonation)
    - Filler words ("um", "eh", "pues")
    - Incomplete sentence patterns
    - Natural speech endings
    
    Much more accurate than VAD-only or text-based detection.
    
    Target: ~50-100ms inference on CPU
    """
    
    # Class-level lock for thread-safe initialization (shared across instances)
    _init_lock: asyncio.Lock = None
    _global_session = None  # Shared ONNX session
    _global_feature_extractor = None  # Shared feature extractor

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        max_duration_secs: float = 8.0,
        threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.max_duration_secs = max_duration_secs
        self.max_samples = int(max_duration_secs * sample_rate)
        self.threshold = threshold

        self.session = None
        self.feature_extractor = None
        self._initialized = False

        # Initialize class-level lock if not exists
        if SmartTurnDetector._init_lock is None:
            SmartTurnDetector._init_lock = asyncio.Lock()

    async def initialize(self):
        """
        Initialize the ONNX model and feature extractor.
        Thread-safe: uses lock to prevent double initialization.
        Downloads model from HuggingFace if not cached.
        Shares model across all instances for efficiency.
        """
        if self._initialized:
            return

        # Use lock to prevent race condition during initialization
        async with SmartTurnDetector._init_lock:
            # Double-check after acquiring lock (another instance might have initialized)
            if self._initialized:
                return

            # Check if global model is already loaded
            if SmartTurnDetector._global_session is not None:
                self.session = SmartTurnDetector._global_session
                self.feature_extractor = SmartTurnDetector._global_feature_extractor
                self._initialized = True
                print("  ✅ Smart Turn v3 model (shared instance)")
                return

            try:
                import onnxruntime as ort
                from transformers import WhisperFeatureExtractor

                print("  📥 Loading Smart Turn v3 model...")

                # Use HuggingFace Hub to get model
                from huggingface_hub import hf_hub_download

                # Smart Turn v3.1 - optimized for CPU inference
                model_path = hf_hub_download(
                    repo_id="pipecat-ai/smart-turn-v3",
                    filename="smart-turn-v3.1-cpu.onnx",
                )

                # Initialize ONNX session with optimizations
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 4

                session = ort.InferenceSession(
                    model_path,
                    sess_options,
                    providers=['CPUExecutionProvider']
                )

                # Feature extractor (Whisper mel-spectrogram)
                feature_extractor = WhisperFeatureExtractor(
                    chunk_length=int(self.max_duration_secs),
                    n_mels=80,
                )

                # Store globally for other instances to reuse
                SmartTurnDetector._global_session = session
                SmartTurnDetector._global_feature_extractor = feature_extractor

                # Set instance references
                self.session = session
                self.feature_extractor = feature_extractor
                self._initialized = True
                print("  ✅ Smart Turn v3 model loaded")

            except ImportError as e:
                print(f"  ⚠️ Smart Turn dependencies missing: {e}")
                print("  📦 Install with: pip install onnxruntime transformers huggingface_hub")
                raise
            except Exception as e:
                print(f"  ❌ Failed to load Smart Turn model: {e}")
                raise
    
    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Prepare audio for inference.
        Truncates/pads to max_duration_secs from the END of the audio.
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        # Truncate to last N seconds (we want the END of the turn)
        if len(audio) > self.max_samples:
            audio = audio[-self.max_samples:]
        elif len(audio) < self.max_samples:
            # Pad with zeros at the beginning
            padding = self.max_samples - len(audio)
            audio = np.pad(audio, (padding, 0), mode='constant')
        
        return audio
    
    async def analyze(self, audio: np.ndarray) -> SmartTurnResult:
        """
        Analyze audio to determine if turn is complete.
        
        Args:
            audio: Audio samples as numpy array (16kHz, mono)
            
        Returns:
            SmartTurnResult with probability and decision
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Prepare audio
            audio = self._prepare_audio(audio)
            
            # Extract mel features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="np",
                padding="max_length",
                max_length=self.max_samples,
                truncation=True,
                do_normalize=True,
            )
            
            input_features = inputs["input_features"].astype(np.float32)
            
            # Run inference
            outputs = self.session.run(None, {"input_features": input_features})
            
            # Get probability via sigmoid
            logit = outputs[0][0].item()
            probability = 1 / (1 + np.exp(-logit))  # Sigmoid
            
            inference_time = (time.time() - start_time) * 1000
            
            return SmartTurnResult(
                is_complete=probability > self.threshold,
                probability=probability,
                inference_time_ms=inference_time,
            )
            
        except Exception as e:
            print(f"  ⚠️ Smart Turn inference error: {e}")
            # Fallback: assume complete after long silence
            return SmartTurnResult(
                is_complete=True,
                probability=0.5,
                inference_time_ms=0,
            )
    
    def analyze_sync(self, audio: np.ndarray) -> SmartTurnResult:
        """Synchronous version of analyze."""
        return asyncio.run(self.analyze(audio))


# Singleton instance
_smart_turn: Optional[SmartTurnDetector] = None

async def get_smart_turn_detector() -> SmartTurnDetector:
    """Get singleton Smart Turn detector instance."""
    global _smart_turn
    if _smart_turn is None:
        _smart_turn = SmartTurnDetector()
        await _smart_turn.initialize()
    return _smart_turn

