"""
Backchannel Detector
Detects short acknowledgement phrases that should NOT trigger turn completion.

Backchannels are listener responses like "mm-hmm", "uh-huh", "ok", "ajá"
that indicate the listener is paying attention but NOT trying to take the floor.

Based on research:
- "Turn-taking and Backchannel Prediction with Acoustic and LLM Fusion" (2024)
- LiveKit Agents backchannel handling
"""
import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class BackchannelResult:
    """Result of backchannel detection."""
    is_backchannel: bool
    confidence: float
    matched_pattern: Optional[str] = None


class BackchannelDetector:
    """
    Detects backchannel responses in Spanish and English.
    
    Backchannels are short utterances that:
    1. Are very short (1-3 words)
    2. Match known acknowledgement patterns
    3. Have low information content
    
    These should NOT trigger turn completion or interruption.
    """
    
    # Spanish backchannels (most common first)
    SPANISH_PATTERNS = [
        # Affirmative responses
        r'^(s[ií]|si+|sip?)\.?$',
        r'^(ok|okey|okay|okei)\.?$',
        r'^(vale|vale vale)\.?$',
        r'^(bueno)\.?$',
        r'^(bien|muy bien)\.?$',
        
        # Acknowledgement sounds
        r'^(aj[áa]|aja+)\.?$',
        r'^(mhm+|mm+|mmhm+|uh[ -]?huh)\.?$',
        r'^(ah[áa]?|aha+)\.?$',
        r'^(eh)\.?$',
        
        # Understanding confirmations
        r'^(claro|claro que s[ií])\.?$',
        r'^(entiendo|ya entiendo)\.?$',
        r'^(ya|ya ya)\.?$',
        r'^(exacto|exactamente)\.?$',
        r'^(correcto)\.?$',
        
        # Continuation encouragers
        r'^(sigue|cont[ií]n[úu]a)\.?$',
        r'^(dale|adelante)\.?$',
        r'^(y luego|y despu[ée]s|y entonces)\??$',
        
        # Interest indicators
        r'^([oo]h|[aa]h)\.?$',
        r'^(interesante|qu[ée] interesante)\.?$',
        r'^([¿?]?en serio[?]?)\.?$',
        r'^([¿?]?de verdad[?]?)\.?$',
        r'^(guau|wow)\.?$',
        
        # Filler/hesitation (not taking floor)
        r'^(pues|bueno pues)\.?$',
        r'^(a ver|vamos a ver)\.?$',
    ]
    
    # English backchannels (for multilingual support)
    ENGLISH_PATTERNS = [
        r'^(yes|yeah|yep|yup)\.?$',
        r'^(ok|okay)\.?$',
        r'^(right|alright)\.?$',
        r'^(sure|of course)\.?$',
        r'^(uh[ -]?huh|mhm+|mm+)\.?$',
        r'^(i see|got it)\.?$',
        r'^(really|oh really)\??$',
        r'^(wow|cool|nice)\.?$',
        r'^(go on|continue)\.?$',
    ]
    
    # Maximum word count for backchannel
    MAX_WORDS = 4
    
    # Maximum character count
    MAX_CHARS = 25
    
    def __init__(self):
        # Compile all patterns
        self.patterns = []
        for pattern in self.SPANISH_PATTERNS + self.ENGLISH_PATTERNS:
            try:
                self.patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass  # Skip invalid patterns
    
    def detect(self, transcript: str) -> BackchannelResult:
        """
        Check if transcript is a backchannel.
        
        Args:
            transcript: The user's utterance
            
        Returns:
            BackchannelResult with detection info
        """
        if not transcript:
            return BackchannelResult(is_backchannel=False, confidence=0.0)
        
        text = transcript.strip().lower()
        
        # Quick length check
        if len(text) > self.MAX_CHARS:
            return BackchannelResult(is_backchannel=False, confidence=0.0)
        
        words = text.split()
        if len(words) > self.MAX_WORDS:
            return BackchannelResult(is_backchannel=False, confidence=0.0)
        
        # Check against patterns
        for pattern in self.patterns:
            match = pattern.match(text)
            if match:
                # Higher confidence for exact matches
                confidence = 0.95 if match.group() == text else 0.8
                return BackchannelResult(
                    is_backchannel=True,
                    confidence=confidence,
                    matched_pattern=pattern.pattern
                )
        
        # Check for very short single-word responses that might be backchannels
        if len(words) == 1 and len(text) <= 5:
            # Common short words that are likely backchannels
            short_backchannels = {'si', 'sí', 'no', 'ya', 'ah', 'oh', 'eh', 'ok'}
            if text in short_backchannels:
                return BackchannelResult(
                    is_backchannel=True,
                    confidence=0.7,
                    matched_pattern="short_word"
                )
        
        return BackchannelResult(is_backchannel=False, confidence=0.0)
    
    def is_likely_backchannel(self, transcript: str, threshold: float = 0.6) -> bool:
        """Quick check if transcript is likely a backchannel."""
        result = self.detect(transcript)
        return result.is_backchannel and result.confidence >= threshold


# Singleton instance
_detector: Optional[BackchannelDetector] = None

def get_backchannel_detector() -> BackchannelDetector:
    """Get singleton backchannel detector instance."""
    global _detector
    if _detector is None:
        _detector = BackchannelDetector()
    return _detector

