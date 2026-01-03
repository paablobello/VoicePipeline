"""
LLM Module - Multi-Provider Support (Updated Dec 2024)
Using official SDKs for all providers:
- Cerebras: cerebras-cloud-sdk (~100-150ms TTFT) - FASTEST
- Groq: groq SDK (~150-200ms TTFT)
- Gemini: google-genai SDK (~400-600ms TTFT)
- OpenAI: openai SDK (~600-800ms TTFT)
"""
import os
import re
import time
from typing import AsyncGenerator, List, Dict, Optional

# Official SDKs
from cerebras.cloud.sdk import AsyncCerebras
from groq import AsyncGroq
from openai import AsyncOpenAI
from google import genai


class LLMProvider:
    """
    Multi-provider LLM with automatic fallback.
    Uses official SDKs for each provider.
    """
    
    SYSTEM_PROMPT = """Eres un asistente de voz conversacional y amigable. Habla en español de forma natural, como en una conversación real.

IMPORTANTE: Tienes acceso al historial de la conversación. Puedes ver y recordar todo lo que el usuario ha dicho anteriormente. Usa este contexto para dar respuestas coherentes y relevantes.

Reglas importantes:
- Responde con 2-4 frases, siendo informativo pero conciso
- Usa lenguaje coloquial y cercano (como "claro", "vale", "perfecto")
- Si el usuario menciona algo de antes, recuerda el contexto y responde apropiadamente
- Evita ser robótico o formal en exceso
- No uses emojis ni asteriscos
- Si no sabes algo, admítelo con naturalidad
- Mantén un tono cálido pero profesional"""
    
    PROVIDERS = {
        "cerebras": {
            "models": ["llama-3.3-70b", "llama3.1-8b"],
            "default": "llama-3.3-70b",
            "latency": "~100-150ms"
        },
        "groq": {
            # Llama 4 Scout es el más nuevo y rápido (750 T/s)
            "models": [
                "meta-llama/llama-4-scout-17b-16e-instruct",  # NEW! Llama 4 - 750 T/s
                "meta-llama/llama-4-maverick-17b-128e-instruct",  # NEW! Llama 4 - 600 T/s
                "llama-3.3-70b-versatile",  # Fallback
                "llama-3.1-8b-instant",  # Fastest Llama 3
            ],
            "default": "meta-llama/llama-4-scout-17b-16e-instruct",
            "latency": "~100-150ms"
        },
        "gemini": {
            # Gemini 3 Flash (Dec 2024) - Inteligente + Rápido
            # Docs: https://ai.google.dev/gemini-api/docs/models
            "models": [
                "gemini-3-flash-preview",  # NEW! Más inteligente y rápido
                "gemini-2.5-flash",  # Stable - mejor precio/rendimiento
                "gemini-2.5-flash-lite",  # Ultra rápido, más barato
                "gemini-3-pro-preview",  # Más inteligente (pero más lento)
            ],
            "default": "gemini-2.5-flash",  # Stable por defecto
            "latency": "~200-400ms"
        },
        "openai": {
            "models": ["gpt-4o-mini", "gpt-4o"],
            "default": "gpt-4o-mini",
            "latency": "~600-800ms"
        }
    }
    
    def __init__(
        self,
        provider: str = "groq",
        model: Optional[str] = None,
        max_tokens: int = 250,  # Increased for more complete responses
        temperature: float = 0.7,
    ):
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = 0.9
        self.model = model
        self.client = None
        
        self._init_provider(provider, model)
    
    def _init_provider(self, provider: str, model: Optional[str] = None):
        """Initialize the selected provider with official SDK."""
        self.provider = provider
        
        if provider == "cerebras":
            api_key = os.getenv("CEREBRAS_API_KEY")
            if not api_key:
                print("  ⚠️ CEREBRAS_API_KEY not set, falling back to Groq")
                return self._init_provider("groq", model)
            
            self.client = AsyncCerebras(api_key=api_key)
            self.model = model or "llama-3.3-70b"
            print(f"  ✅ Cerebras LLM initialized (model: {self.model}) - FASTEST ~100ms")
        
        elif provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("  ⚠️ GROQ_API_KEY not set, falling back to Gemini")
                return self._init_provider("gemini", model)
            
            self.client = AsyncGroq(api_key=api_key)
            # Llama 4 Scout: 750 tokens/s, fastest Llama model
            self.model = model or "meta-llama/llama-4-scout-17b-16e-instruct"
            print(f"  ✅ Groq LLM initialized (model: {self.model}) - ~100-150ms")
        
        elif provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("  ⚠️ GOOGLE_API_KEY not set, falling back to OpenAI")
                return self._init_provider("openai", model)
            
            # New google-genai SDK (Dec 2024)
            self.client = genai.Client(api_key=api_key)
            # Gemini 2.5 Flash: Stable, mejor precio/rendimiento
            # Para probar Gemini 3: usar "gemini-3-flash-preview"
            self.model = model or "gemini-2.5-flash"
            print(f"  ✅ Gemini LLM initialized (model: {self.model}) - ~200-400ms")
        
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No LLM API key found")
            
            self.client = AsyncOpenAI(api_key=api_key)
            self.model = model or "gpt-4o-mini"
            print(f"  ✅ OpenAI LLM initialized (model: {self.model}) - ~600-800ms")
    
    async def generate(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[dict, None]:
        """Generate streaming response."""
        if self.provider == "gemini":
            async for event in self._generate_gemini(user_message, conversation_history):
                yield event
        else:
            # Cerebras, Groq, OpenAI all use OpenAI-compatible API
            async for event in self._generate_openai_compatible(user_message, conversation_history):
                yield event
    
    async def _generate_openai_compatible(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[dict, None]:
        """Generate using OpenAI-compatible API (Cerebras, Groq, OpenAI)."""
        start_time = time.time()
        first_token_time = None
        
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if conversation_history:
            # Use last 10 messages (5 exchanges) for better context
            messages.extend(conversation_history[-10:])
        messages.append({"role": "user", "content": user_message})
        
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            
            full_response = ""
            current_sentence = ""
            sentence_count = 0
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft_ms = (first_token_time - start_time) * 1000
                        print(f"  ⚡ LLM TTFT: {ttft_ms:.0f}ms ({self.provider})")
                        yield {
                            "type": "ttft",
                            "ttft_ms": ttft_ms,
                            "provider": self.provider
                        }
                    
                    full_response += token
                    current_sentence += token
                    
                    yield {"type": "token", "token": token}
                    
                    if self._is_sentence_complete(current_sentence):
                        sentence = current_sentence.strip()
                        if len(sentence) >= 5:
                            sentence_count += 1
                            yield {"type": "sentence", "text": sentence}
                            current_sentence = ""
            
            if current_sentence.strip() and len(current_sentence.strip()) >= 3:
                yield {"type": "sentence", "text": current_sentence.strip()}
            
            total_ms = (time.time() - start_time) * 1000
            print(f"  ✅ LLM complete: {total_ms:.0f}ms total, {sentence_count} sentences")
            yield {
                "type": "complete",
                "full_response": full_response,
                "total_ms": total_ms
            }
            
        except Exception as e:
            print(f"  ❌ LLM error: {e}")
            yield {"type": "error", "error": str(e)}
    
    async def _generate_gemini(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[dict, None]:
        """Generate using Google Gemini API with new google-genai SDK."""
        start_time = time.time()
        first_token_time = None
        
        # Build contents for Gemini
        contents = []
        
        # Add system instruction
        system_instruction = self.SYSTEM_PROMPT
        
        # Add conversation history (last 10 messages for better context)
        if conversation_history:
            for msg in conversation_history[-10:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # Add current user message
        contents.append({"role": "user", "parts": [{"text": user_message}]})
        
        try:
            # Use async streaming with new SDK
            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "system_instruction": system_instruction,
                }
            )
            
            full_response = ""
            current_sentence = ""
            sentence_count = 0
            
            for chunk in response:
                if chunk.text:
                    token = chunk.text
                    
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft_ms = (first_token_time - start_time) * 1000
                        print(f"  ⚡ LLM TTFT: {ttft_ms:.0f}ms ({self.provider})")
                        yield {
                            "type": "ttft",
                            "ttft_ms": ttft_ms,
                            "provider": self.provider
                        }
                    
                    full_response += token
                    current_sentence += token
                    
                    yield {"type": "token", "token": token}
                    
                    if self._is_sentence_complete(current_sentence):
                        sentence = current_sentence.strip()
                        if len(sentence) >= 5:
                            sentence_count += 1
                            yield {"type": "sentence", "text": sentence}
                            current_sentence = ""
            
            if current_sentence.strip() and len(current_sentence.strip()) >= 3:
                yield {"type": "sentence", "text": current_sentence.strip()}
            
            total_ms = (time.time() - start_time) * 1000
            print(f"  ✅ LLM complete: {total_ms:.0f}ms total, {sentence_count} sentences")
            yield {
                "type": "complete",
                "full_response": full_response,
                "total_ms": total_ms
            }
            
        except Exception as e:
            print(f"  ❌ Gemini error: {e}")
            yield {"type": "error", "error": str(e)}
    
    def _is_sentence_complete(self, text: str) -> bool:
        """Detect if we have a complete sentence."""
        text = text.strip()
        if not text:
            return False
        
        if text[-1] in '.!?':
            return True
        
        if '¿' in text and '?' in text:
            return True
        if '¡' in text and '!' in text:
            return True
        
        if len(text) > 40 and ', ' in text[-10:]:
            return True
        
        return False


# Alias for backward compatibility
OpenAILLM = LLMProvider
