"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { VoiceInterface } from "@/components/VoiceInterface";
import { TranscriptPanel } from "@/components/TranscriptPanel";
import { MetricsDashboard } from "@/components/MetricsDashboard";
import { ProviderSelector } from "@/components/ProviderSelector";

interface Message {
  role: "user" | "assistant";
  text: string;
  timestamp: Date;
}

interface Metrics {
  vad: { p50: number; p95: number; avg: number };
  stt: { p50: number; p95: number; avg: number };
  llm_ttft: { p50: number; p95: number; avg: number };
  tts_ttfb: { p50: number; p95: number; avg: number };
  ttfa: { p50: number; p95: number; avg: number };
  turns: number;
  interruptions: number;
  cache_hits: number;
}

export default function Home() {
  const [isConnected, setIsConnected] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentToken, setCurrentToken] = useState("");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [audioLevels, setAudioLevels] = useState<number[]>(Array(24).fill(0));
  
  // Provider selection
  const [llmProvider, setLLMProvider] = useState("groq");
  const [ttsProvider, setTTSProvider] = useState("cartesia");
  
  // UI state
  const [showProviders, setShowProviders] = useState(true);

  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const audioQueueRef = useRef<Float32Array[]>([]);
  const isPlayingRef = useRef(false);
  const nextPlayTimeRef = useRef(0);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  // Track active audio sources for interruption handling
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  // Connect to voice server
  const connect = useCallback(async () => {
    try {
      setError(null);
      setShowProviders(false);
      const sessionId = crypto.randomUUID();
      const ws = new WebSocket(
        `ws://localhost:8000/ws/voice/${sessionId}?llm=${llmProvider}&tts=${ttsProvider}`
      );
      console.log(`🔌 Connecting with LLM: ${llmProvider}, TTS: ${ttsProvider}`);

      ws.binaryType = "arraybuffer";

      ws.onopen = async () => {
        console.log("✅ Connected to voice server");
        setIsConnected(true);
        playbackContextRef.current = new AudioContext({ sampleRate: 24000 });
        nextPlayTimeRef.current = 0;
        await startAudioCapture(ws);
      };

      ws.onmessage = async (event) => {
        if (event.data instanceof ArrayBuffer) {
          handleAudioData(event.data);
        } else {
          const data = JSON.parse(event.data);
          handleMessage(data);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setError("Connection error");
      };

      ws.onclose = () => {
        console.log("❌ Disconnected from voice server");
        setIsConnected(false);
        setIsListening(false);
        setShowProviders(true);
        stopAudioCapture();
      };

      wsRef.current = ws;
    } catch (err) {
      console.error("Connection error:", err);
      setError("Could not connect to server");
    }
  }, [llmProvider, ttsProvider]);

  const startAudioCapture = async (ws: WebSocket) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      mediaStreamRef.current = stream;
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.5;
      analyserRef.current = analyser;

      const processor = audioContext.createScriptProcessor(512, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (ws.readyState === WebSocket.OPEN) {
          const inputData = e.inputBuffer.getChannelData(0);
          const int16 = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }
          ws.send(int16.buffer);
        }
      };

      source.connect(analyser);
      source.connect(processor);
      processor.connect(audioContext.destination);
      startAudioVisualization();
      setIsListening(true);
    } catch (err) {
      console.error("Microphone error:", err);
      setError("Could not access microphone");
    }
  };

  // Smoothed levels for fluid animation (persisted between frames)
  const smoothedLevelsRef = useRef<number[]>(Array(24).fill(0));
  // Phase offsets for organic wave motion
  const phaseOffsetsRef = useRef<number[]>(
    Array(24).fill(0).map(() => Math.random() * Math.PI * 2)
  );
  const frameCountRef = useRef(0);

  const startAudioVisualization = () => {
    const updateLevels = () => {
      if (!analyserRef.current) return;

      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
      analyserRef.current.getByteFrequencyData(dataArray);
      frameCountRef.current++;

      // Calculate overall voice energy (focus on voice frequencies 100-3000Hz)
      let voiceEnergy = 0;
      for (let i = 3; i < 96; i++) {
        voiceEnergy += dataArray[i];
      }
      const normalizedEnergy = (voiceEnergy / (93 * 255));
      const energy = Math.pow(normalizedEnergy, 0.6) * 1.8; // Boost and curve

      const numBars = 24;
      const centerIndex = numBars / 2;
      const time = frameCountRef.current * 0.15;

      const newLevels: number[] = [];

      for (let i = 0; i < numBars; i++) {
        // Distance from center (0 at center, 1 at edges)
        const distFromCenter = Math.abs(i - centerIndex + 0.5) / centerIndex;

        // Base height influenced by energy - center bars taller
        const centerBoost = 1 - distFromCenter * 0.5;
        const baseLevel = energy * centerBoost;

        // Add organic wave motion when there's energy
        const phase = phaseOffsetsRef.current[i];
        const wave = Math.sin(time + phase) * 0.15 + Math.sin(time * 1.7 + phase * 2) * 0.1;
        const waveContribution = energy > 0.05 ? wave * energy : 0;

        // Small random variation for liveliness
        const jitter = (Math.random() - 0.5) * 0.08 * energy;

        const targetLevel = Math.max(0, Math.min(1, baseLevel + waveContribution + jitter));

        // Smooth interpolation
        const current = smoothedLevelsRef.current[i];
        const smoothing = targetLevel > current ? 0.35 : 0.88; // Fast rise, slow fall

        if (targetLevel > current) {
          smoothedLevelsRef.current[i] = current + (targetLevel - current) * smoothing;
        } else {
          smoothedLevelsRef.current[i] = current * smoothing;
        }

        // Minimum threshold
        if (smoothedLevelsRef.current[i] < 0.03) {
          smoothedLevelsRef.current[i] = 0;
        }

        newLevels.push(smoothedLevelsRef.current[i]);
      }

      setAudioLevels(newLevels);
      animationFrameRef.current = requestAnimationFrame(updateLevels);
    };
    updateLevels();
  };

  const stopAudioCapture = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect();
      analyserRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    // Reset all refs and state
    smoothedLevelsRef.current = Array(24).fill(0);
    frameCountRef.current = 0;
    setAudioLevels(Array(24).fill(0));
    setIsListening(false);
  };

  const handleMessage = (data: any) => {
    switch (data.type) {
      case "transcript":
        setMessages((prev) => [
          ...prev,
          { role: data.role, text: data.text, timestamp: new Date() },
        ]);
        if (data.role === "assistant") setCurrentToken("");
        break;
      case "llm_token":
        setCurrentToken((prev) => prev + data.token);
        break;
      case "ttfa":
        console.log(`📊 TTFA: ${data.ms.toFixed(0)}ms`);
        break;
      case "metrics":
        setMetrics(data.data);
        break;
      case "control":
        if (data.action === "stop_playback") {
          stopPlayback();
          setCurrentToken("");  // Clear partial response on stop
        }
        break;
      case "interruption":
        console.log("🛑 Interruption detected");
        stopPlayback();
        setCurrentToken("");  // Clear partial response to prevent mixing with next turn
        break;
      case "error":
        setError(data.message);
        break;
    }
  };

  const handleAudioData = (arrayBuffer: ArrayBuffer) => {
    const ctx = playbackContextRef.current;
    if (!ctx) return;

    const int16Array = new Int16Array(arrayBuffer);
    const float32 = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
      // Convert PCM16 to float32 - full range, gain controlled later
      float32[i] = int16Array[i] / 32768;
    }

    // Improved buffering strategy:
    // - Buffer at least 150ms (3600 samples at 24kHz) before starting playback
    // - This reduces audio artifacts and improves smoothness
    const MIN_BUFFER_SAMPLES = 3600;  // 150ms at 24kHz
    const TARGET_BUFFER_SAMPLES = 7200;  // 300ms for smooth playback

    audioQueueRef.current.push(float32);
    const totalBuffered = audioQueueRef.current.reduce((sum, arr) => sum + arr.length, 0);

    // Wait for minimum buffer before starting playback
    if (!isPlayingRef.current && totalBuffered < MIN_BUFFER_SAMPLES) {
      return;
    }

    // For ongoing playback, combine if we have enough buffered
    if (isPlayingRef.current && totalBuffered < MIN_BUFFER_SAMPLES / 2) {
      return; // Keep buffering
    }

    // Combine all buffered audio
    const totalLength = audioQueueRef.current.reduce((sum, arr) => sum + arr.length, 0);
    const combinedAudio = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of audioQueueRef.current) {
      combinedAudio.set(chunk, offset);
      offset += chunk.length;
    }
    audioQueueRef.current = [];

    // Create audio buffer
    const audioBuffer = ctx.createBuffer(1, combinedAudio.length, 24000);
    audioBuffer.getChannelData(0).set(combinedAudio);

    // Create source with minimal processing for natural sound
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;

    // Use a gain node for volume control (no filters that degrade quality)
    const gainNode = ctx.createGain();
    gainNode.gain.value = 0.95;  // Slight reduction to prevent clipping

    // Light compression for consistent volume (very subtle)
    const compressor = ctx.createDynamicsCompressor();
    compressor.threshold.value = -18;  // Only compress loud peaks
    compressor.knee.value = 40;        // Very soft knee for natural sound
    compressor.ratio.value = 2;        // Gentle compression
    compressor.attack.value = 0.01;    // Slower attack preserves transients
    compressor.release.value = 0.3;    // Natural release

    // Connect: source -> gain -> compressor -> destination
    source.connect(gainNode);
    gainNode.connect(compressor);
    compressor.connect(ctx.destination);

    // Track this source for interruption handling
    activeSourcesRef.current.add(source);

    // Schedule playback with proper timing
    const now = ctx.currentTime;
    // Smaller lookahead for lower latency (was 0.05, now 0.02)
    const startTime = Math.max(now + 0.02, nextPlayTimeRef.current);
    source.start(startTime);
    nextPlayTimeRef.current = startTime + audioBuffer.duration;

    if (!isPlayingRef.current) {
      isPlayingRef.current = true;
      setIsBotSpeaking(true);
    }

    source.onended = () => {
      // Remove from active sources
      activeSourcesRef.current.delete(source);

      // Check if this was the last scheduled audio
      if (ctx.currentTime >= nextPlayTimeRef.current - 0.05) {
        // Small delay before marking as not speaking to handle gaps
        setTimeout(() => {
          if (audioQueueRef.current.length === 0 && activeSourcesRef.current.size === 0) {
            isPlayingRef.current = false;
            setIsBotSpeaking(false);
          }
        }, 100);
      }
    };
  };

  const stopPlayback = useCallback(() => {
    // Stop all active audio sources immediately
    activeSourcesRef.current.forEach((source) => {
      try {
        source.stop();
        source.disconnect();
      } catch (e) {
        // Source may already be stopped
      }
    });
    activeSourcesRef.current.clear();

    // Reset audio context for clean slate
    if (playbackContextRef.current) {
      playbackContextRef.current.close();
      playbackContextRef.current = new AudioContext({ sampleRate: 24000 });
      nextPlayTimeRef.current = 0;
    }

    // Clear audio queue
    audioQueueRef.current = [];
    isPlayingRef.current = false;
    setIsBotSpeaking(false);
    console.log("🛑 Playback stopped - all audio cleared");
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    stopAudioCapture();
    if (playbackContextRef.current) {
      playbackContextRef.current.close();
      playbackContextRef.current = null;
    }
    setIsConnected(false);
    setMessages([]);
    setMetrics(null);
    setCurrentToken("");
  }, []);

  useEffect(() => {
    return () => { disconnect(); };
  }, [disconnect]);

  return (
    <main className="min-h-screen bg-[#0a0a0a]">
      <div className="container mx-auto px-4 sm:px-6 py-4 sm:py-6 max-w-7xl">
        
        {/* Header */}
        <header className="mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-[#8b5cf6] flex items-center justify-center border border-purple-400/20">
                <div className="w-2 h-2 bg-white"></div>
              </div>
              <div>
                <h1 className="text-lg font-semibold text-white tracking-tight">VOICE PIPELINE</h1>
                <p className="text-xs text-[#666] font-mono">REAL-TIME AI COMMUNICATION</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Active providers badge */}
              <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-[#0a0a0a] border border-[#222]">
                <span className="text-[10px] text-[#555] uppercase tracking-wider font-mono">ACTIVE</span>
                <span className="text-xs font-mono text-[#888]">
                  {llmProvider.toUpperCase()} · {ttsProvider.toUpperCase()}
                </span>
              </div>
              
              {/* Connection status */}
              <div className="flex items-center gap-2 px-2 py-1 bg-[#0a0a0a] border border-[#222]">
                <div className={`w-1.5 h-1.5 transition-colors ${
                  isConnected 
                    ? isBotSpeaking 
                      ? 'bg-purple-500' 
                      : isListening 
                        ? 'bg-green-500' 
                        : 'bg-white'
                    : 'bg-[#333]'
                }`} />
                <span className={`text-[10px] font-mono tracking-wider ${isConnected ? 'text-[#888]' : 'text-[#555]'}`}>
                  {isConnected 
                    ? isBotSpeaking 
                      ? 'SPEAKING' 
                      : isListening 
                        ? 'LISTENING' 
                        : 'CONNECTED'
                    : 'OFFLINE'}
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Error display */}
        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 animate-fade-in">
            <div className="flex items-center justify-between">
              <span className="text-sm text-red-400">{error}</span>
              <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300">✕</button>
            </div>
          </div>
        )}

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
          
          {/* Left Panel - Controls */}
          <div className="lg:col-span-4 xl:col-span-3 space-y-4">
            
            {/* Voice Control Card */}
            <VoiceInterface
              isConnected={isConnected}
              isListening={isListening}
              isBotSpeaking={isBotSpeaking}
              audioLevels={audioLevels}
              onConnect={connect}
              onDisconnect={disconnect}
            />
            
            {/* Provider Selector - Collapsible */}
            <div className="panel overflow-hidden">
              <button 
                type="button"
                onClick={() => setShowProviders(!showProviders)}
                disabled={isConnected}
                className={`w-full p-4 flex items-center justify-between ${isConnected ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
              >
                <div className="flex items-center gap-3">
                  <div className="w-6 h-6 border border-[#333] flex items-center justify-center">
                    <div className="w-2 h-2 bg-[#666]"></div>
                  </div>
                  <div className="text-left">
                    <div className="text-xs font-mono tracking-wider text-white">PROVIDER CONFIG</div>
                    <div className="text-[10px] text-[#555] font-mono">{llmProvider.toUpperCase()} · {ttsProvider.toUpperCase()}</div>
                  </div>
                </div>
                <div className={`w-3 h-3 border-l border-b border-[#666] transition-transform duration-200 ${showProviders ? '-rotate-135' : 'rotate-45'}`} />
              </button>
              
              {showProviders && !isConnected && (
                <div className="animate-slide-down">
                  <ProviderSelector
                    llmProvider={llmProvider}
                    ttsProvider={ttsProvider}
                    onLLMChange={setLLMProvider}
                    onTTSChange={setTTSProvider}
                    disabled={isConnected}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Conversation & Metrics */}
          <div className="lg:col-span-8 xl:col-span-9 space-y-4">
            
            {/* Transcript */}
            <TranscriptPanel
              messages={messages}
              currentToken={currentToken}
              isConnected={isConnected}
            />
            
            {/* Metrics - only show when available */}
            {metrics && (
              <div className="animate-fade-in">
                <MetricsDashboard 
                  metrics={metrics} 
                  llmProvider={llmProvider}
                  ttsProvider={ttsProvider}
                />
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-6 pt-4 border-t border-[#1a1a1a]">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2 text-[10px] text-[#444] font-mono">
            <div className="flex items-center gap-3">
              <span className="text-[#333]">STACK</span>
              <span className="text-[#555]">FASTAPI</span>
              <span className="text-[#555]">NEXT.JS</span>
              <span className="text-[#555]">DEEPGRAM</span>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-[#333]">CONFIG</span>
              <span className="text-[#555]">{llmProvider.toUpperCase()}</span>
              <span className="text-[#555]">{ttsProvider.toUpperCase()}</span>
              <span className="text-[#555]">WS-PCM16-16KHZ</span>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
