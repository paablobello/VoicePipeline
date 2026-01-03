"use client";

interface ProviderSelectorProps {
  llmProvider: string;
  ttsProvider: string;
  onLLMChange: (provider: string) => void;
  onTTSChange: (provider: string) => void;
  disabled?: boolean;
}

const LLM_PROVIDERS = [
  { id: "cerebras", name: "Cerebras", model: "Llama 3.3", latency: "100ms", color: "#f59e0b" },
  { id: "groq", name: "Groq", model: "Llama 4", latency: "120ms", color: "#8b5cf6" },
  { id: "gemini", name: "Gemini", model: "2.5 Flash", latency: "300ms", color: "#3b82f6" },
  { id: "openai", name: "OpenAI", model: "GPT-4o", latency: "700ms", color: "#10b981" },
];

const TTS_PROVIDERS = [
  { id: "cartesia", name: "Cartesia", model: "Sonic 3", latency: "50ms", color: "#ec4899" },
  { id: "elevenlabs", name: "ElevenLabs", model: "Flash", latency: "75ms", color: "#f97316" },
];

export function ProviderSelector({
  llmProvider,
  ttsProvider,
  onLLMChange,
  onTTSChange,
  disabled = false,
}: ProviderSelectorProps) {
  return (
    <div className="px-4 pb-4 space-y-4">
      {/* LLM Selection */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] text-[#555] uppercase tracking-wider font-mono">LLM PROVIDER</span>
          <span className="text-[10px] text-[#666] font-mono">
            {LLM_PROVIDERS.find(p => p.id === llmProvider)?.latency}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {LLM_PROVIDERS.map((provider) => {
            const isActive = llmProvider === provider.id;
            return (
              <button
                key={provider.id}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  if (!disabled) {
                    console.log(`Selecting LLM: ${provider.id}`);
                    onLLMChange(provider.id);
                  }
                }}
                disabled={disabled}
                type="button"
                className={`
                  relative p-2.5 text-left transition-all duration-200
                  ${isActive
                    ? 'bg-purple-500/15 border-2 border-purple-500'
                    : 'bg-[#0a0a0a] border border-[#1a1a1a] hover:border-white/30 hover:bg-[#111]'
                  }
                  ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer active:scale-95'}
                `}
              >
                <div className="flex items-center gap-2 mb-1">
                  <div 
                    className={`w-1.5 h-1.5 transition-opacity ${isActive ? 'opacity-100' : 'opacity-30'}`}
                    style={{ backgroundColor: provider.color }}
                  />
                  <span className={`text-xs font-medium ${isActive ? 'text-white' : 'text-[#888]'}`}>
                    {provider.name}
                  </span>
                </div>
                <div className={`text-[10px] pl-3.5 ${isActive ? 'text-[#888]' : 'text-[#555]'}`}>
                  {provider.model}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* TTS Selection */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] text-[#555] uppercase tracking-wider font-mono">TTS PROVIDER</span>
          <span className="text-[10px] text-[#666] font-mono">
            {TTS_PROVIDERS.find(p => p.id === ttsProvider)?.latency}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {TTS_PROVIDERS.map((provider) => {
            const isActive = ttsProvider === provider.id;
            return (
              <button
                key={provider.id}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  if (!disabled) {
                    console.log(`Selecting TTS: ${provider.id}`);
                    onTTSChange(provider.id);
                  }
                }}
                disabled={disabled}
                type="button"
                className={`
                  relative p-2.5 text-left transition-all duration-200
                  ${isActive
                    ? 'bg-purple-500/15 border-2 border-purple-500'
                    : 'bg-[#0a0a0a] border border-[#1a1a1a] hover:border-white/30 hover:bg-[#111]'
                  }
                  ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer active:scale-95'}
                `}
              >
                <div className="flex items-center gap-2 mb-1">
                  <div 
                    className={`w-1.5 h-1.5 transition-opacity ${isActive ? 'opacity-100' : 'opacity-30'}`}
                    style={{ backgroundColor: provider.color }}
                  />
                  <span className={`text-xs font-medium ${isActive ? 'text-white' : 'text-[#888]'}`}>
                    {provider.name}
                  </span>
                </div>
                <div className={`text-[10px] pl-3.5 ${isActive ? 'text-[#888]' : 'text-[#555]'}`}>
                  {provider.model}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Estimated Latency */}
      <div className="pt-3 border-t border-[#1a1a1a]">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-[#555] font-mono tracking-wider">EST. TOTAL LATENCY</span>
          <span className="text-xs text-[#888] font-mono">
            ~{
              parseInt(LLM_PROVIDERS.find(p => p.id === llmProvider)?.latency || "0") +
              parseInt(TTS_PROVIDERS.find(p => p.id === ttsProvider)?.latency || "0")
            }MS
          </span>
        </div>
      </div>
    </div>
  );
}
