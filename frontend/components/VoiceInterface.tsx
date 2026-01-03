"use client";

interface VoiceInterfaceProps {
  isConnected: boolean;
  isListening: boolean;
  isBotSpeaking: boolean;
  audioLevels: number[];
  onConnect: () => void;
  onDisconnect: () => void;
}

export function VoiceInterface({
  isConnected,
  isListening,
  isBotSpeaking,
  audioLevels,
  onConnect,
  onDisconnect,
}: VoiceInterfaceProps) {
  const isActive = isConnected && (isListening || isBotSpeaking);

  // Calculate bar properties for smooth visualization
  const getBarStyle = (index: number, level: number) => {
    const centerIndex = audioLevels.length / 2;
    const distanceFromCenter = Math.abs(index - centerIndex + 0.5);
    const normalizedDistance = distanceFromCenter / centerIndex;

    // Height: minimum 15%, scales with level (smaller range for compact view)
    const minHeight = 15;
    const maxHeight = 100;
    const height = isActive
      ? minHeight + level * (maxHeight - minHeight)
      : minHeight;

    // Opacity: center bars slightly brighter
    const baseOpacity = isActive ? 0.6 + (1 - normalizedDistance) * 0.4 : 0.3;

    return {
      height: `${height}%`,
      opacity: baseOpacity,
    };
  };

  return (
    <div className={`panel p-6 transition-all duration-300 ${isActive ? 'panel-active' : ''}`}>

      {/* Center indicator (icon) */}
      <div className="flex justify-center mb-6">
        <div className={`
          relative flex items-center justify-center
          transition-all duration-300
          ${isConnected && isBotSpeaking ? 'scale-105' : ''}
        `}>
          {/* Outer frame */}
          <div className={`
            w-16 h-16 border-2 transition-colors
            ${isConnected
              ? isBotSpeaking
                ? 'border-purple-500'
                : isListening
                  ? 'border-white/50'
                  : 'border-white/20'
              : 'border-[#222]'
            }
          `} />

          {/* Inner core */}
          <div className={`
            absolute w-8 h-8 transition-colors
            ${isConnected
              ? isBotSpeaking
                ? 'bg-purple-500'
                : isListening
                  ? 'bg-white/80'
                  : 'bg-white/30'
              : 'bg-[#333]'
            }
          `} />

          {/* Pulse effect when speaking */}
          {isConnected && isBotSpeaking && (
            <div className="absolute inset-0 border-2 border-purple-500 animate-ping opacity-30" />
          )}
        </div>
      </div>

      {/* Audio Visualizer - compact bar below icon */}
      <div className="h-10 flex items-center justify-center gap-[2px] px-2 mb-6">
        {audioLevels.map((level, i) => (
          <div
            key={i}
            className={`w-[5px] rounded-full transition-all duration-100 ease-out ${
              isActive
                ? isBotSpeaking
                  ? 'bg-gradient-to-t from-purple-600 to-purple-400'
                  : 'bg-gradient-to-t from-white/50 to-white'
                : 'bg-[#333]'
            }`}
            style={getBarStyle(i, level)}
          />
        ))}
      </div>

      {/* Status */}
      <div className="text-center mb-6">
        <div className={`text-sm font-mono tracking-wider mb-1 ${isConnected ? 'text-[#888]' : 'text-[#555]'}`}>
          {!isConnected
            ? 'SYSTEM READY'
            : isBotSpeaking
              ? 'OUTPUT ACTIVE'
              : isListening
                ? 'INPUT MONITORING'
                : 'SESSION ACTIVE'}
        </div>
        <div className="text-xs text-[#444] font-mono">
          {isConnected ? 'TERMINATE SESSION' : 'INITIALIZE CONNECTION'}
        </div>
      </div>

      {/* Main Button */}
      <button
        onClick={isConnected ? onDisconnect : onConnect}
        className={`
          w-full py-3 px-6 font-mono text-sm tracking-wider
          flex items-center justify-center gap-3
          transition-all duration-200 border
          ${isConnected
            ? 'bg-[#0a0a0a] border-white/30 text-white hover:bg-[#111]'
            : 'bg-white/90 border-white text-black hover:bg-white'
          }
        `}
      >
        {isConnected ? (
          <>
            <span className="w-2 h-2 bg-white" />
            <span>TERMINATE</span>
          </>
        ) : (
          <>
            <span className="w-2 h-2 bg-black" />
            <span>INITIALIZE</span>
          </>
        )}
      </button>
    </div>
  );
}
