"use client";

import { useEffect, useRef } from "react";

interface Message {
  role: "user" | "assistant";
  text: string;
  timestamp: Date;
}

interface TranscriptPanelProps {
  messages: Message[];
  currentToken: string;
  isConnected: boolean;
}

export function TranscriptPanel({
  messages,
  currentToken,
  isConnected,
}: TranscriptPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, currentToken]);

  return (
    <div className="panel p-5 h-[400px] flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-[#1a1a1a]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 border border-[#333] flex items-center justify-center">
            <div className="w-3 h-3 border border-[#666]"></div>
          </div>
          <div>
            <h2 className="text-xs font-mono tracking-wider text-white">TRANSCRIPT</h2>
            <p className="text-[10px] text-[#555] font-mono">{messages.length} ENTRIES</p>
          </div>
        </div>
        {isConnected && (
          <div className="flex items-center gap-2 px-2 py-1 bg-green-500/10 border border-green-500/20">
            <div className="w-1.5 h-1.5 bg-green-500" />
            <span className="text-[10px] text-green-400 font-mono">LIVE</span>
          </div>
        )}
      </div>

      {/* Messages */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto space-y-3 pr-2"
      >
        {messages.length === 0 && !currentToken ? (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="w-16 h-16 border border-[#222] flex items-center justify-center mb-4">
              <div className="w-6 h-6 border border-[#333]"></div>
            </div>
            <p className="text-xs text-[#555] mb-1 font-mono tracking-wider">
              {isConnected ? 'AWAITING INPUT' : 'NO DATA'}
            </p>
            <p className="text-[10px] text-[#444] font-mono">
              {isConnected ? 'MONITORING ACTIVE' : 'INITIALIZE TO BEGIN'}
            </p>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageBubble key={i} message={msg} />
            ))}
            
            {/* Streaming response */}
            {currentToken && (
              <div className="animate-fade-in">
                <div className="flex items-start gap-3">
                  <div className="w-7 h-7 border border-purple-500/50 bg-purple-500/10 flex items-center justify-center flex-shrink-0">
                    <span className="text-[10px] text-purple-400 font-mono">AI</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-white/90 leading-relaxed">
                      {currentToken}
                      <span className="cursor-blink text-purple-400 font-mono">_</span>
                    </p>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  
  return (
    <div className={`flex items-start gap-3 animate-fade-in ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`
        w-7 h-7 border flex items-center justify-center flex-shrink-0
        ${isUser ? 'border-white/30 bg-white/5' : 'border-purple-500/50 bg-purple-500/10'}
      `}>
        <span className={`text-[10px] font-mono ${isUser ? 'text-white/60' : 'text-purple-400'}`}>
          {isUser ? 'U' : 'AI'}
        </span>
      </div>
      
      <div className={`flex-1 min-w-0 ${isUser ? 'text-right' : ''}`}>
        <p className={`text-sm leading-relaxed ${isUser ? 'text-white/60' : 'text-white/90'}`}>
          {message.text}
        </p>
        <p className="text-[10px] text-[#444] mt-1 font-mono">
          {message.timestamp.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit',
            hour12: false 
          })}
        </p>
      </div>
    </div>
  );
}
