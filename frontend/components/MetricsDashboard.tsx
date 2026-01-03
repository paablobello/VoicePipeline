"use client";

import { useState } from "react";

interface MetricData {
  p50: number;
  p95: number;
  avg: number;
}

interface Metrics {
  vad: MetricData;
  stt: MetricData;
  llm_ttft: MetricData;
  tts_ttfb: MetricData;
  ttfa: MetricData;
  turns: number;
  interruptions: number;
  cache_hits: number;
}

interface MetricsDashboardProps {
  metrics: Metrics;
  llmProvider?: string;
  ttsProvider?: string;
}

interface StageConfig {
  key: keyof Pick<Metrics, 'vad' | 'stt' | 'llm_ttft' | 'tts_ttfb'>;
  label: string;
  fullName: string;
  target: number;
  color: string;
  colorLight: string;
}

const STAGES: StageConfig[] = [
  {
    key: "vad",
    label: "VAD",
    fullName: "Voice Activity Detection",
    target: 20,
    color: "#3b82f6",
    colorLight: "#60a5fa"
  },
  {
    key: "stt",
    label: "STT",
    fullName: "Speech to Text",
    target: 400,
    color: "#f59e0b",
    colorLight: "#fbbf24"
  },
  {
    key: "llm_ttft",
    label: "LLM",
    fullName: "LLM Time to First Token",
    target: 500,
    color: "#8b5cf6",
    colorLight: "#a78bfa"
  },
  {
    key: "tts_ttfb",
    label: "TTS",
    fullName: "TTS Time to First Byte",
    target: 300,
    color: "#10b981",
    colorLight: "#34d399"
  },
];

const TARGET_TTFA = 1000; // Target total latency

function getStatusColor(value: number, target: number): string {
  const ratio = value / target;
  if (ratio <= 0.8) return "text-green-400";
  if (ratio <= 1.0) return "text-yellow-400";
  return "text-red-400";
}

function getStatusBg(value: number, target: number): string {
  const ratio = value / target;
  if (ratio <= 0.8) return "bg-green-500/10 border-green-500/30";
  if (ratio <= 1.0) return "bg-yellow-500/10 border-yellow-500/30";
  return "bg-red-500/10 border-red-500/30";
}

export function MetricsDashboard({ metrics, llmProvider, ttsProvider }: MetricsDashboardProps) {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);

  const totalLatency = metrics.ttfa.p95;
  const isWithinTarget = totalLatency < TARGET_TTFA;

  // Calculate total pipeline time for percentages
  const totalPipelineTime = STAGES.reduce((sum, stage) => sum + metrics[stage.key].p95, 0);

  return (
    <div className="panel p-5 space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-gradient-to-br from-purple-500/20 to-blue-500/20 border border-purple-500/30 flex items-center justify-center">
            <svg className="w-4 h-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <div>
            <h2 className="text-sm font-semibold text-white tracking-wide">Performance Metrics</h2>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-[10px] text-[#666] font-mono">P95 LATENCY</span>
              {llmProvider && ttsProvider && (
                <span className="text-[10px] px-1.5 py-0.5 bg-purple-500/10 border border-purple-500/20 text-purple-400 font-mono rounded">
                  {llmProvider.toUpperCase()} + {ttsProvider.toUpperCase()}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* TTFA Badge */}
        <div className={`px-4 py-2.5 border rounded-lg ${isWithinTarget ? 'bg-green-500/5 border-green-500/30' : 'bg-red-500/5 border-red-500/30'}`}>
          <div className="text-[10px] text-[#666] font-mono tracking-wider text-center">TIME TO FIRST AUDIO</div>
          <div className={`text-2xl font-bold font-mono text-center ${isWithinTarget ? 'text-green-400' : 'text-red-400'}`}>
            {totalLatency.toFixed(0)}
            <span className="text-xs ml-1 opacity-50">ms</span>
          </div>
        </div>
      </div>

      {/* Pipeline Waterfall */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-[#555] font-mono tracking-wider">PIPELINE WATERFALL</span>
          <span className="text-[10px] text-[#444] font-mono">
            Target: &lt;{TARGET_TTFA}ms
          </span>
        </div>

        <div className="relative bg-[#0a0a0a] border border-[#1a1a1a] rounded-lg p-4">
          {/* Timeline background */}
          <div className="absolute inset-x-4 top-4 bottom-4 flex">
            {[0, 25, 50, 75, 100].map((pct) => (
              <div key={pct} className="flex-1 border-l border-[#1a1a1a] first:border-l-0" />
            ))}
          </div>

          {/* Waterfall bars */}
          <div className="relative space-y-2">
            {STAGES.map((stage, index) => {
              const value = metrics[stage.key].p95;
              const percentage = totalPipelineTime > 0 ? (value / totalPipelineTime) * 100 : 0;
              const offset = STAGES.slice(0, index).reduce((sum, s) => sum + (metrics[s.key].p95 / totalPipelineTime) * 100, 0);
              const isGood = value <= stage.target;

              return (
                <div
                  key={stage.key}
                  className="flex items-center gap-3 group cursor-pointer"
                  onMouseEnter={() => setSelectedMetric(stage.key)}
                  onMouseLeave={() => setSelectedMetric(null)}
                >
                  {/* Label */}
                  <div className="w-10 text-right">
                    <span className="text-[10px] font-mono text-[#666] group-hover:text-white transition-colors">
                      {stage.label}
                    </span>
                  </div>

                  {/* Bar container */}
                  <div className="flex-1 h-7 relative">
                    {/* Bar */}
                    <div
                      className="absolute h-full rounded transition-all duration-500 ease-out flex items-center justify-end pr-2"
                      style={{
                        left: `${offset}%`,
                        width: `${Math.max(percentage, 3)}%`,
                        background: isGood
                          ? `linear-gradient(90deg, ${stage.color}dd, ${stage.colorLight}aa)`
                          : 'linear-gradient(90deg, #ef4444dd, #f87171aa)',
                        boxShadow: selectedMetric === stage.key ? `0 0 20px ${stage.color}40` : 'none',
                      }}
                    >
                      <span className="text-[10px] font-mono text-white font-medium">
                        {value.toFixed(0)}ms
                      </span>
                    </div>
                  </div>

                  {/* Percentage */}
                  <div className="w-12 text-right">
                    <span className={`text-[10px] font-mono ${isGood ? 'text-[#555]' : 'text-red-400'}`}>
                      {percentage.toFixed(0)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Timeline labels */}
          <div className="flex justify-between mt-3 px-10">
            {[0, 25, 50, 75, 100].map((pct) => (
              <span key={pct} className="text-[9px] text-[#333] font-mono">
                {((totalPipelineTime * pct) / 100).toFixed(0)}ms
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Detailed Metrics Cards */}
      <div className="grid grid-cols-4 gap-3">
        {STAGES.map((stage) => {
          const data = metrics[stage.key];
          const isGood = data.p95 <= stage.target;
          const isSelected = selectedMetric === stage.key;

          return (
            <div
              key={stage.key}
              className={`p-3 rounded-lg border transition-all duration-200 cursor-pointer ${
                isSelected
                  ? 'bg-[#111] border-[#333] scale-[1.02]'
                  : 'bg-[#0a0a0a] border-[#1a1a1a] hover:border-[#282828]'
              }`}
              onMouseEnter={() => setSelectedMetric(stage.key)}
              onMouseLeave={() => setSelectedMetric(null)}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: stage.color }}
                  />
                  <span className="text-[10px] font-mono text-[#888] tracking-wider">{stage.label}</span>
                </div>
                <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${getStatusBg(data.p95, stage.target)}`}>
                  {isGood ? 'OK' : 'SLOW'}
                </span>
              </div>

              {/* Main value */}
              <div className={`text-lg font-bold font-mono mb-2 ${getStatusColor(data.p95, stage.target)}`}>
                {data.p95.toFixed(0)}
                <span className="text-[10px] text-[#444] ml-1">ms</span>
              </div>

              {/* Detailed breakdown */}
              <div className="space-y-1 text-[10px] font-mono">
                <div className="flex justify-between">
                  <span className="text-[#444]">p50</span>
                  <span className="text-[#666]">{data.p50.toFixed(0)}ms</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#444]">avg</span>
                  <span className="text-[#666]">{data.avg.toFixed(0)}ms</span>
                </div>
                <div className="flex justify-between border-t border-[#1a1a1a] pt-1 mt-1">
                  <span className="text-[#444]">target</span>
                  <span className="text-[#555]">&lt;{stage.target}ms</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Session Statistics */}
      <div className="pt-4 border-t border-[#1a1a1a]">
        <div className="flex items-center justify-between mb-3">
          <span className="text-[10px] text-[#555] font-mono tracking-wider">SESSION STATISTICS</span>
        </div>

        <div className="grid grid-cols-4 gap-3">
          <StatCard
            label="TURNS"
            value={metrics.turns}
            icon={
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            }
          />
          <StatCard
            label="INTERRUPTS"
            value={metrics.interruptions}
            highlight={metrics.interruptions > 0}
            icon={
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
              </svg>
            }
          />
          <StatCard
            label="CACHE HITS"
            value={metrics.cache_hits}
            icon={
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            }
          />
          <StatCard
            label="AVG TTFA"
            value={metrics.ttfa.avg}
            suffix="ms"
            icon={
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            }
          />
        </div>
      </div>
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: number;
  suffix?: string;
  highlight?: boolean;
  icon?: React.ReactNode;
}

function StatCard({ label, value, suffix, highlight, icon }: StatCardProps) {
  return (
    <div className={`p-3 rounded-lg border transition-all ${
      highlight
        ? 'bg-orange-500/5 border-orange-500/20'
        : 'bg-[#0a0a0a] border-[#1a1a1a] hover:border-[#282828]'
    }`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={`${highlight ? 'text-orange-400' : 'text-[#444]'}`}>
          {icon}
        </span>
        <span className={`text-[10px] font-mono tracking-wider ${highlight ? 'text-orange-400' : 'text-[#555]'}`}>
          {label}
        </span>
      </div>
      <div className={`text-xl font-bold font-mono ${highlight ? 'text-orange-400' : 'text-white'}`}>
        {typeof value === 'number' && !suffix ? value : value.toFixed(0)}
        {suffix && <span className="text-xs text-[#444] ml-1">{suffix}</span>}
      </div>
    </div>
  );
}
