import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        tactical: {
          bg: {
            primary: "#0a0f14",
            secondary: "#111820",
            tertiary: "#1a2332",
          },
          accent: {
            primary: "#00ff88",
            secondary: "#00d4ff",
            warning: "#ffb347",
            danger: "#ff4444",
          },
          text: {
            primary: "#e5e7eb",
            secondary: "#9ca3af",
            muted: "#6b7280",
          },
          border: {
            primary: "#1f2937",
            accent: "#374151",
          },
        },
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "bounce-slow": "bounce 2s infinite",
        "scan": "scanline 8s linear infinite",
        "data-stream": "dataStream 3s linear infinite",
      },
      keyframes: {
        scanline: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
        dataStream: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;

