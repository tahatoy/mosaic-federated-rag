/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        garanti: {
          navy: '#004481',          // MosaicBank Deep Navy
          navyDark: '#043263',      // Darker Navy for hovers/headers
          green: '#00853F',         // MosaicBank Green
          greenHover: '#006e34',    // Darker Green for buttons
          greenSoft: '#E8F5E9',     // Soft emerald tint for badges
          bg: '#F4F6F9',            // Soft banking gray background
          surface: '#FFFFFF',       // Pure white card background
          border: '#DDE3EA',        // Corporate light gray border
          borderLight: '#EDF2F7',   // Ultra light divider border
          textDark: '#192A3E',      // Primary navy-slate text
          textMuted: '#5A6B7C',     // Muted slate description text
        },
        bank: {
          darkest: '#080c14',
          bg: '#0b1120',
          surface: '#111b2e',
          card: '#16233b',
          border: '#1e2f4d',
          borderLight: '#2a4168',
          text: '#f1f5f9',
          muted: '#94a3b8',
          accent: '#2563eb', // Corporate Trust Blue
          accentHover: '#1d4ed8',
          gold: '#d97706',
          emerald: '#059669',
          danger: '#dc2626'
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace']
      }
    },
  },
  plugins: [],
}
