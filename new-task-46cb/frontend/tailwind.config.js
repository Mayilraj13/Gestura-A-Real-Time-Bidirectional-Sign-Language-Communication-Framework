/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: { 50:"#eff6ff", 500:"#3b82f6", 600:"#2563eb", 900:"#1e3a8a" },
      },
      animation: {
        pulse2: "pulse 1.5s cubic-bezier(0.4,0,0.6,1) infinite",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
