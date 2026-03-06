import { useState, useEffect } from "react";
import { Outlet } from "react-router-dom";
import { Navbar } from "./Navbar";
import { Footer } from "./Footer";

export function Root() {
  const [darkMode, setDarkMode] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("truthlens-dark") === "true" ||
        window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    return false;
  });

  useEffect(() => {
    localStorage.setItem("truthlens-dark", String(darkMode));
  }, [darkMode]);

  // Set favicon and page title
  useEffect(() => {
    document.title = "TruthLens — AI Fact Checker";

    // Set SVG favicon (shield icon in blue/indigo)
    const svgFavicon = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
      <defs>
        <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#3b82f6"/>
          <stop offset="100%" stop-color="#6366f1"/>
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill="url(#g)"/>
      <path d="M16 6 L26 10 L26 17 C26 22 21 26.5 16 28 C11 26.5 6 22 6 17 L6 10 Z" fill="white" opacity="0.9"/>
      <path d="M12 16.5 L14.5 19 L20 13" stroke="#3b82f6" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`;
    const blob = new Blob([svgFavicon], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    let link = document.querySelector<HTMLLinkElement>("link[rel='icon']");
    if (!link) {
      link = document.createElement("link");
      link.rel = "icon";
      document.head.appendChild(link);
    }
    link.type = "image/svg+xml";
    link.href = url;
    return () => URL.revokeObjectURL(url);
  }, []);

  return (
    <div className={darkMode ? "dark" : ""}>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 transition-colors duration-300">
        <Navbar darkMode={darkMode} onToggleDarkMode={() => setDarkMode(!darkMode)} />
        <Outlet context={{ darkMode }} />
        <Footer />
      </div>
    </div>
  );
}