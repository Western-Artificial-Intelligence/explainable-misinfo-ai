import { useState } from "react";
import { Link, useLocation } from "react-router";
import { Shield, Moon, Sun, Puzzle, Menu, X, Zap } from "lucide-react";
import { Button } from "./ui/button";

interface NavbarProps {
  darkMode: boolean;
  onToggleDarkMode: () => void;
}

export function Navbar({ darkMode, onToggleDarkMode }: NavbarProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  const navLinks = [
    { label: "Analyzer", href: "/" },
    { label: "About", href: "/about" },
  ];

  const isActive = (href: string) =>
    href === "/" ? location.pathname === "/" : location.pathname.startsWith(href);

  return (
    <nav className="sticky top-0 z-50 border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-950/80 backdrop-blur-md transition-colors duration-300">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-md group-hover:shadow-blue-400/40 transition-shadow duration-300">
              <Shield className="size-5 text-white" />
            </div>
            <div className="flex flex-col leading-none">
              <span className="text-gray-900 dark:text-white tracking-tight" style={{ fontWeight: 700, fontSize: "1.1rem" }}>
                TruthLens
              </span>
              <span className="text-[10px] text-blue-500 dark:text-blue-400 tracking-wide" style={{ fontWeight: 500 }}>
                AI FACT CHECKER
              </span>
            </div>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                to={link.href}
                className={`px-4 py-2 rounded-lg text-sm transition-colors duration-200 ${
                  isActive(link.href)
                    ? "bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400"
                    : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800"
                }`}
                style={{ fontWeight: 500 }}
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* Right actions */}
          <div className="flex items-center gap-2">
            {/* Chrome Extension Button */}
            <a
              href="https://chrome.google.com/webstore"
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white text-sm shadow-md hover:shadow-orange-400/40 transition-all duration-300 group"
              style={{ fontWeight: 600 }}
            >
              <Puzzle className="size-4 group-hover:rotate-12 transition-transform duration-300" />
              <span className="hidden lg:block">Get Extension</span>
              <span className="lg:hidden">Extension</span>
            </a>

            {/* Dark mode toggle */}
            <Button
              variant="outline"
              size="icon"
              onClick={onToggleDarkMode}
              className="rounded-full w-9 h-9 border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 hover:bg-gray-50 dark:hover:bg-gray-800"
            >
              {darkMode ? (
                <Sun className="size-4 text-yellow-500" />
              ) : (
                <Moon className="size-4 text-gray-600" />
              )}
              <span className="sr-only">Toggle dark mode</span>
            </Button>

            {/* Mobile menu */}
            <button
              className="md:hidden p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              onClick={() => setMobileOpen(!mobileOpen)}
            >
              {mobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
            </button>
          </div>
        </div>

        {/* Mobile menu dropdown */}
        {mobileOpen && (
          <div className="md:hidden py-3 border-t border-gray-200 dark:border-gray-800 space-y-1 animate-in slide-in-from-top-2 duration-200">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                to={link.href}
                onClick={() => setMobileOpen(false)}
                className={`block px-4 py-2.5 rounded-lg text-sm transition-colors ${
                  isActive(link.href)
                    ? "bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400"
                    : "text-gray-600 dark:text-gray-400"
                }`}
                style={{ fontWeight: 500 }}
              >
                {link.label}
              </Link>
            ))}
            <a
              href="https://chrome.google.com/webstore"
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => setMobileOpen(false)}
              className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm text-orange-600 dark:text-orange-400"
              style={{ fontWeight: 500 }}
            >
              <Puzzle className="size-4" />
              Get Chrome Extension
            </a>
          </div>
        )}
      </div>
    </nav>
  );
}
