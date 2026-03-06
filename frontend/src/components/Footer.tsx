import { Link } from "react-router-dom";
import { Shield, Github, Puzzle, ExternalLink } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-950 transition-colors duration-300">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
                <Shield className="size-4 text-white" />
              </div>
              <span className="text-gray-900 dark:text-white" style={{ fontWeight: 700 }}>TruthLens</span>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed max-w-xs">
              AI-powered misinformation detection. Helping people navigate the information landscape with confidence.
            </p>
          </div>

          {/* Links */}
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-400 dark:text-gray-600 mb-3" style={{ fontWeight: 600 }}>
              Navigation
            </p>
            <div className="space-y-2">
              <Link to="/" className="block text-sm text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors">
                Analyzer
              </Link>
              <Link to="/about" className="block text-sm text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors">
                About
              </Link>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
              >
                <Github className="size-3.5" />
                GitHub
                <ExternalLink className="size-3 opacity-50" />
              </a>
            </div>
          </div>

          {/* Extension */}
          <div>
            <p className="text-xs uppercase tracking-widest text-gray-400 dark:text-gray-600 mb-3" style={{ fontWeight: 600 }}>
              Browser Extension
            </p>
            <a
              href="https://chrome.google.com/webstore"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white text-sm shadow-sm hover:shadow-md transition-all duration-300 group"
              style={{ fontWeight: 600 }}
            >
              <Puzzle className="size-4 group-hover:rotate-12 transition-transform duration-300" />
              Add to Chrome
            </a>
            <p className="mt-2 text-xs text-gray-400 dark:text-gray-600">Free · Chrome 88+</p>
          </div>
        </div>

        <div className="pt-6 border-t border-gray-200 dark:border-gray-800 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-gray-400 dark:text-gray-600">
          <p>© 2026 TruthLens · Built with FastAPI + Transformer Models</p>
          <p>v1.0 · For educational use only</p>
        </div>
      </div>
    </footer>
  );
}
