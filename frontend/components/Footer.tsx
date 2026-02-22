import { Github } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-16 pt-8 border-t border-gray-200 dark:border-gray-800">
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-600 dark:text-gray-400">
        <p>Built with FastAPI + Transformer Models</p>
        <div className="flex items-center gap-4">
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-gray-900 dark:hover:text-gray-200 transition-colors"
          >
            <Github className="size-5" />
            <span className="sr-only">GitHub</span>
          </a>
          <span>v1.0</span>
        </div>
      </div>
    </footer>
  );
}
