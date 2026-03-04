import { Shield, Moon, Sun } from "lucide-react";
import { Button } from "./ui/button";

interface HeaderProps {
  darkMode: boolean;
  onToggleDarkMode: () => void;
}

export function Header({ darkMode, onToggleDarkMode }: HeaderProps) {
  return (
    <header className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 dark:from-blue-400 dark:to-indigo-500 flex items-center justify-center shadow-lg">
          <Shield className="size-6 text-white" />
        </div>
        <div>
          <h1 className="text-gray-900 dark:text-white">TruthLens</h1>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            AI-powered misinformation detection
          </p>
        </div>
      </div>

      <Button
        variant="outline"
        size="icon"
        onClick={onToggleDarkMode}
        className="rounded-full border-gray-300 dark:border-gray-700"
      >
        {darkMode ? (
          <Sun className="size-5 text-yellow-500" />
        ) : (
          <Moon className="size-5 text-gray-700" />
        )}
        <span className="sr-only">Toggle dark mode</span>
      </Button>
    </header>
  );
}
