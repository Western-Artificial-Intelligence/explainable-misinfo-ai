import { Sparkles, Loader2 } from "lucide-react";
import { Textarea } from "./ui/textarea";
import { Button } from "./ui/button";
import { Card } from "./ui/card";

interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  disabled: boolean;
}

export function TextInput({
  value,
  onChange,
  onAnalyze,
  isAnalyzing,
  disabled,
}: TextInputProps) {
  return (
    <Card className="p-6 shadow-lg border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
      <div className="space-y-4">
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Paste or type any text to analyze…"
          className="min-h-[250px] resize-none border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-950 text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-600 focus-visible:ring-blue-500"
          disabled={disabled}
        />

        <Button
          onClick={onAnalyze}
          disabled={!value.trim() || isAnalyzing}
          className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 dark:from-emerald-500 dark:to-cyan-500 dark:hover:from-emerald-600 dark:hover:to-cyan-600 text-white shadow-md disabled:opacity-50"
          size="lg"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="size-5 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Sparkles className="size-5 mr-2" />
              Analyze with AI
            </>
          )}
        </Button>
      </div>
    </Card>
  );
}