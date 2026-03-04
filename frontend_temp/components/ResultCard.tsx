import { AlertTriangle, CheckCircle, MessageCircle, Minus } from "lucide-react";
import { Card } from "./ui/card";
import { Progress } from "./ui/progress";
import { Badge } from "./ui/badge";
import type { AnalysisResult } from "../App";

interface ResultCardProps {
  result: AnalysisResult;
}

const predictionConfig = {
  MISINFORMATION: {
    label: "Misinformation",
    icon: AlertTriangle,
    color: "bg-red-500 dark:bg-red-600",
    badgeClass: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400 border-red-300 dark:border-red-800",
    progressClass: "bg-red-500",
  },
  RELIABLE: {
    label: "Reliable",
    icon: CheckCircle,
    color: "bg-green-500 dark:bg-green-600",
    badgeClass: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 border-green-300 dark:border-green-800",
    progressClass: "bg-green-500",
  },
  OPINION: {
    label: "Opinion",
    icon: MessageCircle,
    color: "bg-yellow-500 dark:bg-yellow-600",
    badgeClass: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400 border-yellow-300 dark:border-yellow-800",
    progressClass: "bg-yellow-500",
  },
  NEUTRAL: {
    label: "Neutral",
    icon: Minus,
    color: "bg-gray-500 dark:bg-gray-600",
    badgeClass: "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400 border-gray-300 dark:border-gray-700",
    progressClass: "bg-gray-500",
  },
};

export function ResultCard({ result }: ResultCardProps) {
  const config = predictionConfig[result.prediction];
  const Icon = config.icon;

  return (
    <Card className="p-6 shadow-lg border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="space-y-6">
        <div className="flex items-start gap-4">
          <div className={`w-12 h-12 rounded-xl ${config.color} flex items-center justify-center flex-shrink-0 shadow-md`}>
            <Icon className="size-6 text-white" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">Prediction:</span>
              <Badge variant="outline" className={config.badgeClass}>
                {config.label.toUpperCase()}
              </Badge>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {result.prediction === "MISINFORMATION" &&
                "This text contains claims that appear to be false or misleading based on known facts."}
              {result.prediction === "RELIABLE" &&
                "This text appears to contain factual information supported by evidence."}
              {result.prediction === "OPINION" &&
                "This text primarily expresses opinions or predictions rather than verifiable facts."}
              {result.prediction === "NEUTRAL" &&
                "This text doesn't contain strong indicators of misinformation or reliability."}
            </p>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 dark:text-gray-400">Confidence:</span>
            <span className="text-gray-900 dark:text-gray-100">{result.confidence}%</span>
          </div>
          <Progress value={result.confidence} className="h-3 bg-gray-200 dark:bg-gray-800" indicatorClassName={config.progressClass} />
        </div>
      </div>
    </Card>
  );
}
