import { Button } from "./ui/button";

interface ExampleButtonsProps {
  onExampleClick: (text: string) => void;
}

const examples = [
  "COVID-19 vaccines cause microchips.",
  "The Earth orbits the Sun.",
  "AI will replace all human jobs.",
  "Climate change is a hoax.",
];

export function ExampleButtons({ onExampleClick }: ExampleButtonsProps) {
  return (
    <div className="space-y-3">
      <p className="text-sm text-gray-600 dark:text-gray-400">
        Try an example:
      </p>
      <div className="flex flex-wrap gap-2">
        {examples.map((example, index) => (
          <Button
            key={index}
            variant="outline"
            size="sm"
            onClick={() => onExampleClick(example)}
            className="rounded-full border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300"
          >
            {example}
          </Button>
        ))}
      </div>
    </div>
  );
}
