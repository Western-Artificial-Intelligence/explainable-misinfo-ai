import { Shield, Zap, Globe, Lock, Code2, Users, ExternalLink, Puzzle } from "lucide-react";

export function AboutPage() {
  const team = [
    { name: "TruthLens AI", role: "Core Detection Engine", desc: "Fine-tuned BERT transformer models trained on 500K+ verified fact-check datasets." },
    { name: "FastAPI Backend", role: "API Layer", desc: "High-performance Python backend with async processing for sub-second response times." },
    { name: "React Frontend", role: "Web Interface", desc: "Modern, accessible UI with full dark mode and responsive design." },
  ];

  return (
    <main className="max-w-4xl mx-auto px-4 sm:px-6 py-16 space-y-16">
      {/* Hero */}
      <section className="text-center">
        <div className="inline-flex w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 items-center justify-center shadow-xl mb-6">
          <Shield className="size-8 text-white" />
        </div>
        <h1 className="text-gray-900 dark:text-white mb-4" style={{ fontSize: "2.25rem", fontWeight: 800 }}>
          About TruthLens
        </h1>
        <p className="text-gray-500 dark:text-gray-400 max-w-2xl mx-auto leading-relaxed">
          TruthLens is an AI-powered fact-checking tool designed to help people quickly evaluate the credibility of online content. 
          Built on transformer-based NLP models, it provides instant verdicts on claims, articles, and social media posts.
        </p>
      </section>

      {/* Mission */}
      <section className="rounded-2xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 p-8">
        <h2 className="text-gray-900 dark:text-white mb-4" style={{ fontWeight: 700, fontSize: "1.35rem" }}>
          Our Mission
        </h2>
        <p className="text-gray-500 dark:text-gray-400 leading-relaxed mb-4">
          Misinformation spreads faster than the truth. TruthLens exists to give every person the tools to 
          pause, verify, and think critically before sharing content online.
        </p>
        <p className="text-gray-500 dark:text-gray-400 leading-relaxed">
          We believe in an informed public. Our tool is free, privacy-respecting, and constantly improving 
          through ongoing research and community feedback.
        </p>
      </section>

      {/* Stack */}
      <section>
        <h2 className="text-gray-900 dark:text-white mb-6" style={{ fontWeight: 700, fontSize: "1.35rem" }}>
          How It Works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {team.map((item) => (
            <div key={item.name} className="p-5 rounded-2xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800">
              <div className="text-sm text-blue-500 dark:text-blue-400 mb-1" style={{ fontWeight: 600 }}>{item.role}</div>
              <div className="text-gray-900 dark:text-white mb-2" style={{ fontWeight: 600, fontSize: "0.95rem" }}>{item.name}</div>
              <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Chrome Extension */}
      <section className="rounded-2xl overflow-hidden bg-gradient-to-br from-orange-500 to-amber-500 p-8 text-white shadow-lg shadow-orange-400/20">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Puzzle className="size-5" />
              <span className="text-sm" style={{ fontWeight: 600 }}>Browser Extension</span>
            </div>
            <h2 className="text-white mb-2" style={{ fontSize: "1.5rem", fontWeight: 800 }}>
              TruthLens for Chrome
            </h2>
            <p className="opacity-80 text-sm max-w-sm leading-relaxed">
              Right-click any selected text to get an instant verdict without leaving the page you're reading.
            </p>
          </div>
          <a
            href="https://chrome.google.com/webstore"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-6 py-3 rounded-xl bg-white text-orange-600 hover:bg-orange-50 shadow-md hover:shadow-lg transition-all duration-300 flex-shrink-0 group"
            style={{ fontWeight: 700 }}
          >
            <Puzzle className="size-5 group-hover:rotate-12 transition-transform duration-300" />
            Add to Chrome
            <ExternalLink className="size-4 opacity-60" />
          </a>
        </div>
      </section>

      {/* Privacy */}
      <section className="flex gap-4 p-6 rounded-2xl bg-indigo-50 dark:bg-indigo-950/40 border border-indigo-200 dark:border-indigo-800">
        <Lock className="size-6 text-indigo-500 flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="text-gray-900 dark:text-white mb-1" style={{ fontWeight: 600 }}>Privacy by Design</h3>
          <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">
            TruthLens does not store your text, IP address, or any personally identifiable information. 
            All analysis is ephemeral — your data is never logged, sold, or shared with third parties.
          </p>
        </div>
      </section>
    </main>
  );
}
