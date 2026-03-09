// src/components/PixelFooter.jsx
export default function PixelFooter() {
  return (
    <footer
      id="footer"
      className="mt-16 border-t border-[rgba(51,65,85,0.4)] bg-[rgba(15,23,42,0.5)]"
    >
      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-8">
          <div className="max-w-md">
            <div className="font-semibold text-slate-100 text-lg">NeuroLearn</div>
            <p className="mt-2 text-sm text-slate-300/90 leading-relaxed">
              Story-based MCQs with real-time attention modes and adaptive difficulty.
            </p>
            <p className="mt-2 text-xs text-slate-400/80">
              Hackathon build • Offline-friendly
            </p>
          </div>
          <div className="max-w-sm">
            <div className="font-semibold text-slate-100 text-sm">ADHD-friendly</div>
            <ul className="mt-2 space-y-1.5 text-sm text-slate-300/80">
              <li>Calm mode (reduced motion)</li>
              <li>Short tasks + instant feedback</li>
              <li>Clear progress + mastery unlocks</li>
            </ul>
          </div>
        </div>
        <div className="mt-8 pt-6 border-t border-[rgba(51,65,85,0.3)] text-center text-xs text-slate-400/80">
          © {new Date().getFullYear()} NeuroLearn • Built for learning with focus
        </div>
      </div>
    </footer>
  );
}