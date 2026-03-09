// src/components/PixelSettingsModal.jsx
export default function PixelSettingsModal({ open, onClose, settings, setSettings }) {
  if (!open) return null;

  const Toggle = ({ value, onChange }) => (
    <button
      onClick={() => onChange(!value)}
      className={`h-9 w-16 rounded-full border-2 relative transition ${
        value
          ? "border-[rgba(56,189,248,0.65)] bg-[rgba(56,189,248,0.18)]"
          : "border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)]"
      }`}
    >
      <span
        className={`absolute top-1/2 -translate-y-1/2 h-7 w-7 rounded-full transition ${
          value ? "left-8 bg-[rgb(56,189,248)]" : "left-1 bg-slate-400"
        }`}
      />
    </button>
  );

  const Pill = ({ active, children, onClick }) => (
    <button
      onClick={onClick}
      className={`px-3 py-2 rounded-xl border-2 text-sm transition ${
        active
          ? "border-[rgba(56,189,248,0.65)] bg-[rgba(56,189,248,0.18)]"
          : "border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.8)]"
      }`}
    >
      {children}
    </button>
  );

  const Row = ({ label, hint, right }) => (
    <div className="flex items-start justify-between gap-4 py-3">
      <div>
        <div className="font-semibold">{label}</div>
        {hint && <div className="text-sm text-slate-300/80 mt-0.5">{hint}</div>}
      </div>
      <div className="shrink-0">{right}</div>
    </div>
  );

  return (
    <div className="fixed inset-0 z-50">
      <div className="absolute inset-0 bg-black/45" onClick={onClose} />
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className="w-full max-w-lg rounded-2xl border-2 border-[rgba(48,68,105,0.9)] bg-[rgb(10,20,44)] shadow-2xl overflow-hidden">
          <div className="p-5 border-b border-[rgba(48,68,105,0.6)] bg-[rgba(13,26,58,0.55)]">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xl font-bold tracking-wide">Settings</div>
                <div className="text-sm text-slate-300/80 mt-0.5">
                  Tune your quest UI (calm, sound, sprint time)
                </div>
              </div>
              <button
                onClick={onClose}
                className="px-3 py-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.8)]"
              >
                ✕
              </button>
            </div>
          </div>

          <div className="p-5">
            <Row
              label="Sound FX"
              hint="Soft chime on submit"
              right={
                <Toggle
                  value={settings.sound}
                  onChange={(v) => setSettings((s) => ({ ...s, sound: v }))}
                />
              }
            />
            <div className="border-t border-[rgba(48,68,105,0.6)]" />

            <Row
              label="Calm Mode"
              hint="Reduce motion + intensity"
              right={
                <Toggle
                  value={settings.calm}
                  onChange={(v) => setSettings((s) => ({ ...s, calm: v }))}
                />
              }
            />
            <Row
              label="Auto-Calm"
              hint="Enable calm when drifting/overwhelmed"
              right={
                <Toggle
                  value={settings.autoCalm}
                  onChange={(v) => setSettings((s) => ({ ...s, autoCalm: v }))}
                />
              }
            />

            <div className="border-t border-[rgba(48,68,105,0.6)]" />

            <div className="py-3">
              <div className="font-semibold">Sprint Length</div>
              <div className="text-sm text-slate-300/80 mt-0.5">
                Choose focus timer
              </div>
              <div className="mt-3 flex gap-2 flex-wrap">
                {[60, 90, 120].map((sec) => (
                  <Pill
                    key={sec}
                    active={settings.sprintSeconds === sec}
                    onClick={() => setSettings((s) => ({ ...s, sprintSeconds: sec }))}
                  >
                    {sec}s
                  </Pill>
                ))}
              </div>
            </div>
          </div>

          <div className="p-5 border-t border-[rgba(48,68,105,0.6)] flex items-center justify-between">
            <button
              onClick={() =>
                setSettings({
                  sound: true,
                  calm: false,
                  autoCalm: true,
                  sprintSeconds: 60,
                })
              }
              className="px-4 py-2 rounded-xl border border-[rgba(48,68,105,0.8)] bg-[rgba(13,26,58,0.55)] hover:bg-[rgba(13,26,58,0.8)]"
            >
              Reset
            </button>
            <button onClick={onClose} className="btn-pixel px-5 py-2 rounded-xl">
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}