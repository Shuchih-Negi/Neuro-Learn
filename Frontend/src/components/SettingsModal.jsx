// src/components/SettingsModal.jsx
export default function SettingsModal({
  open,
  onClose,
  settings,
  setSettings,
}) {
  if (!open) return null;

  const Row = ({ label, hint, right }) => (
    <div className="flex items-start justify-between gap-4 py-3">
      <div>
        <div className="font-semibold text-slate-900">{label}</div>
        {hint && <div className="text-sm text-slate-600 mt-0.5">{hint}</div>}
      </div>
      <div className="shrink-0">{right}</div>
    </div>
  );

  const Toggle = ({ value, onChange }) => (
    <button
      onClick={() => onChange(!value)}
      className={`h-9 w-16 rounded-full border transition relative ${
        value
          ? "border-sky-300 bg-sky-100"
          : "border-slate-200 bg-white hover:bg-slate-50"
      }`}
      aria-pressed={value}
    >
      <span
        className={`absolute top-1/2 -translate-y-1/2 h-7 w-7 rounded-full transition ${
          value ? "left-8 bg-sky-600" : "left-1 bg-slate-300"
        }`}
      />
    </button>
  );

  const Pill = ({ active, children, onClick }) => (
    <button
      onClick={onClick}
      className={`px-3 py-2 rounded-2xl border text-sm transition ${
        active
          ? "border-sky-300 bg-sky-100 text-slate-900"
          : "border-slate-200 bg-white hover:bg-slate-50 text-slate-700"
      }`}
    >
      {children}
    </button>
  );

  return (
    <div className="fixed inset-0 z-50">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-slate-900/30"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="absolute inset-0 flex items-center justify-center p-4">
        <div className="w-full max-w-lg rounded-[28px] border border-slate-200 bg-white shadow-xl overflow-hidden">
          {/* Header */}
          <div className="p-5 bg-gradient-to-br from-cyan-50 to-white border-b border-slate-200">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xl font-bold">Settings</div>
                <div className="text-sm text-slate-600 mt-0.5">
                  Make it calmer, faster, or more kid-friendly ✨
                </div>
              </div>
              <button
                onClick={onClose}
                className="rounded-2xl border border-slate-200 bg-white hover:bg-slate-50 px-3 py-2 text-sm"
              >
                ✕
              </button>
            </div>
          </div>

          {/* Body */}
          <div className="p-5">
            <Row
              label="Sound effects"
              hint="Soft chime on correct / wrong."
              right={
                <Toggle
                  value={settings.sound}
                  onChange={(v) => setSettings((s) => ({ ...s, sound: v }))}
                />
              }
            />

            <div className="border-t border-slate-200" />

            <Row
              label="Calm Mode"
              hint="Less motion and simpler visual intensity."
              right={
                <Toggle
                  value={settings.calm}
                  onChange={(v) => setSettings((s) => ({ ...s, calm: v }))}
                />
              }
            />

            <Row
              label="Auto-Calm"
              hint="Turn on Calm automatically when drifting / overwhelmed."
              right={
                <Toggle
                  value={settings.autoCalm}
                  onChange={(v) => setSettings((s) => ({ ...s, autoCalm: v }))}
                />
              }
            />

            <div className="border-t border-slate-200" />

            <Row
              label="Big text"
              hint="Bigger font + buttons (kids-friendly)."
              right={
                <Toggle
                  value={settings.bigText}
                  onChange={(v) => setSettings((s) => ({ ...s, bigText: v }))}
                />
              }
            />

            <div className="border-t border-slate-200" />

            <div className="py-3">
              <div className="font-semibold text-slate-900">Sprint length</div>
              <div className="text-sm text-slate-600 mt-0.5">
                Choose a focus timer for quick sessions.
              </div>

              <div className="mt-3 flex gap-2 flex-wrap">
                {[60, 90, 120].map((sec) => (
                  <Pill
                    key={sec}
                    active={settings.sprintSeconds === sec}
                    onClick={() =>
                      setSettings((s) => ({ ...s, sprintSeconds: sec }))
                    }
                  >
                    {sec}s
                  </Pill>
                ))}
              </div>
            </div>

            <div className="mt-4 rounded-3xl border border-slate-200 bg-slate-50 p-4">
              <div className="text-sm font-semibold">Tip for ADHD</div>
              <div className="text-sm text-slate-700 mt-1 leading-relaxed">
                If you feel stuck: enable <span className="font-semibold">Calm Mode</span>,
                take one slow breath, then try again. Small wins count.
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="p-5 border-t border-slate-200 flex items-center justify-between">
            <button
              onClick={() =>
                setSettings({
                  sound: true,
                  calm: false,
                  autoCalm: true,
                  bigText: false,
                  sprintSeconds: 60,
                })
              }
              className="rounded-2xl border border-slate-200 bg-white hover:bg-slate-50 px-4 py-2 text-sm"
            >
              Reset
            </button>

            <button
              onClick={onClose}
              className="rounded-2xl bg-sky-600 text-white hover:bg-sky-500 px-5 py-2 text-sm"
            >
              Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}