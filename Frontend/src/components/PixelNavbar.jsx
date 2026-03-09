// src/components/PixelNavbar.jsx
export default function PixelNavbar({
  brand = "NeuroLearn",
  title,
  subtitle,
  links = [],
  rightButtonLabel = "Sign in",
  onRightButton,
  onSettings,
  rightSlot,
}) {
  const isGameMode = title != null;
  const hasLinks = Array.isArray(links) && links.length > 0;

  return (
    <div className="sticky top-0 z-50">
      <div className="backdrop-blur-md border-b border-white/[0.06] bg-[rgba(15,23,42,0.82)]">
        <div className="max-w-6xl mx-auto px-5 md:px-8 h-16 flex items-center justify-between">
          {/* Left */}
          <div className="min-w-0 shrink-0">
            {isGameMode ? (
              <div className="text-base font-semibold text-slate-100 truncate max-w-[200px] md:max-w-sm" title={subtitle}>
                {subtitle}
              </div>
            ) : (
              <span className="text-lg font-semibold text-white tracking-tight">{brand}</span>
            )}
          </div>

          {/* Center: optional nav links (only if provided) */}
          {!isGameMode && hasLinks && (
            <nav className="hidden md:flex items-center gap-6">
              {links.map((l) => (
                <a
                  key={l.href}
                  href={l.href}
                  className="text-sm font-medium text-slate-300 hover:text-white transition"
                >
                  {l.label}
                </a>
              ))}
            </nav>
          )}

          {/* Right */}
          <div className="flex items-center gap-2 shrink-0">
            {isGameMode ? (
              <>
                {rightSlot}
                {onSettings && (
                  <button
                    onClick={onSettings}
                    className="p-2 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition text-base"
                    title="Settings"
                    aria-label="Settings"
                  >
                    ⚙️
                  </button>
                )}
              </>
            ) : (
              onRightButton && (
                <button
                  onClick={onRightButton}
                  className="px-5 py-2.5 rounded-lg bg-[rgb(94,234,212)] text-slate-900 font-semibold text-sm hover:brightness-110 transition"
                >
                  {rightButtonLabel}
                </button>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
}