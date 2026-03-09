// src/components/PixelCursor.jsx
import { useEffect, useRef } from "react";

export default function PixelCursor() {
  const dotRef = useRef(null);
  const ringRef = useRef(null);

  useEffect(() => {
    const isTouch = "ontouchstart" in window || navigator.maxTouchPoints > 0;
    const reduce =
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (isTouch || reduce) return;

    const dot = dotRef.current;
    const ring = ringRef.current;
    if (!dot || !ring) return;

    let x = window.innerWidth / 2;
    let y = window.innerHeight / 2;
    let rx = x;
    let ry = y;

    const move = (e) => {
      if (document.body.classList.contains("calm")) return;
      x = e.clientX;
      y = e.clientY;
      dot.style.transform = `translate3d(${x}px, ${y}px, 0)`;
      dot.style.opacity = "1";
      ring.style.opacity = "1";
    };

    let raf = 0;
    const loop = () => {
      rx += (x - rx) * 0.18;
      ry += (y - ry) * 0.18;
      ring.style.transform = `translate3d(${rx}px, ${ry}px, 0)`;
      raf = requestAnimationFrame(loop);
    };

    raf = requestAnimationFrame(loop);
    window.addEventListener("mousemove", move, { passive: true });

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("mousemove", move);
    };
  }, []);

  return (
    <>
      <div
        ref={ringRef}
        className="pixel-cursor-ring pointer-events-none fixed left-0 top-0 z-[9999]"
        aria-hidden="true"
      />
      <div
        ref={dotRef}
        className="pixel-cursor-dot pointer-events-none fixed left-0 top-0 z-[10000]"
        aria-hidden="true"
      />
    </>
  );
}