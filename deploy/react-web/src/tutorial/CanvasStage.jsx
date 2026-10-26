import { useCallback, useLayoutEffect, useRef, useState } from "react";

import { renderTutorialFrame } from "./canvas-engine.js";

export function CanvasStage({ lesson, frame, fraction, parameters, onResult }) {
  const canvasRef = useRef(null);
  const propsRef = useRef(null);
  const drawingRef = useRef(null);
  const [label, setLabel] = useState(lesson.title);
  propsRef.current = { lesson, frame, fraction, parameters, onResult };

  const paint = useCallback((announce, resize = false) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (resize || !drawingRef.current) {
      const bounds = canvas.getBoundingClientRect();
      const scale = Math.min(window.devicePixelRatio || 1, 2);
      const width = bounds.width;
      const height = bounds.height;
      const styles = getComputedStyle(document.documentElement);
      const colors = {};
      ["text", "muted", "green", "blue", "purple", "amber", "red", "line", "panel", "stage"].forEach(key => { colors[key] = styles.getPropertyValue(`--${key}`).trim(); });
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      const context = canvas.getContext("2d");
      context.setTransform(scale, 0, 0, scale, 0, 0);
      drawingRef.current = { context, width, height, colors };
    }
    const drawing = drawingRef.current;
    const current = propsRef.current;
    const result = renderTutorialFrame(drawing.context, { ...drawing, ...current });
    if (announce) {
      current.onResult(result);
      setLabel(current.lesson.title + "。当前计算：" + result);
    }
  }, []);

  useLayoutEffect(() => {
    paint(true, true);
    const canvas = canvasRef.current;
    const observer = new ResizeObserver(() => paint(true, true));
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [paint]);
  useLayoutEffect(() => { paint(true); }, [lesson, frame, parameters, paint]);
  useLayoutEffect(() => { paint(false); }, [fraction, paint]);
  return <canvas ref={canvasRef} id="animation" role="img" aria-label={label}>请启用 Canvas，查看逐项计算动画。所有公式和解释仍可阅读。</canvas>;
}
