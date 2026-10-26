import { Fragment, useEffect, useMemo, useReducer, useState } from "react";

import { UiButton } from "../components/ui.jsx";
import lessons from "../lib/tutorial-lessons.js";
import guidance from "../lib/lesson-guidance.js";
import { readStored, writeStored } from "../lib/storage.js";
import { CanvasStage } from "./CanvasStage.jsx";
import { getFrameCount } from "./canvas-engine.js";
import { createTutorialState, tutorialReducer } from "./tutorial-state.js";

export function TutorialPage() {
  const [state, dispatch] = useReducer(tutorialReducer, null, () => createTutorialState(location.hash.slice(1), readStored("hush-math-visited", "[]")));
  const [sceneResult, setSceneResult] = useState("");
  const lesson = lessons[state.selected];
  const lessonGuide = guidance[lesson.id];
  const frames = getFrameCount(lesson, state.parameters);
  useEffect(() => { writeStored("hush-math-visited", JSON.stringify(state.visited)); }, [state.visited]);
  useEffect(() => {
    const onHash = () => { const index = lessons.findIndex(item => item.id === location.hash.slice(1)); if (index >= 0 && index !== state.selected) dispatch({ type: "choose", index }); };
    window.addEventListener("hashchange", onHash); return () => window.removeEventListener("hashchange", onHash);
  }, [state.selected]);
  useEffect(() => {
    if (!state.playing) return undefined;
    let last = performance.now();
    let animation;
    const tick = now => {
      const portion = Math.min(1, (now - last) / Number(state.speed));
      if (portion >= 1) { last = now; dispatch({ type: "advance" }); }
      else dispatch({ type: "fraction", value: portion });
      animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animation);
  }, [state.playing, state.speed]);
  useEffect(() => {
    history.replaceState(null, "", `#${lessons[state.selected].id}`);
  }, [state.selected]);
  useEffect(() => {
    const pauseWhenHidden = () => { if (document.hidden) dispatch({ type: "pause" }); };
    document.addEventListener("visibilitychange", pauseWhenHidden);
    return () => document.removeEventListener("visibilitychange", pauseWhenHidden);
  }, []);
  const choose = (index, scroll = false) => { const next = Math.max(0, Math.min(lessons.length - 1, index)); dispatch({ type: "choose", index: next }); history.replaceState(null, "", `#${lessons[next].id}`); if (scroll) document.getElementById("lesson")?.scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); };
  const result = useMemo(() => sceneResult || `步骤 ${state.frame + 1} / ${frames} · ${lesson.title} · 教学数字由浏览器本地计算`, [sceneResult, state.frame, frames, lesson.title]);
  const formulaMap = lessons.map((item, index) => <article key={item.id}><h3>{String(index + 1).padStart(2, "0")} · {item.title}</h3><pre>{item.equations.join("\n\n")}</pre></article>);
  return <main className="course">
    <header className="masthead"><a href="./" className="brand">HUSH / RUNTIME LAB</a><a href="./">打开真实推理回放 ↗</a></header>
    <section className="intro"><div><p className="eyebrow">给第一次接触音频与 GPU 的你</p><h1>一段声音的<br /><span>数学旅行。</span></h1><p className="intro-copy">从麦克风里的一串数字，走到频谱、神经网络和 GPU 线程。每次只学一个原理，亲手拨动参数，看答案怎样算出来。</p></div><div className="route" aria-label="课程主线"><span>01 / 听见数字</span><i>↓</i><span>02 / 学会乘加</span><i>↓</i><span>03 / 理解增强</span><i>↓</i><span>04 / 走进 GPU</span></div></section>
    <div className="course-info"><span>18 节短课 · 零基础起步 · 每课可单步演算</span><span id="progress">{`已读 ${state.visited.length} / ${lessons.length}`}</span></div>
    <details className="lesson-menu" id="lessonMenu" open><summary>课程目录 <span>点任意一课开始</span></summary><nav id="chapters" aria-label="选择课程">{lessons.map((item, index) => <UiButton key={item.id} aria-current={index === state.selected ? "step" : undefined} onClick={() => choose(index, true)}><small>{String(index + 1).padStart(2, "0")} / {item.category.split(" / ")[1]}</small>{item.title}</UiButton>)}</nav></details>
    <section className="lesson" id="lesson"><header className="lesson-heading"><div><p className="eyebrow" id="lessonNumber">{`${lesson.category} · LESSON ${String(state.selected + 1).padStart(2, "0")}`}</p><h2 id="lessonTitle">{lesson.title}</h2></div><span className="tag" id="lessonLocation">{lesson.category.split(" / ")[1]}</span></header><p className="question" id="lessonQuestion">{lesson.question}</p>
      <div className="stage"><div className="stage-head"><span id="sceneTitle">{lesson.title} · 数字可以自己验算</span><span className="tag">教学数字 · 浏览器本地计算</span></div><CanvasStage lesson={lesson} frame={state.frame} fraction={state.fraction} parameters={state.parameters} onResult={setSceneResult} /><output id="sceneResult" aria-live="polite">{result}</output><div className="animation-controls"><UiButton className="primary" onClick={() => dispatch({ type: "play" })}>{state.playing ? "Ⅱ 暂停动画" : "▶ 播放动画"}</UiButton><UiButton onClick={() => dispatch({ type: "step" })}>一步计算 →</UiButton><UiButton onClick={() => dispatch({ type: "restart" })}>重演</UiButton><label className="speed">速度 <select value={state.speed} onChange={event => dispatch({ type: "speed", value: event.target.value })}><option value="1800">慢慢看</option><option value="1050">正常</option><option value="500">快一点</option></select></label><span className="count" id="stepCount">{`步骤 ${state.frame + 1} / ${frames}`}</span></div><div className="parameter-controls" id="parameters">{lesson.controls.map(control => <div className="parameter" key={control.id}><label htmlFor={`parameter-${control.id}`}>{control.label}<span>{control.options ? "" : `${state.parameters[control.id] ?? control.value}${control.unit || ""}`}</span></label>{control.options ? <select id={`parameter-${control.id}`} value={state.parameters[control.id] ?? control.value} onChange={event => dispatch({ type: "control", id: control.id, value: event.target.value })}>{control.options.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select> : <input id={`parameter-${control.id}`} type="range" min={control.min} max={control.max} step={control.step} value={state.parameters[control.id] ?? control.value} onChange={event => dispatch({ type: "control", id: control.id, value: Number(event.target.value) })} />}</div>)}</div></div>
      <section className="formula-guide" aria-live="polite"><div className="formula-guide-heading"><div><p className="section-label">公式导游 · 跟着动画读</p><h3>{lessonGuide.hook}</h3></div><span className="formula-pulse"><i />第 {state.frame + 1} 步</span></div><p className="formula-reading"><b>整句怎么读：</b>{lessonGuide.read}</p><div className="formula-parts">{lessonGuide.parts.map(([symbol, meaning], index) => <article className={index === state.frame % lessonGuide.parts.length ? "active" : ""} key={symbol}><span className="formula-part-index">{String(index + 1).padStart(2, "0")}</span><code>{symbol}</code><p>{meaning}</p></article>)}</div><div className="formula-example"><span>代入一个数字</span><p>{lessonGuide.example}</p></div><div className="animation-note"><span>动画观察</span><p>{lessonGuide.animation}</p></div></section>
      <div className="reading-grid"><article className="explain"><p className="section-label">把它想成</p><p className="analogy" id="analogy">{lesson.analogy}</p><p id="explanation">{lesson.explanation}</p><div className="walkthrough" id="walkthrough">{lesson.steps.map((step, index) => <div key={step}><b>{String(index + 1).padStart(2, "0")}</b><span>{step}</span></div>)}</div><div className="takeaway"><span>这一课记住</span><p id="takeaway">{lesson.takeaway}</p></div></article><aside className="math"><p className="section-label">现在再看公式</p><div className="equations" id="equations">{lesson.equations.map(equation => <div className="equation" key={equation}>{equation}</div>)}</div><dl id="symbols">{lesson.symbols.map(([symbol, explanation]) => <Fragment key={symbol}><dt>{symbol}</dt><dd>{explanation}</dd></Fragment>)}</dl></aside></div>
      <section className="implementation"><div><p className="section-label">回到你的 runtime</p><p id="implementation">{lesson.implementation}</p></div><div className="sources" id="sources">{lesson.sources.map(source => <a href={`https://github.com/johankoi91/AI_runtime/blob/main/${source}`} target="_blank" rel="noreferrer" key={source}>{source} ↗</a>)}</div></section>
      <section className="quiz"><p className="section-label">停一下，自己想想</p><h3 id="quizQuestion">{lesson.quiz.question}</h3><div id="quizOptions">{lesson.quiz.choices.map((choice, index) => <UiButton key={choice} aria-pressed={state.answer === index} onClick={() => dispatch({ type: "answer", index })}>{choice}</UiButton>)}</div><p id="quizFeedback" role="status" className={state.answer === lesson.quiz.answer ? "correct" : ""}>{state.answer === null ? "先自己选一个答案，再看解释。" : `${state.answer === lesson.quiz.answer ? "答对了。 " : "再想想："}${lesson.quiz.reason}`}</p></section>
      <div className="lesson-navigation"><UiButton disabled={state.selected === 0} onClick={() => choose(state.selected - 1, true)}>← 上一课</UiButton><span id="lessonPosition">{`${state.selected + 1} / ${lessons.length}`}</span><UiButton className="primary" onClick={() => choose((state.selected + 1) % lessons.length, true)}>{state.selected === lessons.length - 1 ? "回到第一课 ↺" : "下一课 →"}</UiButton></div>
    </section>
    <details className="reference"><summary>完整公式地图 / 复习小抄</summary><div id="formulaIndex">{formulaMap}</div></details>
    <footer><p>本教程覆盖当前 Hush 音频链路及 runtime 中的核心数学。小矩阵、波形和权重是教学例子，不是从麦克风或模型导出的隐私数据；这里不运行 Hush ONNX，也不上传任何输入。</p><p>按源码区分 CPU 与 Metal；前端、Mask / DF 和合成当前在 CPU。真实执行时间与设备请查看 <a href="./">推理回放</a>。数值对齐不等于模型在所有场景下都能保留主讲人。</p></footer>
  </main>;
}
