import { useEffect, useReducer, useRef, useState } from "react";
import { UiButton } from "../components/ui.jsx";
import lessons, { lessonById, modules } from "../lib/curriculum.js";
import walkthroughs from "../lib/formula-walkthroughs.js";
import variableGlossary from "../lib/lesson-variable-glossary.js";
import { readStored, writeStored } from "../lib/storage.js";
import { CanvasStage } from "./CanvasStage.jsx";
import { FoundationStage } from "./FoundationStage.jsx";
import { SoundIntuition } from "./SoundIntuition.jsx";
import { getFrameCount } from "./canvas-engine.js";
import { createTutorialState, tutorialReducer } from "./tutorial-state.js";
import "../styles/curriculum.css";

const learningSteps = ["先提一个问题", "动手做实验", "再读懂公式", "用到 Hush 里"];
const formulaSteps = ["整句怎么读", "每个符号", "数字算一遍", "为什么这样算"];

function FormulaWorkshop({ lesson }) {
  const [step, setStep] = useState(0);
  const formula = walkthroughs[lesson.id];
  return <section className="formula-workshop" aria-label="逐步解读公式">
    <div className="workshop-heading"><span className="eyebrow">公式只是把操作写短一点</span><span className="tag">一次读一条</span></div>
    <pre className="workshop-expression">{formula.expression}</pre>
    <div className="workshop-tabs" aria-label="公式阅读步骤">{formulaSteps.map((label, index) => <UiButton key={label} aria-pressed={step === index} onClick={() => setStep(index)}>{index + 1}. {label}</UiButton>)}</div>
    <div className="workshop-content" aria-live="polite"><h3>{formulaSteps[step]}</h3>
      {step === 0 && <p className="workshop-reading">{formula.read}</p>}
      {step === 1 && <dl className="workshop-variables">{formula.variables.map(([symbol, meaning]) => <div key={symbol}><dt>{symbol}</dt><dd>{meaning}</dd></div>)}</dl>}
      {step === 2 && <ol className="workshop-example">{formula.example.map(line => <li key={line}>{line}</li>)}</ol>}
      {step === 3 && <><p className="workshop-reading">{formula.why}</p><div className="workshop-use"><strong>它在这套工程中的作用</strong><p>{lesson.implementation}</p></div></>}
    </div>
    <div className="workshop-controls"><UiButton disabled={step === 0} onClick={() => setStep(step - 1)}>← 上一段</UiButton><span>{step + 1} / {formulaSteps.length}</span><UiButton className="primary" disabled={step === formulaSteps.length - 1} onClick={() => setStep(step + 1)}>继续解读 →</UiButton></div>
    <details className="workshop-reference"><summary>进一步阅读：完整公式与源码约定</summary><p>先理解上面的核心动作，再展开这些完整写法。教学简式与模型多维实现使用相同的计算原则。</p>{lesson.equations.map(equation => <pre key={equation}>{equation}</pre>)}<dl className="workshop-variables">{[...new Map([...(variableGlossary[lesson.id] || []), ...lesson.symbols, ...formula.variables].map(entry => [entry[0], entry])).values()].map(([symbol, meaning]) => <div key={symbol}><dt>{symbol}</dt><dd>{meaning}</dd></div>)}</dl></details>
  </section>;
}

function Experiment({ lesson, state, dispatch }) {
  const [sceneResult, setSceneResult] = useState("");
  const frames = getFrameCount(lesson, state.parameters);
  const pause = () => dispatch({ type: "pause" });
  if (lesson.kind.startsWith("intuition-")) return <div className="course-intuition"><SoundIntuition focusTopic={lesson.kind === "intuition-rms" ? 0 : 2} sirDb={Number(state.parameters.sir ?? 12)} onSirChange={value => dispatch({ type: "control", id: "sir", value })} onPauseMain={pause} /></div>;
  return <div className="stage"><div className="stage-head"><span id="sceneTitle">{lesson.title} · 亲手改变数字</span><span className="tag">教学数据 · 本地计算</span></div>
    {lesson.kind === "foundation" ? <FoundationStage lesson={lesson} frame={state.frame} parameters={state.parameters} onResult={setSceneResult} /> : <CanvasStage lesson={lesson} frame={state.frame} fraction={state.fraction} parameters={state.parameters} onResult={setSceneResult} />}
    <output id="sceneResult" aria-live="polite">{sceneResult || "调整下面的控件，观察图形与结果怎样变化。"}</output>
    <div className="animation-controls"><UiButton className="primary" onClick={() => dispatch({ type: "play" })}>{state.playing ? "Ⅱ 暂停动画" : "▶ 播放动画"}</UiButton><UiButton onClick={() => dispatch({ type: "step" })}>一步计算 →</UiButton><UiButton onClick={() => dispatch({ type: "restart" })}>重演</UiButton><label className="speed">速度 <select value={state.speed} onChange={event => dispatch({ type: "speed", value: event.target.value })}><option value="2400">慢慢看</option><option value="1050">正常</option><option value="500">快一点</option></select></label><span className="count">步骤 {state.frame + 1} / {frames}</span></div>
    <div className="parameter-controls">{lesson.controls.map(control => <div className="parameter" key={control.id}><label htmlFor={`parameter-${control.id}`}>{control.label}<span>{control.options ? "" : `${state.parameters[control.id]}${control.unit || ""}`}</span></label>{control.options ? <select id={`parameter-${control.id}`} value={state.parameters[control.id]} onChange={event => dispatch({ type: "control", id: control.id, value: event.target.value })}>{control.options.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select> : <input id={`parameter-${control.id}`} type="range" min={control.min} max={control.max} step={control.step} value={state.parameters[control.id]} onChange={event => dispatch({ type: "control", id: control.id, value: Number(event.target.value) })} />}</div>)}</div>
  </div>;
}

export function TutorialPage() {
  const [state, dispatch] = useReducer(tutorialReducer, null, () => createTutorialState(location.hash.slice(1), readStored("hush-math-visited", "[]"), readStored("hush-math-completed-v2", "[]")));
  const lesson = lessons[state.selected];
  const moduleIndex = modules.findIndex(module => module.id === lesson.moduleId);
  const module = modules[moduleIndex];
  const workspaceRef = useRef(null);
  useEffect(() => { writeStored("hush-math-visited", JSON.stringify(state.visited)); }, [state.visited]);
  useEffect(() => { writeStored("hush-math-completed-v2", JSON.stringify(state.completed)); }, [state.completed]);
  useEffect(() => {
    const onHash = () => { const index = lessons.findIndex(item => item.id === location.hash.slice(1)); if (index >= 0 && index !== state.selected) { dispatch({ type: "choose", index }); requestAnimationFrame(() => workspaceRef.current?.scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" })); } };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [state.selected]);
  useEffect(() => { history.replaceState(null, "", `#${lesson.id}`); }, [lesson.id]);
  useEffect(() => {
    if (!state.playing) return undefined;
    let last = performance.now();
    let animation;
    const tick = now => {
      const fraction = Math.min(1, (now - last) / Number(state.speed));
      if (fraction >= 1) { last = now; dispatch({ type: "advance" }); }
      else dispatch({ type: "fraction", value: fraction });
      animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animation);
  }, [state.playing, state.speed]);
  useEffect(() => {
    const pause = () => { if (document.hidden) dispatch({ type: "pause" }); };
    document.addEventListener("visibilitychange", pause);
    return () => document.removeEventListener("visibilitychange", pause);
  }, []);
  function scrollToLesson() {
    requestAnimationFrame(() => workspaceRef.current?.scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }));
  }
  function choose(index) { dispatch({ type: "choose", index: Math.max(0, Math.min(lessons.length - 1, index)) }); scrollToLesson(); }
  function chooseId(id) { choose(lessons.findIndex(item => item.id === id)); }
  function learningStep(index) { dispatch({ type: "learning-step", index }); }
  const resumeIndex = lessons.findIndex(item => !state.completed.includes(item.id));
  const finished = state.completed.includes(lesson.id);
  const allFinished = state.completed.length === lessons.length;

  return <main className="course redesigned-course">
    <header className="masthead"><a href="./" className="brand">HUSH / MATH TO RUNTIME</a><a href="./">查看真实推理 Trace ↗</a></header>
    <section className="course-hero"><div><p className="eyebrow">从四则运算开始 · 一步步走到 GPU</p><h1>先看懂数学，<br /><span>再读懂 Hush。</span></h1><p>从一个采样乘0.25开始，走到频谱、神经网络与Metal。每课先动手，再给字母起名字，算一个例子，最后回到你的工程。</p><UiButton className="primary" onClick={() => choose(resumeIndex < 0 ? 0 : resumeIndex)}>{allFinished ? "从第1课重新复习 →" : state.completed.length ? "继续第一节未完成的课 →" : "从第1课开始 →"}</UiButton></div><aside className="course-contract"><span>每节课都走同一条路</span><ol>{learningSteps.map((step, index) => <li key={step}><b>0{index + 1}</b>{step}</li>)}</ol><p>字母含义、公式由来和计算示例分开讲。所有课程都可自由跳转。</p></aside></section>
    <div className="course-info"><span>7 个阶段 · {lessons.length} 节课 · 从基础到完整链路</span><span id="progress">已完成 {state.completed.length} / {lessons.length}</span></div>
    <progress className="course-progress" max={lessons.length} value={state.completed.length} aria-label="课程完成进度" />
    <nav className="learning-map" aria-label="七阶段学习路线">{modules.map((item, index) => <div className={`learning-module${item.id === module.id ? " current" : ""}`} key={item.id}><UiButton className="module-entry" aria-current={item.id === module.id ? "step" : undefined} onClick={() => chooseId(item.ids[0])}><span>阶段 {index + 1} <small>{item.ids.filter(id => state.completed.includes(id)).length}/{item.ids.length} 完成</small></span><strong>{item.title}</strong><p>{item.outcome}</p></UiButton><details open={item.id === module.id}><summary>{item.ids.length} 节课 · 展开目录</summary><ol>{item.ids.map(id => <li key={id}><UiButton aria-current={id === lesson.id ? "step" : undefined} onClick={() => chooseId(id)}><span>{state.completed.includes(id) ? "✓" : String(lessons.indexOf(lessonById[id]) + 1).padStart(2, "0")}</span>{lessonById[id].title}</UiButton></li>)}</ol></details></div>)}</nav>
    <section className="lesson course-workspace" id="lesson" ref={workspaceRef} aria-labelledby="lessonTitle">
      <header className="lesson-heading"><div><p className="eyebrow">阶段 {moduleIndex + 1} / 7 · 第 {state.selected + 1} / {lessons.length} 课</p><h2 id="lessonTitle">{lesson.title}</h2></div><span className="tag">{finished ? "✓ 已完成本课" : "一步一步来"}</span></header>
      <div className="lesson-prerequisites"><span>本课会用到：</span>{lesson.prerequisites.length ? lesson.prerequisites.map(id => <UiButton key={id} onClick={() => chooseId(id)}>{lessonById[id].title} ↗</UiButton>) : <span>会做加减乘除就可以开始。</span>}</div>
      <nav className="learning-stepper" aria-label="本课学习步骤">{learningSteps.map((label, index) => <UiButton key={label} aria-current={state.learningStep === index ? "step" : undefined} onClick={() => learningStep(index)}><span>{index + 1}</span>{label}</UiButton>)}</nav>
      <div className="learning-body" key={`${lesson.id}-${state.selectionVersion}-${state.learningStep}`}>
        {state.learningStep === 0 && <section className="lesson-question"><p className="eyebrow">先带着一个具体问题</p><h3>{lesson.question}</h3><div className="lesson-analogy"><span>先把它想成</span><p>{lesson.analogy}</p></div><p className="lesson-explanation">{lesson.explanation}</p><div className="lesson-goal"><span>这一课，你要带走</span><p>{lesson.takeaway}</p></div></section>}
        {state.learningStep === 1 && <><div className="experiment-instructions"><span className="eyebrow">先观察，再改一个参数</span><p>{lesson.steps[0]}</p><details><summary>这个实验怎么操作？</summary><ol>{lesson.steps.map(step => <li key={step}>{step}</li>)}</ol></details></div><Experiment lesson={lesson} state={state} dispatch={dispatch} /></>}
        {state.learningStep === 2 && <FormulaWorkshop lesson={lesson} />}
        {state.learningStep === 3 && <><section className="runtime-connection"><p className="eyebrow">这个数学动作，在你的项目里做什么</p><h3>{module.outcome}</h3><p>{lesson.implementation}</p><details><summary>对应的源代码位置</summary><div className="sources">{lesson.sources.map(source => <a href={`https://github.com/johankoi91/AI_runtime/blob/main/${source}`} target="_blank" rel="noreferrer" key={source}>{source} ↗</a>)}</div></details></section><section className="quiz"><p className="section-label">用一道小题确认理解</p><h3 id="quizQuestion">{lesson.quiz.question}</h3><div id="quizOptions">{lesson.quiz.choices.map((choice, index) => <UiButton key={choice} aria-pressed={state.answer === index} onClick={() => dispatch({ type: "answer", index })}>{choice}</UiButton>)}</div><p id="quizFeedback" role="status" className={state.answer === lesson.quiz.answer ? "correct" : ""}>{state.answer === null ? "选一个答案，解释会出现在这里。" : `${state.answer === lesson.quiz.answer ? "答对了。" : "再看一下："}${lesson.quiz.reason}`}</p><UiButton className="primary complete-lesson" disabled={state.answer !== lesson.quiz.answer && !finished} onClick={() => { dispatch({ type: "complete" }); if (state.selected < lessons.length - 1) choose(state.selected + 1); }}>{state.selected < lessons.length - 1 ? "记为完成，进入下一课 →" : allFinished ? "✓ 全部课程已完成" : finished ? "✓ 本课已完成" : "完成最后一课 ✓"}</UiButton>{state.selected === lessons.length - 1 && !allFinished && <p className="completion-note">完成这课后，可从顶部继续补齐前面未完成的课程。</p>}</section></>}
      </div>
      <div className="learning-step-navigation"><UiButton disabled={state.learningStep === 0} onClick={() => learningStep(state.learningStep - 1)}>← 返回上一步</UiButton><span>本课 {state.learningStep + 1} / 4</span>{state.learningStep < 3 && <UiButton className="primary" onClick={() => learningStep(state.learningStep + 1)}>{state.learningStep === 0 ? "开始动手实验 →" : state.learningStep === 1 ? "把刚才的动作写成公式 →" : "看看在 Hush 中怎么用 →"}</UiButton>}</div>
      <div className="lesson-navigation"><UiButton disabled={state.selected === 0} onClick={() => choose(state.selected - 1)}>← 上一课</UiButton><span>第 {state.selected + 1} / {lessons.length} 课</span><UiButton disabled={state.selected === lessons.length - 1} onClick={() => choose(state.selected + 1)}>直接看下一课 →</UiButton></div>
    </section>
    <details className="reference"><summary>{lessons.length} 课核心公式地图 · 复习时展开</summary><div id="formulaIndex">{lessons.map((item, index) => <article key={item.id}><h3><a href={`#${item.id}`}>{String(index + 1).padStart(2, "0")} · {item.title}</a></h3><pre>{walkthroughs[item.id].expression}</pre><p>{walkthroughs[item.id].read}</p></article>)}</div></details>
    <footer><p>图形、权重和性能数字是浏览器本地生成的教学例子，不采集麦克风，不运行Hush模型。真实设备和时间请看 <a href="./">推理 Trace</a>。完成进度只保存在当前浏览器，通过本课小题后点“记为完成”才计入。</p><p>课程按当前源码区分CPU与Metal执行路径；理解数学、计算对齐、实时性能和音质效果是不同的学习与验证目标。</p></footer>
  </main>;
}
