import { useEffect, useId, useState } from "react";
import { UiButton } from "../components/ui.jsx";
import { mixingCalculation, rmsCalculation } from "./sound-derivation-math.js";
import "../styles/sound-intuition.css";

const topics = ["① 看懂 RMS", "② 拧小背景", "③ 算出 12 dB"];
const recordings = [
  { label: "轻轻摆", samples: [1, -1, 1, -1] },
  { label: "放大两倍", samples: [2, -2, 2, -2] },
  { label: "有强有弱", samples: [1, -3, 3, -1] }
];
const rmsScenes = [
  ["先看摆动，暂时不看公式", "蓝点的位置就是一次采样。正数在右边，负数在左边；两边都在动。点下面的数字，让蓝点走一遍。"],
  ["把每次摆动，画成一块正方形", "拿采样值的绝对值当边长。+1和−1都画成边长1；每块的面积就是这个采样的平方。"],
  ["把面积加起来，再平均分成四份", "我们要比较每次采样的平均强度。同样的波形重复一遍，记录会变长，平均强度仍然一样。"],
  ["最后，找回这个正方形的边长", "我们手里是平均面积，但音频旋钮乘的是振幅。找一个数，让它乘自己等于平均面积——这件事就叫开平方根。"]
];
const dbScenes = [
  ["先把“dB 刻度”翻译成“功率倍数”", "分贝是一把压缩过的比例尺：功率比每乘10，刻度就加10 dB。下面的平方和比例都使用相同测量条件。"],
  ["主讲当作 1，背景应该留多少面积？", "SIR比较的是主讲除以背景。已知主讲是1，就把1除以刚才的倍数，得到背景允许留下的功率比例。"],
  ["音频乘的是边长，所以再开一次根号", "黄色面积代表功率比例，蓝色边长代表振幅比例。找到黄色正方形的边长，就是背景每个采样要乘的数。"],
  ["公式，只是把刚才的三步写短", "现在再把具体的dB数值换成字母D。你已经算过这条公式，字母只是为了让同一个做法适用于别的数。"]
];

function number(value, digits = 3) {
  return Number(value.toFixed(digits)).toString();
}

function AreaSquare({ side, maximum = 3, label, detail, tone = "amber", showSide = true }) {
  const pixels = 78 * side / maximum;
  return <figure className={`intuition-square ${tone}`}>
    <figcaption>{label}</figcaption>
    <svg viewBox="0 0 140 142" role="img" aria-label={`${label}：${showSide ? `边长${number(side, 6)}，` : "求边长，"}面积${number(side * side, 6)}`}>
      <rect className="square-frame" x="31" y="18" width="78" height="78" />
      <rect className="square-area" x="31" y={96 - pixels} width={pixels} height={pixels} />
      {[1, 2].map(position => <g className="square-grid" key={position}><path d={`M${31 + position * 26} 18 V96`} /><path d={`M31 ${18 + position * 26} H109`} /></g>)}
      <path className="square-edge" d={`M31 101 h${pixels}`} />
      <text x="70" y="120" textAnchor="middle">边长 {showSide ? number(side) : "？"}</text>
      <text className="square-area-label" x="70" y="137" textAnchor="middle">面积 {number(side * side, 4)}</text>
    </svg>
    {detail && <p>{detail}</p>}
  </figure>;
}

function MixingWaves({ gain }) {
  const wave = (center, amplitude) => Array.from({ length: 121 }, (_, index) => {
    const horizontal = 100 + index * 4;
    const vertical = center - Math.sin(index / 120 * Math.PI * 8) * 23 * amplitude;
    return `${index === 0 ? "M" : "L"}${horizontal.toFixed(2)} ${vertical.toFixed(2)}`;
  }).join(" ");
  return <svg className="intuition-waves" viewBox="0 0 600 160" role="img" aria-label={`主讲振幅保持1，背景振幅为${number(gain)}，两条波形使用相同纵轴比例`}>
    <text x="12" y="49">主讲 × 1</text><text x="12" y="122">背景 × {number(gain)}</text>
    <path className="wave-baseline" d="M100 45 H580 M100 118 H580" />
    <path className="wave-target" d={wave(45, 1)} /><path className="wave-background" d={wave(118, gain)} />
    <text x="340" y="154" textAnchor="middle">两条波形的纵轴比例相同 · 横轴都是时间</text>
  </svg>;
}

function Checkpoint({ question, choices, correct, explanation }) {
  const [answer, setAnswer] = useState(null);
  return <div className="intuition-checkpoint"><strong>{question}</strong><div>{choices.map((choice, index) => <UiButton key={choice} aria-pressed={answer === index} onClick={() => setAnswer(index)}>{choice}</UiButton>)}</div>{answer !== null && <p role="status">{answer === correct ? "对，" : "再看一下，"}{explanation}</p>}</div>;
}

export function SoundIntuition({ sirDb, onSirChange, onPauseMain }) {
  const prefix = useId();
  const [topic, setTopic] = useState(0);
  const [recording, setRecording] = useState(0);
  const [rmsScene, setRmsScene] = useState(0);
  const [dbScene, setDbScene] = useState(0);
  const [sample, setSample] = useState(0);
  const [gain, setGain] = useState(0.5);
  const [guess, setGuess] = useState(0.5);
  const [playing, setPlaying] = useState(false);
  const rms = rmsCalculation(recordings[recording].samples);
  const mix = mixingCalculation(sirDb);
  const scene = topic === 0 ? rmsScene : dbScene;
  const [title, explanation] = (topic === 0 ? rmsScenes : dbScenes)[scene];

  useEffect(() => {
    if (!playing || topic === 1) return undefined;
    const timer = setTimeout(() => {
      if (scene === 3) setPlaying(false);
      else if (topic === 0) setRmsScene(current => current + 1);
      else setDbScene(current => current + 1);
    }, 7000);
    return () => clearTimeout(timer);
  }, [playing, scene, topic]);
  useEffect(() => {
    if (!playing || topic !== 0 || rmsScene !== 0) return undefined;
    const timer = setInterval(() => setSample(current => (current + 1) % 4), 1000);
    return () => clearInterval(timer);
  }, [playing, topic, rmsScene]);
  useEffect(() => {
    const pause = () => { if (document.hidden) setPlaying(false); };
    document.addEventListener("visibilitychange", pause);
    return () => document.removeEventListener("visibilitychange", pause);
  }, []);

  function chooseTopic(index) {
    setPlaying(false);
    onPauseMain();
    setTopic(index);
  }
  function chooseScene(index) {
    setPlaying(false);
    onPauseMain();
    if (topic === 0) setRmsScene(index);
    else setDbScene(index);
  }
  function changeSir(value) {
    setPlaying(false);
    onPauseMain();
    onSirChange(value);
  }

  return <div className="sound-intuition" id="sound-intuition">
    <div className="intuition-intro"><span>先做实验，再认识字母</span><h3>蓝色看边长，黄色看面积。</h3><p>把“声音强弱”变成能看见的图形。一次只做一个动作，最后再把动作写成公式。</p></div>
    <div className="intuition-tabs" role="tablist" aria-label="直觉实验">{topics.map((label, index) => <UiButton key={label} role="tab" id={`${prefix}-tab-${index}`} aria-controls={`${prefix}-panel-${index}`} aria-selected={topic === index} tabIndex={topic === index ? 0 : -1} onClick={() => chooseTopic(index)} onKeyDown={event => {
      const next = event.key === "ArrowRight" ? (index + 1) % topics.length : event.key === "ArrowLeft" ? (index + topics.length - 1) % topics.length : event.key === "Home" ? 0 : event.key === "End" ? topics.length - 1 : null;
      if (next !== null) { event.preventDefault(); chooseTopic(next); document.getElementById(`${prefix}-tab-${next}`)?.focus(); }
    }}>{label}</UiButton>)}</div>
    <div className="intuition-panel" role="tabpanel" id={`${prefix}-panel-${topic}`} aria-labelledby={`${prefix}-tab-${topic}`} tabIndex={0}>
      {topic !== 1 && <ol className="intuition-stepper" aria-label="实验步骤">{(topic === 0 ? ["看摆动", "画面积", "平均分", "找边长"] : ["翻译刻度", "求背景面积", "找振幅边长", "写成公式"]).map((label, index) => <li key={label}><UiButton aria-current={scene === index ? "step" : undefined} onClick={() => chooseScene(index)}><span>{index + 1}</span>{label}</UiButton></li>)}</ol>}
      <div className="intuition-title"><span>{topic === 1 ? "亲手拖动" : `镜头 ${scene + 1} / 4`}</span><h4>{topic === 1 ? "背景振幅减半，功率会减到多少？" : title}</h4></div>
      <p className="intuition-explain">{topic === 1 ? "主讲固定为1，背景原本也为1。拖动旋钮，让背景波形和正方形一起缩小；观察蓝色边长和黄色面积变化的速度。" : explanation}</p>

      {topic === 0 && <>
        <div className="intuition-presets" aria-label="选择教学波形">{recordings.map((item, index) => <UiButton key={item.label} aria-pressed={recording === index} onClick={() => { setRecording(index); setGuess(0.5); setPlaying(false); onPauseMain(); }}>{item.label} <small>[{item.samples.join(", ")}]</small></UiButton>)}</div>
        {rmsScene === 0 ? <div className="intuition-motion">
          <svg viewBox="0 0 420 135" role="img" aria-label={`当前采样${rms.samples[sample]}，正负表示左右位置`}><path className="motion-rail" d="M40 60 H380" />{[-3, -2, -1, 0, 1, 2, 3].map(value => <g key={value}><path className="motion-tick" d={`M${210 + value * 50} 52 v16`} /><text x={210 + value * 50} y="95" textAnchor="middle">{value > 0 ? `+${value}` : value}</text></g>)}<circle className="motion-puck" cx={210 + rms.samples[sample] * 50} cy="60" r="13" /><text className="motion-label" x="210" y="126" textAnchor="middle">左边 ← 位置 → 右边</text></svg>
          <div className="intuition-samples">{rms.samples.map((value, index) => <UiButton key={index} aria-pressed={sample === index} aria-label={`第${index + 1}个采样：${value}`} onClick={() => { setSample(index); setPlaying(false); }}><small>第{index + 1}次</small><b>{value > 0 ? `+${value}` : value}</b></UiButton>)}</div>
          <div className="intuition-line"><span>直接平均</span><code>({rms.samples.join(" + ")}) ÷ 4 = {rms.mean}</code><p>中心位置是0，但蓝点确实在左右摆动。我们要衡量的是摆动强弱。</p></div>
        </div> : <>
          {rmsScene < 3 && <div className="intuition-squares">{rms.samples.map((value, index) => <AreaSquare key={index} side={rmsScene === 1 ? Math.abs(value) : rms.rms} label={`第${index + 1}次${rmsScene === 1 ? `：${value}` : "：分到的面积"}`} detail={rmsScene === 1 ? `${Math.abs(value)} × ${Math.abs(value)} = ${value ** 2}` : `${rms.squareSum} ÷ 4 = ${rms.meanSquare}`} />)}</div>}
          {rmsScene === 1 && <div className="intuition-line"><span>每次的平方值</span><code>[{rms.squares.join(", ")}]</code><p>试点“放大两倍”：边长从1变2，面积就从1变4。平方，是在记录这种强度变化。</p></div>}
          {rmsScene === 2 && <div className="intuition-line"><span>合起来 → 平均分</span><code>({rms.squares.join(" + ")}) ÷ 4 = {rms.meanSquare}</code><p>原来的四块面积合计{rms.squareSum}；重新分成四块，每块面积{rms.meanSquare}，总量没有变。</p></div>}
          {rmsScene === 3 && <>
            <div className="intuition-root-game"><AreaSquare side={rms.rms} label={`目标：面积 ${rms.meanSquare}`} detail="这是上一镜头算出的平均面积" showSide={false} /><div className="intuition-root-controls"><label htmlFor={`${prefix}-root`}>试着找边长：<b>{number(guess, 4)}</b></label><input id={`${prefix}-root`} type="range" min="0" max="3" step="any" value={guess} onChange={event => { setGuess(Number(event.target.value)); setPlaying(false); }} /><code>{number(guess, 4)} × {number(guess, 4)} ≈ {number(guess * guess, 4)}</code><p role="status">{Math.abs(guess * guess - rms.meanSquare) < 0.025 ? "找到了！它乘自己，约等于目标面积。这个边长就是RMS。" : `当前面积${number(guess * guess, 3)}，${guess * guess < rms.meanSquare ? "还小了，把边长调大" : "太大了，把边长调小"}。`}</p><UiButton onClick={() => { setGuess(rms.rms); setPlaying(false); }}>帮我找到边长</UiButton></div><AreaSquare side={guess} label="你正在尝试的边长" tone="blue" /></div>
            <div className="intuition-line"><span>开根号 = 从面积找边长</span><code>√{rms.meanSquare} ≈ {number(rms.rms, 6)}</code><p>RMS就是这个等效边长。它与原波形有相同的均方值，不是在预测人耳听起来有多响。</p></div>
            <details className="intuition-formula"><summary>现在把四个动作缩成公式</summary><div className="intuition-formula-chain"><span>每个数平方 <b>y[n]²</b></span><i>→</i><span>全部加起来 <b>Σ y[n]²</b></span><i>→</i><span>除以个数 <b>Σ y[n]² / N</b></span><i>→</i><span>从面积找边长 <b>√(Σ y[n]² / N)</b></span></div><p><b>y</b>是整段声音，<b>n</b>是从0开始的采样编号，<b>y[n]</b>是第n个数；<b>N</b>是数的总个数，本例为4；<b>Σ</b>表示求和，<b>√</b>表示找一个非负数，使它的平方等于括号里的值。<b>RMS</b>是这四个动作得到的有效振幅。</p></details>
          </>}
          <details className="intuition-physics"><summary>为什么强度用平方衡量，而不是只去掉负号？</summary><p>正方形是用来展示平方的画法；采用平方，还有物理来由。同一个电阻上，电压变成2倍，电流也变成2倍；功率是电压乘电流，因此变成4倍。</p><code>电流 I = U / R；功率 P = U × I = U² / R</code><p><b>U</b>是电压（伏），<b>I</b>是电流（安），<b>R</b>是电阻（欧），<b>P</b>是功率（瓦）。例如R=1欧：U=1伏时I=1安、P=1瓦；U=2伏时I=2安、P=4瓦。</p><p>RMS采用“同样平均功率对应多大有效振幅”的思路。数字音频使用均方值作相对功率指标；未经校准的采样平方不能直接当作瓦特。只取绝对值再平均也能衡量幅度，但不会保留这里的平方功率关系。</p></details>
        </>}
      </>}

      {topic === 1 && <>
        <label className="intuition-gain" htmlFor={`${prefix}-gain`}>背景振幅旋钮 <strong>{(gain * 100).toFixed(1)}%</strong><input id={`${prefix}-gain`} type="range" min="0.05" max="1" step="0.01" value={gain} onChange={event => { setGain(Number(event.target.value)); onPauseMain(); }} /></label>
        <div className="intuition-presets">{[1, 0.5, 0.25].map(value => <UiButton key={value} aria-pressed={gain === value} onClick={() => { setGain(value); onPauseMain(); }}>{value === 1 ? "一样强" : value === 0.5 ? "背景振幅减半" : "再减半"}</UiButton>)}</div>
        <MixingWaves gain={gain} />
        <div className="intuition-comparison"><AreaSquare side={1} maximum={1} label="主讲：边长固定为1" detail="功率指标固定为1" tone="blue" /><AreaSquare side={gain} maximum={1} label={`背景：边长 ${number(gain)}`} detail={`功率指标 ${number(gain ** 2, 4)}`} /><div className="intuition-power-result"><span>每个背景采样乘</span><strong>{number(gain)}</strong><p>振幅剩 <b>{number(gain * 100, 1)}%</b></p><p>功率剩 <b>{number(gain ** 2 * 100, 2)}%</b></p><p>主讲功率是背景的 <b>{number(1 / gain ** 2, 3)}倍</b></p><small>同一个比值，分贝尺上读作 {(20 * Math.log10(1 / gain)).toFixed(2)} dB。</small></div></div>
        <div className="intuition-line"><span>用手算检查</span><code>{number(gain)} × {number(gain)} = {number(gain ** 2, 4)}</code><p>两条边一起缩小，面积缩小得更快。比如边长只留1/4，面积就只留1/16。</p></div>
        <div className="intuition-line"><span>真正混音时，程序做什么？</span><code>背景样本 0.6 × {number(gain)} = {number(0.6 * gain, 4)}</code><p>对背景的每一个采样都乘同一个系数a，本例a={number(gain)}；再与对应时刻的主讲采样相加，得到混合输入。</p></div>
        <Checkpoint question="背景振幅只留一半，功率会留多少？" choices={["1/2", "1/4", "不变"]} correct={1} explanation="两条边都乘1/2，面积就是1/2×1/2=1/4。" />
      </>}

      {topic === 2 && <>
        <div className="intuition-assumption">本实验先把主讲、原背景调到相同RMS。主讲保持为1，只缩小背景。</div>
        <label className="intuition-gain" htmlFor={`${prefix}-db`}>想让主讲比背景强 <strong>{sirDb} dB</strong><input id={`${prefix}-db`} type="range" min="0" max="24" step="3" value={sirDb} onChange={event => changeSir(Number(event.target.value))} /></label>
        <div className="intuition-presets">{[0, 6, 12, 24].map(value => <UiButton key={value} aria-pressed={sirDb === value} onClick={() => changeSir(value)}>{value} dB</UiButton>)}</div>
        {dbScene === 0 && <div className="intuition-decibel-ruler"><strong>分贝，是给“功率倍数”换一套刻度</strong><div>{[[0, 1], [10, 10], [20, 100], [30, 1000]].map(([decibels, ratio]) => <article key={decibels}><b>{decibels} dB</b><span>↕</span><b>{ratio}倍</b></article>)}</div><p>每往右一格：上面加10，下面乘10。当前{sirDb} dB对应约{number(mix.powerRatio, 4)}倍。</p><details><summary>10 的几次方，是什么意思？</summary><p>10¹=10，10²=10×10=100。指数也可以有小数：10^1.2≈15.849，位于10与100之间。把{sirDb} dB除以10，就是这把尺上的指数{number(sirDb / 10)}。</p><p>反过来问“10的几次方等于这个倍数”，就是log₁₀。分贝对功率比的定义是：D=10×log₁₀(主讲功率/背景功率)。D是以dB表示的SIR；前面的10是刻度定义。</p></details></div>}
        {dbScene > 0 && <div className="intuition-comparison"><AreaSquare side={1} maximum={1} label="主讲：功率指标 1" tone="blue" /><AreaSquare side={mix.amplitudeRatio} maximum={1} label={`背景：面积 ${number(mix.backgroundPower, 6)}`} detail={dbScene >= 2 ? `边长≈${number(mix.amplitudeRatio, 6)}` : "下一镜头：从面积找边长"} showSide={dbScene >= 2} /><div className="intuition-power-result"><span>{dbScene === 1 ? "背景应留的功率比例" : "背景每个采样乘"}</span><strong>{number(dbScene === 1 ? mix.backgroundPower : mix.amplitudeRatio, 6)}</strong>{dbScene >= 2 && <p>振幅约 {number(mix.amplitudeRatio * 100, 1)}%</p>}<p>功率约 {number(mix.backgroundPower * 100, 2)}%</p></div></div>}
        <div className="intuition-number-trail" aria-label="本次计算路线">
          <article className={dbScene === 0 ? "active" : ""}><span>① 刻度 → 功率倍数</span><code>10^({sirDb} ÷ 10)</code><b>= {number(mix.powerRatio, 6)}</b></article>
          {dbScene >= 1 && <article className={dbScene === 1 ? "active" : ""}><span>② 背景面积</span><code>1 ÷ {number(mix.powerRatio, 6)}</code><b>≈ {number(mix.backgroundPower, 6)}</b></article>}
          {dbScene >= 2 && <article className={dbScene === 2 ? "active" : ""}><span>③ 背景边长</span><code>√{number(mix.backgroundPower, 6)}</code><b>≈ {number(mix.amplitudeRatio, 6)}</b></article>}
        </div>
        {dbScene === 3 && <div className="intuition-formula-reveal"><h5>把上面三个动作，接成同一句：</h5><div><code>a = √(1 / 10^(D/10))</code><p><b>a</b>是背景样本要乘的数；<b>D</b>是希望达到的SIR，单位dB。1除以功率倍数，再开根号。</p></div><div><code>= √(10^(−D/10))</code><p><b>负号</b>：1除以10的某次方，就等于10的负次方。例如1/10²=10⁻²=0.01。</p></div><div><code>= 10^(−D/20)</code><p><b>20的由来</b>：开根号把指数减半。−D÷10再除以2，就是−D÷20。例如√(10⁻²)=√0.01=0.1=10⁻¹。</p></div><p>本次代入：a=10^(−{sirDb}/20)≈{number(mix.gain, 6)}。D越大，背景允许留下的面积越小，边长也越小。</p><p className="intuition-assumption">两源RMS不相等时，要先校正强弱：a=(Rₛ/Rᵥ)×10^(−D/20)。Rₛ是主讲RMS，Rᵥ是背景缩放前RMS。下方完整推导可以切换验证。</p></div>}
        {dbScene === 3 && <Checkpoint question="这里除以20，是哪两步合在一起？" choices={["分贝尺除以10，再开根号", "固定把背景降低20倍"]} correct={0} explanation="分贝尺先给出功率比例；开根号把功率比例变成振幅比例，指数也随之减半。" />}
      </>}

      {topic !== 1 && <div className="intuition-playback"><UiButton onClick={() => { onPauseMain(); if (scene === 3) { if (topic === 0) setRmsScene(0); else setDbScene(0); } setPlaying(!playing); }}>{playing ? "暂停演示" : "自动演示（每镜头7秒）"}</UiButton><UiButton disabled={scene === 0} onClick={() => chooseScene(scene - 1)}>← 上一镜头</UiButton><UiButton className="primary" disabled={scene === 3} onClick={() => chooseScene(scene + 1)}>下一镜头 →</UiButton></div>}
      <div className="intuition-next-topic">{topic < 2 && <UiButton className="primary" onClick={() => chooseTopic(topic + 1)}>{topic === 0 ? "接着拧一拧背景音量 →" : "反过来：给12 dB，怎么算旋钮位置？ →"}</UiButton>}</div>
      <p className="intuition-caption">图形展示真实计算的相对比例；正方形面积是平方指标的画法。这里使用教学数字，不采集音频，也不运行Hush。背景系数用于准备测试混音，不是模型内部的增强强度。</p>
    </div>
  </div>;
}
