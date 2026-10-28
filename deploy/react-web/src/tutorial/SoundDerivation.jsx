import { useEffect, useId, useState } from "react";
import { UiButton } from "../components/ui.jsx";
import { mixingCalculation, rmsCalculation, teachingSamples } from "./sound-derivation-math.js";
import { SoundIntuition } from "./SoundIntuition.jsx";
import "../styles/sound-derivation.css";

const tabs = ["① 从四个数字认识 RMS", "② 一步步推出 ÷20", "③ 代入 12 dB 并验算"];
const rmsSteps = [
  ["先看看：平均为 0，为什么还有声音？", "正负号表示波形向两边摆动。把正负样本直接相加，会抵消；平均值衡量的是中心偏移，不能代替振动强弱。"],
  ["第一步：平方，让正负摆动都贡献强度", "1²=1，(−1)²也等于1。在相同测量条件下，平均功率与振幅的平方成正比；所以这里选择平方来度量强度。"],
  ["第二步：平均，消除片段长短的影响", "把平方值相加，再除以样本数。同样的波形重复播放两遍，总平方和会加倍，但平均强度不该改变。"],
  ["第三步：开根号，回到幅度尺度", "均方值是幅度的平方；开根号把单位变回幅度。以后把波形放大2倍，RMS也放大2倍，方便直接计算增益。"]
];
const dbSteps = [
  { title: "先定义：SIR 用来比较哪两个量？", formula: "D = 10 × log₁₀(Pₛ / Pᵦ)", detail: "D 是以 dB 表示的主讲/背景功率比。Pₛ 是主讲的平均功率指标，Pᵦ 是缩放后背景的平均功率指标。这里的 10 来自功率比的分贝定义。" },
  { title: "把功率指标换成 RMS 的平方", formula: "Pₛ / Pᵦ = Rₛ² / Rᵦ² = (Rₛ / Rᵦ)²", detail: "Rₛ、Rᵦ分别是主讲与缩放后背景的 RMS。在相同测量条件下，功率与 RMS²成正比；公共比例系数在相除时抵消。" },
  { title: "代回第一行：只做一次替换", formula: "D = 10 × log₁₀[(Rₛ / Rᵦ)²]", detail: "把上一行的功率比替换进来，其他都不动。现在比较的是两个有效幅度，但平方还留在 log 里面。" },
  { title: "平方走到前面：20 = 10 × 2", formula: "log₁₀(q²) = 2 × log₁₀(q)\nD = 20 × log₁₀(Rₛ / Rᵦ)", detail: "q 是任意正数，在这里等于 Rₛ/Rᵦ。例如100=10²，所以log₁₀(100)=2。对数的幂法则把平方变成外面的乘2，于是10×2=20。" },
  { title: "把背景乘 a：RMS 也跟着乘 a", formula: "Rᵦ = a × Rᵥ\n当 Rₛ = Rᵥ 时：Rₛ / Rᵦ = 1/a\nD = 20 × log₁₀(1/a) = −20 × log₁₀(a)", detail: "Rᵥ 是背景源缩放前的 RMS；a 是正的背景增益。先把两源配到相同RMS，才可以约掉 Rₛ 和 Rᵥ。这个条件不能跳过。" },
  { title: "最后反解：从目标 dB 求背景增益", formula: "log₁₀(a) = −D/20\na = 10^(−D/20)", detail: "两边除以−20，再做log₁₀的逆运算。log₁₀(a)=u 的意思就是10ᵘ=a。因此负号、20和10的幂，每一项都有来源。" }
];

function SampleBars({ values, squared = false }) {
  const maximum = squared ? 4 : 2;
  return <div className={`derivation-bars ${squared ? "squared" : ""}`} aria-label={squared ? "样本平方值" : "有正有负的样本"}>
    {values.map((value, index) => <div className="derivation-bar-slot" key={index}>
      <span className="derivation-bar-value">{value}</span>
      <span className={`derivation-bar ${value < 0 ? "negative" : "positive"}`} style={{ height: `${Math.abs(value) / maximum * 44}%` }} />
      <small>n={index}</small>
    </div>)}
  </div>;
}

function Playback({ step, count, setStep, playing, setPlaying }) {
  return <div className="derivation-controls">
    <UiButton className="primary" onClick={() => { if (step === count - 1) setStep(0); setPlaying(!playing); }}>{playing ? "暂停推导" : "播放这段推导"}</UiButton>
    <UiButton disabled={step === 0} onClick={() => { setPlaying(false); setStep(step - 1); }}>← 上一步</UiButton>
    <UiButton disabled={step === count - 1} onClick={() => { setPlaying(false); setStep(step + 1); }}>下一步 →</UiButton>
    <UiButton onClick={() => { setPlaying(false); setStep(0); }}>重新推导</UiButton>
    <span>{step + 1} / {count}</span>
  </div>;
}

export function SoundDerivation({ sirDb, onSirChange, onPauseMain }) {
  const prefix = useId();
  const [tab, setTab] = useState(0);
  const [rmsStep, setRmsStep] = useState(0);
  const [dbStep, setDbStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [scale, setScale] = useState(1);
  const [sourceRms, setSourceRms] = useState(1);
  const rms = rmsCalculation(teachingSamples.map(value => value * scale));
  const mix = mixingCalculation(sirDb, 1, sourceRms);
  const step = tab === 0 ? rmsStep : dbStep;
  const count = tab === 0 ? rmsSteps.length : dbSteps.length;

  useEffect(() => {
    if (!playing || tab === 2) return undefined;
    const timer = setTimeout(() => {
      if (step >= count - 1) setPlaying(false);
      else if (tab === 0) setRmsStep(current => current + 1);
      else setDbStep(current => current + 1);
    }, 3200);
    return () => clearTimeout(timer);
  }, [playing, tab, step, count]);
  useEffect(() => {
    const pause = () => { if (document.hidden) setPlaying(false); };
    document.addEventListener("visibilitychange", pause);
    return () => document.removeEventListener("visibilitychange", pause);
  }, []);

  function chooseTab(index) {
    setPlaying(false);
    onPauseMain();
    setTab(index);
  }
  const definitions = tab === 0 ? [
    ["y[n]", "第n个波形样本；本例是4个教学数字"], ["n", "样本编号，从0数到3"], ["N=4", "这段波形共有4个样本"], ["Σ / √", "Σ表示全部相加；√表示平方根"]
  ] : [
    ["D", "目标SIR，单位dB；当前是" + sirDb + " dB"], ["Pₛ / Pᵦ", "主讲/缩放后背景的平均功率指标之比"], ["Rₛ / Rᵦ", "主讲/缩放后背景的RMS之比"], ["Rᵥ / a", "Rᵥ是背景源原始RMS；a是要算的正增益"], ["log₁₀ / 10ᵘ", "以10为底的对数/指数，二者互为逆运算"], ["s / v / b", "下标分别表示主讲/原背景/缩放后背景"]
  ];

  return <section className="sound-derivation" id="sound-derivation" aria-label="RMS与背景增益的图形实验">
    <SoundIntuition sirDb={sirDb} onSirChange={onSirChange} onPauseMain={onPauseMain} />
    <details className="derivation-full-proof" onToggle={event => { if (!event.currentTarget.open) setPlaying(false); }}>
    <summary>理解图形后，再展开完整字母推导、变量解释与双路线验算</summary>
    <div className="derivation-heading"><div><p className="section-label">第一课 · 从头推导，不跳步骤</p><h3 id={`${prefix}-heading`}>先有要解决的问题，才有这两个公式。</h3><p>先算四个数字，再看功率与幅度，最后反解背景增益。</p></div><span className="tag">约 5 分钟 · 可单步</span></div>
    <div className="derivation-tabs" role="tablist" aria-label="选择推导主题">{tabs.map((title, index) => <UiButton key={title} role="tab" id={`${prefix}-tab-${index}`} aria-selected={tab === index} aria-controls={`${prefix}-panel-${index}`} tabIndex={tab === index ? 0 : -1} onClick={() => chooseTab(index)} onKeyDown={event => {
      const next = event.key === "ArrowRight" ? (index + 1) % tabs.length : event.key === "ArrowLeft" ? (index + tabs.length - 1) % tabs.length : event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : null;
      if (next !== null) { event.preventDefault(); chooseTab(next); document.getElementById(`${prefix}-tab-${next}`)?.focus(); }
    }}>{title}</UiButton>)}</div>
    <div className="derivation-panel" role="tabpanel" id={`${prefix}-panel-${tab}`} aria-labelledby={`${prefix}-tab-${tab}`} tabIndex={0}>
      <dl className="derivation-symbols">{definitions.map(([symbol, meaning]) => <div key={symbol}><dt>{symbol}</dt><dd>{meaning}</dd></div>)}</dl>
      {tab === 0 && <>
        <div className="derivation-stage-title"><span>第 {rmsStep + 1} 幕</span><h4>{rmsSteps[rmsStep][0]}</h4></div>
        <p className="derivation-explain">{rmsSteps[rmsStep][1]}</p>
        <div className="derivation-example-switch"><span>教学样本：</span>{[1, 2].map(value => <UiButton key={value} aria-pressed={scale === value} onClick={() => { setScale(value); setPlaying(false); }}>{value === 1 ? "[1, −1, 1, −1]" : "每个样本放大2倍"}</UiButton>)}</div>
        <div className="derivation-rms-grid"><div><SampleBars values={rmsStep >= 1 ? rms.squares : rms.samples} squared={rmsStep >= 1} /><p className="derivation-caption">{rmsStep === 0 ? "蓝色有正有负：方向不同，不代表负数没有振动。" : "黄色都是平方值：向两边的振动都贡献强度。"}</p></div>
          <div className="derivation-calculations" aria-live="polite">
            <div className="wrong-average"><small>如果直接平均</small><code>({rms.samples.join(" + ")}) / 4 = {rms.mean}</code><p>平均值为0，但波形仍在摆动。</p></div>
            {rmsStep >= 1 && <div><small>① 平方</small><code>{rms.samples.map(value => `(${value})²`).join("，")} → {rms.squares.join("，")}</code><p>平方和 = {rms.squareSum}</p></div>}
            {rmsStep >= 2 && <div><small>② 平均</small><code>({rms.squares.join(" + ")}) / 4 = {rms.meanSquare}</code><p>这是均方值，用来比较平均强度。</p></div>}
            {rmsStep >= 3 && <div className="derivation-answer"><small>③ 开根号</small><code>√{rms.meanSquare} = {rms.rms}</code><p>RMS={rms.rms}：回到振幅尺度。</p></div>}
          </div>
        </div>
        <Playback step={rmsStep} count={4} setStep={setRmsStep} playing={playing} setPlaying={value => { onPauseMain(); setPlaying(value); }} />
        {rmsStep === 3 && <div className="derivation-conclusion"><strong>现在公式是把刚才三步写短一点：</strong><code>RMS(y) = √[(y[0]² + y[1]² + y[2]² + y[3]²) / N]</code><p>把“四项”推广为任意N项，就写成 √(Σ y[n]² / N)。RMS衡量有效幅度，不等于人耳感到的响度；平均绝对值也能衡量幅度，但与这里的平方功率关系不同。</p></div>}
        <div className="derivation-footnote">为什么选平方？相同电阻下，电压产生的平均功率是 mean(V²)/电阻；音频波形常用均方值作为能量强弱指标。数字样本没有校准时，这里比较的是相对指标，不把它当瓦特。</div>
      </>}
      {tab === 1 && <>
        <div className="derivation-stage-title"><span>第 {dbStep + 1} 幕</span><h4>{dbSteps[dbStep].title}</h4></div>
        <div className="derivation-big-formula" key={dbStep}>{dbSteps[dbStep].formula}</div>
        <p className="derivation-explain">{dbSteps[dbStep].detail}</p>
        <div className="derivation-log-table"><strong>先认识 log₁₀：它问“10的几次方等于这个数？”</strong><div>{[["10⁰ = 1", "log₁₀(1)=0", "功率比1 → 0 dB"], ["10¹ = 10", "log₁₀(10)=1", "功率比10 → 10 dB"], ["10² = 100", "log₁₀(100)=2", "功率比100 → 20 dB"]].map(([power, logarithm, decibel]) => <article key={power}><code>{power}</code><code>{logarithm}</code><small>{decibel}</small></article>)}</div></div>
        <Playback step={dbStep} count={6} setStep={setDbStep} playing={playing} setPlaying={value => { onPauseMain(); setPlaying(value); }} />
        <div className="derivation-conclusion"><strong>20的来源只有一条：</strong><p>功率比的dB定义带一个10；功率与有效幅度的平方成正比，再带来一个2。因此 20 = 10 × 2。</p><p>背景缩放前若未与主讲配平，完整公式是 a = (Rₛ/Rᵥ) × 10^(−D/20)。下一页可以切换两种情况验证。</p></div>
      </>}
      {tab === 2 && <>
        <div className="derivation-stage-title"><span>动手验算</span><h4>{sirDb} dB 到底要把背景乘多少？</h4></div>
        <label className="derivation-sir-control" htmlFor={`${prefix}-sir`}>目标SIR：<strong>{sirDb} dB</strong><input id={`${prefix}-sir`} type="range" min="0" max="24" step="3" value={sirDb} onChange={event => { onPauseMain(); onSirChange(Number(event.target.value)); }} /></label>
        <div className="derivation-presets">{[0, 6, 12, 18, 24].map(value => <UiButton key={value} aria-pressed={sirDb === value} onClick={() => { onPauseMain(); onSirChange(value); }}>{value} dB</UiButton>)}</div>
        <div className="derivation-example-switch"><span>缩放前两源RMS：</span><UiButton aria-pressed={sourceRms === 1} onClick={() => setSourceRms(1)}>主讲1 / 背景1</UiButton><UiButton aria-pressed={sourceRms === 2} onClick={() => setSourceRms(2)}>主讲1 / 背景2</UiButton></div>
        <div className="derivation-two-routes">
          <article><h4>路线A：从功率比算</h4><p>① 主讲/背景功率比：</p><code>10^({sirDb}/10) = {mix.powerRatio.toFixed(4)}</code><p>② 主讲功率指标为1，背景应为：</p><code>1 / {mix.powerRatio.toFixed(4)} ≈ {mix.backgroundPower.toFixed(6)}</code><p>③ 开根号得到背景RMS：</p><code>√{mix.backgroundPower.toFixed(6)} ≈ {mix.backgroundRms.toFixed(6)}</code></article>
          <article><h4>路线B：用刚推出的幅度公式</h4><p>① 幅度比：</p><code>10^(−{sirDb}/20) = {mix.amplitudeRatio.toFixed(6)}</code><p>② 按原背景RMS求增益a：</p><code>a = (1/{sourceRms}) × {mix.amplitudeRatio.toFixed(6)} ≈ {mix.gain.toFixed(6)}</code><p>③ 缩放后背景RMS：</p><code>{sourceRms} × {mix.gain.toFixed(6)} ≈ {mix.backgroundRms.toFixed(6)}</code></article>
        </div>
        <div className="derivation-ratio-bars"><div><span>主讲RMS：1</span><i style={{ width: "100%" }} /></div><div><span>背景RMS：{mix.backgroundRms.toFixed(3)}（主讲的 {(mix.amplitudeRatio * 100).toFixed(1)}%）</span><i style={{ width: `${mix.amplitudeRatio * 100}%` }} /></div><div><span>背景功率指标：主讲的 {(mix.amplitudeRatio ** 2 * 100).toFixed(2)}%</span><i style={{ width: `${mix.amplitudeRatio ** 2 * 100}%` }} /></div></div>
        <div className="derivation-conclusion"><strong>再反算一遍，确认目标真的达到：</strong><code>10 × log₁₀(1 / {mix.backgroundPower.toFixed(6)}) ≈ {mix.verifiedDb.toFixed(2)} dB</code><p>以上显示值做了四舍五入，计算用完整精度。{sirDb === 12 ? "12 dB时，幅度约25.1%，功率约6.31%；二者不能混为一谈。" : "拖动SIR：目标比值越大，允许留下的背景幅度越小。"}</p></div>
        <div className="derivation-footnote">Hush App演示先将主讲RMS定为0.04，再将背景RMS定为0.04×10^(−D/20)，最后混合。这里的a用来准备测试输入，不是调整模型内部的降噪强度。</div>
      </>}
      <div className="derivation-topic-navigation">{tab > 0 && <UiButton onClick={() => chooseTab(tab - 1)}>← 返回上一部分</UiButton>}{tab < 2 && <UiButton className="primary" onClick={() => chooseTab(tab + 1)}>{tab === 0 ? "我理解RMS了，接着看20从哪来 →" : "代入12 dB，亲手验算 →"}</UiButton>}</div>
    </div>
    </details>
  </section>;
}
