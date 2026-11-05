import { useEffect } from "react";

const format = value => Number(value.toFixed(5)).toString();

function Wave({ gain, color, center = 70 }) {
  const path = Array.from({ length: 101 }, (_, index) => `${index ? "L" : "M"}${30 + index * 5.2},${center - gain * Math.sin(index / 100 * Math.PI * 6) * 38}`).join(" ");
  return <path d={path} fill="none" stroke={color} strokeWidth="2.5" />;
}

function Calculation({ title, children, active = false }) {
  return <div className={`foundation-calculation${active ? " active" : ""}`}><span>{title}</span><strong>{children}</strong></div>;
}

export function FoundationStage({ lesson, frame, parameters, onResult }) {
  let visual;
  let result;
  if (lesson.id === "ratio") {
    const gain = Number(parameters.gain);
    result = `0.8 × ${format(gain)} = ${format(0.8 * gain)}；每个背景采样都乘同一系数。`;
    visual = <><svg viewBox="0 0 600 205" role="img" aria-label={`原波形与乘${gain}后的波形，纵轴比例相同`}><text x="30" y="20">原始背景</text><path className="foundation-axis" d="M30 70 H550 M30 165 H550" /><Wave gain={0.8} color="var(--blue)" /><text x="30" y="115">缩放后背景 × {format(gain)}</text><Wave gain={0.8 * gain} center={165} color="var(--green)" /></svg><div className="foundation-calculations"><Calculation title="① 原采样" active={frame === 0}>0.8</Calculation><Calculation title="② 振幅比例" active={frame === 1}>{format(gain * 100)}% = {format(gain)}</Calculation><Calculation title="③ 相乘后" active={frame === 2}>{format(0.8 * gain)}</Calculation></div></>;
  } else if (lesson.id === "square") {
    const side = Number(parameters.side);
    const pixels = side * 45;
    result = `边长${format(side)} × 边长${format(side)} = 面积${format(side ** 2)}；√${format(side ** 2)} = ${format(side)}。`;
    visual = <><svg viewBox="0 0 600 235" role="img" aria-label={`边长${side}，面积${side ** 2}`}><rect className="foundation-square-frame" x="55" y="30" width="135" height="135" /><rect className="foundation-square" x="55" y={165 - pixels} width={pixels} height={pixels} /><path className="foundation-blue-edge" d={`M55 175 h${pixels}`} /><text x="55" y="205">边长 {format(side)}</text><text x="280" y="85">边长 × 边长 = 面积</text><text className="foundation-emphasis" x="280" y="125">{format(side)} × {format(side)} = {format(side ** 2)}</text><text x="280" y="170">反过来：√{format(side ** 2)} = {format(side)}</text></svg><div className="foundation-calculations"><Calculation title="① 先给边长" active={frame === 0}>{format(side)}</Calculation><Calculation title="② 平方求面积" active={frame === 1}>{format(side ** 2)}</Calculation><Calculation title="③ 开根号找边长" active={frame === 2}>{format(side)}</Calculation></div></>;
  } else if (lesson.id === "mean") {
    const repeat = Number(parameters.repeat);
    const values = Array.from({ length: repeat }, () => [1, 9, 9, 1]).flat();
    const shown = frame === 2 ? values.map(() => 5) : values;
    const cellWidth = 480 / shown.length;
    result = `总和${20 * repeat} ÷ 个数${4 * repeat} = 平均值5；重复数据不会改变平均值。`;
    visual = <><svg viewBox="0 0 600 245" role="img" aria-label={frame === 2 ? "平均分后每份5" : `原值${values.join("、")}`}><text x="50" y="25">{frame === 2 ? "同样的总量，平均分回每份" : "每根柱子是一块面积；点“一步计算”看平均分"}</text><path className="foundation-axis" d="M50 185 H550" />{shown.map((value, index) => <g key={index}><rect className="foundation-square" x={55 + index * cellWidth} y={185 - value * 14} width={cellWidth - 10} height={value * 14} /><text x={55 + index * cellWidth + (cellWidth - 10) / 2} y="211" textAnchor="middle">{value}</text></g>)}</svg><div className="foundation-calculations"><Calculation title="① 总和" active={frame === 0}>{20 * repeat}</Calculation><Calculation title="② 个数" active={frame === 1}>{4 * repeat}</Calculation><Calculation title="③ 平均" active={frame === 2}>{20 * repeat} / {4 * repeat} = 5</Calculation></div></>;
  } else if (lesson.id === "powers") {
    const exponent = Number(parameters.exponent);
    const value = 10 ** exponent;
    result = `10^${format(exponent)} ≈ ${format(value)}；log₁₀(${format(value)}) ≈ ${format(exponent)}。`;
    visual = <><div className="foundation-power-pair"><Calculation title="给指数，找数值" active={frame < 2}>10^{format(exponent)} ≈ {format(value)}</Calculation><span>⇄</span><Calculation title="给数值，找指数" active={frame === 2}>log₁₀({format(value)}) ≈ {format(exponent)}</Calculation></div><svg viewBox="0 0 600 165" role="img" aria-label={`指数尺上的位置${exponent}，这是对数刻度`}><path className="foundation-axis" d="M50 70 H550" />{[-2, -1, 0, 1, 2].map(position => <g key={position}><path className="foundation-axis" d={`M${300 + position * 120} 60 v20`} /><text x={300 + position * 120} y="105" textAnchor="middle">10^{position}</text><text x={300 + position * 120} y="129" textAnchor="middle">{10 ** position}</text></g>)}<circle cx={300 + exponent * 120} cy="70" r="8" fill="var(--green)" /><text x="300" y="155" textAnchor="middle">指数等距：往右一格，数值乘10</text></svg></>;
  } else if (lesson.id === "vector") {
    const inputs = [1, 2, 3];
    const weights = [2, Number(parameters.weight), 0.5];
    const products = inputs.map((value, index) => value * weights[index]);
    const subtotal = products.slice(0, frame + 1).reduce((sum, value) => sum + value, 0);
    result = `已累加${frame + 1}项：${products.slice(0, frame + 1).map(format).join(" + ")} = ${format(subtotal)}。`;
    visual = <><div className="foundation-vector">{inputs.map((value, index) => <div className={index === frame ? "active" : ""} key={index}><span>位置 {index}</span><b>{value}</b><i>×</i><b>{format(weights[index])}</b><i>↓</i><strong>{format(products[index])}</strong></div>)}</div><div className="foundation-line"><span>把亮起位置及它之前的乘积加起来</span><code>{products.slice(0, frame + 1).map(format).join(" + ")} = {format(subtotal)}</code><p>每一对相乘只是中间结果，点积还要把它们汇总。</p></div></>;
  } else if (lesson.id === "sine") {
    const totalDegrees = Number(parameters.angle) + frame * 30;
    const degrees = totalDegrees === 360 ? 360 : totalDegrees % 360;
    const angle = degrees * Math.PI / 180;
    const horizontal = 130 + Math.cos(angle) * 70;
    const vertical = 115 - Math.sin(angle) * 70;
    const path = Array.from({ length: 121 }, (_, index) => `${index ? "L" : "M"}${270 + index * 2.4},${115 - Math.sin(index / 120 * Math.PI * 2) * 70}`).join(" ");
    result = `角度${degrees}°：纵向投影sin≈${format(Math.sin(angle))}；横向投影cos≈${format(Math.cos(angle))}。`;
    visual = <><svg viewBox="0 0 600 255" role="img" aria-label={`圆上角度${degrees}度与正弦波对应位置`}><circle className="foundation-circle" cx="130" cy="115" r="70" /><path className="foundation-axis" d="M40 115 H220 M130 25 V205 M270 115 H558" /><path className="foundation-arrow" d={`M130 115 L${horizontal} ${vertical}`} /><circle cx={horizontal} cy={vertical} r="6" fill="var(--amber)" /><path className="foundation-projection" d={`M${horizontal} ${vertical} H${270 + degrees / 360 * 288}`} /><path className="foundation-wave" d={path} /><circle cx={270 + degrees / 360 * 288} cy={vertical} r="6" fill="var(--green)" /><text x="130" y="233" textAnchor="middle">转一圈 = 360° = 2π弧度</text><text x="414" y="233" textAnchor="middle">把纵向高度沿时间展开</text></svg><div className="foundation-calculations"><Calculation title="角度">{degrees}°</Calculation><Calculation title="sin：纵向">{format(Math.sin(angle))}</Calculation><Calculation title="cos：横向">{format(Math.cos(angle))}</Calculation></div></>;
  } else {
    const hop = Number(parameters.hop);
    const milliseconds = hop / 16000 * 1000;
    result = `${hop}点 / 16000点每秒 = ${format(hop / 16000)}秒 = ${milliseconds}毫秒。`;
    visual = <><svg viewBox="0 0 600 215" role="img" aria-label={`示意采样点，真实本次有${hop}点，时长${milliseconds}毫秒`}><text x="30" y="25">连续波形 → 均匀记录采样（下图用少量点示意）</text><path className="foundation-axis" d="M30 110 H550" /><Wave gain={1} center={110} color="var(--blue)" />{Array.from({ length: 25 }, (_, index) => <circle key={index} cx={30 + index * 520 / 24} cy={110 - Math.sin(index / 24 * Math.PI * 6) * 38} r="4" fill="var(--amber)" />)}<text x="30" y="195">一秒16000点；本次取{hop}点</text></svg><div className="foundation-calculations"><Calculation title="① 点数" active={frame === 0}>{hop}</Calculation><Calculation title="② 每秒记录次数" active={frame === 1}>16000</Calculation><Calculation title="③ 秒 → 毫秒" active={frame >= 2}>{format(hop / 16000)}s = {milliseconds}ms</Calculation></div></>;
  }
  useEffect(() => { onResult(result); }, [onResult, result]);
  return <div className="foundation-stage" aria-label={`${lesson.title}的交互实验`}>{visual}</div>;
}
