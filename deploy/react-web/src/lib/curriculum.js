import originalLessons from "./tutorial-lessons.js";

export const modules = [
  { id: "basics", title: "比例、平方与平均", outcome: "能亲手算出一段波形的有效振幅。", ids: ["ratio", "square", "mean", "rms"] },
  { id: "decibels", title: "指数、对数与分贝", outcome: "知道12 dB为什么对应约0.251的背景振幅。", ids: ["powers", "decibels"] },
  { id: "linear", title: "向量、点积与矩阵", outcome: "能算出神经网络中的一个输出格子。", ids: ["vector", "matmul"] },
  { id: "waves", title: "波动、采样与复数", outcome: "看懂频率、采样点和带方向的频谱。", ids: ["sine", "sound", "complex"] },
  { id: "spectrum", title: "从波形到频谱", outcome: "理解窗口、傅里叶变换及模型输入特征。", ids: ["window", "dft", "erb", "norm"] },
  { id: "model", title: "把数学拼成 Hush", outcome: "跟着卷积、GRU、Mask和DF走完增强链路。", ids: ["conv", "activation", "gru", "graphs", "mask", "deep", "istft"] },
  { id: "runtime", title: "让 runtime 算对、跑快", outcome: "看懂内存地址、GPU工作分配、状态和验收。", ids: ["tensor", "gpu", "state", "quality"] }
];

const additions = [
  {
    id: "ratio", title: "比例：旋钮到底乘了什么", kind: "foundation", frames: 3,
    question: "背景采样是0.8，把振幅旋钮调到25%，程序会写出什么数？",
    analogy: "像把一张照片的每个像素按同一比例缩小：先处理每个数字，再组成整段信号。",
    explanation: "25%就是25/100，也就是0.25。把每个采样乘0.25，正数和负数都会向0靠近；改变的是振幅，样本的数量与播放时长保持不变。",
    steps: ["先选中一个原始采样。", "把百分数换成小数，再做一次乘法。", "对每个时刻做同样的事，得到缩放后的波形。"],
    equations: ["b = a × v", "x = s + a × v"],
    symbols: [["v", "缩放前的一个背景采样"], ["a", "振幅系数，25%对应0.25"], ["b", "缩放后的背景采样"], ["s / x", "同一时刻的主讲采样 / 混合采样"]],
    controls: [{ id: "gain", label: "背景振幅保留比例", min: 0, max: 1, step: 0.05, value: 0.25, unit: "×" }],
    takeaway: "系数乘在每一个采样上；25%是乘0.25，不是减去0.25。",
    implementation: "App准备Mixed Input时缩放背景，再逐样本与主讲相加。这里只解释测试输入，不修改Hush模型内部参数。",
    sources: ["apps/hush_mic_ui.mm"],
    quiz: { question: "采样−0.8乘0.25后是多少？", choices: ["−0.2", "−1.05", "0.2"], answer: 0, reason: "−0.8×0.25=−0.2；保留方向，幅度缩小。" }
  },
  {
    id: "square", title: "平方与开根号：边长和面积", kind: "foundation", frames: 3,
    question: "一个正方形两条边都缩小一半，面积会剩一半吗？",
    analogy: "把振幅画成边长，把平方指标画成正方形面积：边长变2倍，面积就变4倍。",
    explanation: "平方就是一个数乘自己；开平方根是反过来找一个非负数，让它乘自己得到已知面积。同样测量条件下，功率与振幅平方成正比。",
    steps: ["先看蓝色边长。", "边长乘自己，得到黄色面积。", "给出面积，反过来找边长。"],
    equations: ["q = u²；u = √q（u≥0）"],
    symbols: [["u", "用来画图的非负边长"], ["q", "正方形面积，也用来演示平方指标"], ["²", "这个数乘自己"], ["√", "求非负平方根，从面积找边长"]],
    controls: [{ id: "side", label: "正方形边长", min: 0, max: 3, step: 0.25, value: 2, unit: "" }],
    takeaway: "边长减半，面积剩1/4；功率比例要开根号才能变回振幅比例。",
    implementation: "声音功率指标、复数模长和RMS都使用平方与开根号。未校准的音频样本平方不直接当作瓦特。",
    sources: ["src/hush_frontend.cpp"],
    quiz: { question: "面积是0.25，对应的非负边长是多少？", choices: ["0.5", "0.125", "0.25"], answer: 0, reason: "0.5×0.5=0.25，所以√0.25=0.5。" }
  },
  {
    id: "mean", title: "求和与平均：合起来，再分开", kind: "foundation", frames: 3,
    question: "四块面积是1、9、9、1，怎样得到每块的平均面积？",
    analogy: "把四杯水倒在一起，再平均分回四杯。总量没变，每杯的量变得一样。",
    explanation: "求和只管一共有多少；平均还要除以个数。相同的数据重复一遍，总和加倍，平均值不变。Σ只是把“这些数全部加起来”写短一点。",
    steps: ["逐个加入1、9、9、1。", "总面积是20。", "分成4份，每份是5。"],
    equations: ["m = (q[0] + q[1] + … + q[N−1]) / N"],
    symbols: [["q[n]", "编号为n的一块面积"], ["n", "从0开始的编号"], ["N", "共有几项，本例为4"], ["m", "平均值"], ["… / Σ", "中间各项继续相加 / 求和符号"]],
    controls: [{ id: "repeat", label: "重复这组数据", min: 1, max: 2, step: 1, value: 1, unit: "次" }],
    takeaway: "总量和平均强度是两个量；比较长短不同的录音时不能只比较总和。",
    implementation: "频带功率聚合和RMS计算都需要求和与平均。后面GRU还会用到“按不同权重求平均”。",
    sources: ["src/hush_frontend.cpp"],
    quiz: { question: "把1、9、9、1完整重复一遍，平均值变成多少？", choices: ["5", "10", "40"], answer: 0, reason: "总和从20变40，个数从4变8，40/8仍然是5。" }
  },
  {
    id: "rms", title: "RMS：给整段摆动一个强度数", kind: "intuition-rms", frames: 4,
    question: "正负采样平均为0，为什么这段声音仍然有强弱？",
    analogy: "每次摆动变成面积，求平均面积，再找到这个等效正方形的边长。",
    explanation: "RMS把前面三课接起来：逐样本平方、求平均、开平方根。这样正负不会抵消，片段长度不会混进强度，结果也回到振幅尺度。",
    steps: ["对每个采样平方。", "求平方值的平均。", "开根号得到有效振幅。"],
    equations: ["RMS(y) = √(Σ y[n]² / N)"],
    symbols: [["y", "整段采样序列"], ["y[n]", "编号n的采样值"], ["n / N", "从0开始的编号 / 总个数"], ["Σ", "把N个平方值加起来"], ["√", "求非负平方根"], ["RMS", "最终得到的有效振幅"]], controls: [],
    takeaway: "RMS是有效振幅，不能直接当作人耳感受到的响度，也不能只靠RMS下降判断降噪成功。",
    implementation: "App用RMS配平主讲和背景的原始强度，之后才能稳定设置测试SIR。",
    sources: ["apps/hush_mic_ui.mm"],
    quiz: { question: "[1,−1,1,−1]的普通平均值为0，RMS是多少？", choices: ["1", "0", "4"], answer: 0, reason: "平方后全是1，平均是1，√1=1。" }
  },
  {
    id: "powers", title: "指数与对数：同一件事的正反问法", kind: "foundation", frames: 3,
    question: "10的几次方等于100？这个问题怎样写成数学？",
    analogy: "指数像电梯的楼层：给楼层找数值；对数反过来，给数值找楼层。",
    explanation: "10²=100，意思是10乘10。log₁₀(100)=2，问的是10的几次方等于100。负指数表示倒数：10⁻²=1/100。小数指数填满两个整数楼层之间的位置。",
    steps: ["选一个指数u。", "算出10ᵘ对应的数值。", "用log₁₀问回原来的指数。"],
    equations: ["p = 10ᵘ ⇔ log₁₀(p) = u", "10⁻ᵘ = 1 / 10ᵘ"],
    symbols: [["u", "指数，也就是10的几次方"], ["p", "指数运算得到的正数"], ["log₁₀", "以10为底的对数，反问指数"], ["⇔", "两种写法表达同一关系"]],
    controls: [{ id: "exponent", label: "指数u", min: -2, max: 2, step: 0.2, value: 1.2, unit: "" }],
    takeaway: "指数和对数互相反问；log不是乘法，负指数也不是负数结果。",
    implementation: "频带特征通过log₁₀压缩数值跨度，分贝转换也需要指数与对数。",
    sources: ["src/hush_frontend.cpp"],
    quiz: { question: "log₁₀(0.01)等于多少？", choices: ["−2", "−100", "2"], answer: 0, reason: "10⁻²=1/100=0.01，因此log₁₀(0.01)=−2。" }
  },
  {
    id: "decibels", title: "分贝：从12 dB找到背景旋钮", kind: "intuition-db", frames: 4,
    question: "主讲要比背景强12 dB，背景每个采样应该乘多少？",
    analogy: "先把分贝尺读数翻译成功率倍数，再找背景面积，最后从面积找振幅边长。",
    explanation: "功率比每乘10，分贝刻度加10。主讲和原背景RMS相等时，12 dB对应约15.849倍功率比，背景应留约0.0631的功率，即约0.251的振幅。",
    steps: ["12 dB翻译成10^(12/10)倍。", "主讲功率取1，背景取其倒数。", "开根号回到振幅比例，最后再写公式。"],
    equations: ["a = 10^(−D/20)（两源原始RMS相等）", "D = 10 log₁₀(Pₛ/Pᵦ)"],
    symbols: [["a", "乘到原背景每个采样上的振幅系数"], ["D", "目标SIR，单位dB"], ["Pₛ / Pᵦ", "主讲 / 缩放后背景的相对功率指标"], ["10 / 20", "功率分贝定义的10 / 开根号后指数分母成为20"]],
    controls: [{ id: "sir", label: "目标SIR", min: 0, max: 24, step: 3, value: 12, unit: "dB" }],
    takeaway: "20来自分贝刻度的10与平方根的2；SIR越大，背景系数越小。",
    implementation: "a用于生成可控测试混音；原始两源RMS不相等时，还需乘主讲RMS/背景RMS作校正。",
    sources: ["apps/hush_mic_ui.mm"],
    quiz: { question: "两源RMS相等时，SIR从12增加到24 dB，背景系数怎样变？", choices: ["更小", "更大", "不变"], answer: 0, reason: "主讲相对背景要求更强，背景振幅必须更小。" }
  },
  {
    id: "vector", title: "向量与点积：一排数字怎样打分", kind: "foundation", frames: 3,
    question: "三个输入分别乘三个权重，再加起来，为什么能得到一个输出？",
    analogy: "三种食材各称一点，再按配方汇总。向量是一排数，点积是两排数对位相乘后求和。",
    explanation: "输入向量和权重向量长度相等。每对位置贡献一个乘积，正权重加分，负权重减分。神经网络里大量输出都由这个小动作组成。",
    steps: ["把输入与权重一一配对。", "每对数字相乘。", "把所有乘积相加得到一个数。"],
    equations: ["d = Σⱼ xⱼ wⱼ"],
    symbols: [["xⱼ", "输入向量的第j个数"], ["wⱼ", "同位置的权重"], ["j", "从0开始的位置编号"], ["Σⱼ", "遍历所有位置，把乘积相加"], ["d", "最终的点积，一个标量"]],
    controls: [{ id: "weight", label: "第二项权重", min: -2, max: 3, step: 0.5, value: -1, unit: "" }],
    takeaway: "向量是一排数；点积把两排等长的数压成一个数；矩阵乘法会重复很多次点积。",
    implementation: "线性层、卷积的局部输出、频带聚合都可以从对位乘加理解。",
    sources: ["src/kernels/cpu/matmul.cpp"],
    quiz: { question: "[1,2,3]与[2,0,−1]的点积是多少？", choices: ["−1", "1", "[2,0,−3]"], answer: 0, reason: "1×2+2×0+3×(−1)=−1；点积最终还要把乘积加起来。" }
  },
  {
    id: "sine", title: "正弦与余弦：转一圈，画一段波", kind: "foundation", frames: 13,
    question: "一支匀速旋转的指针，为什么能画出起伏的波形？",
    analogy: "盯着转动指针的尖端，把它的高度沿时间记录下来，就得到正弦波。",
    explanation: "正弦是单位圆上的纵向投影，余弦是横向投影。振幅决定摆多高，频率决定每秒转几圈，相位决定从哪里出发。一圈为360°，也写成2π弧度。",
    steps: ["看单位圆上指针的角度。", "把指针尖端投影到纵轴。", "把连续的投影沿横轴展开成波。"],
    equations: ["y = A sin(θ)；θ = 2π f t + φ"],
    symbols: [["y", "当前时刻的波形数值"], ["A", "振幅，最大摆动高度"], ["θ", "当前角度，单位弧度"], ["f", "频率，每秒几圈，单位Hz"], ["t", "时间，单位秒"], ["φ", "初相位，起始角度"], ["π", "圆周率，一圈为2π弧度"], ["sin / cos", "单位圆的纵向 / 横向投影"]],
    controls: [{ id: "angle", label: "起始角度（每步再转30°）", min: 0, max: 360, step: 15, value: 90, unit: "°" }],
    takeaway: "同一个波可以用时间上的起伏表示，也可以用旋转指针表示。下一课再学习怎样记录采样点。",
    implementation: "傅里叶变换用不同频率的正弦、余弦作为匹配参考；真实人声由许多变化的频率成分组成。",
    sources: ["src/hush_frontend.cpp"],
    quiz: { question: "指针从圆的最右端转到最高点，sin(90°)是多少？", choices: ["1", "0", "−1"], answer: 0, reason: "最高点的纵向投影是1；横向投影cos(90°)则是0。" }
  }
];

const changes = {
  sound: {
    title: "采样：每秒记录16000次", kind: "foundation", frames: 4,
    question: "16 kHz的录音里，160个采样为什么表示10 ms？",
    analogy: "给连续摆动拍连拍照片：每秒拍多少张，就是采样率。",
    explanation: "16 kHz表示每秒记录16000次。把采样个数除以每秒采样次数，就得到秒数。可靠记录一个周期需要足够密的采样；采样率的一半是Nyquist边界。",
    equations: ["t = H / fₛ", "f_N = fₛ / 2"],
    symbols: [["t", "这批采样覆盖的时间，单位秒"], ["H", "采样个数，本模型每次推进160点"], ["fₛ", "采样率，单位Hz，本模型为16000"], ["f_N", "Nyquist频率边界，可靠处理还需要抗混叠滤波"]],
    controls: [{ id: "hop", label: "这批采样点数", min: 80, max: 320, step: 80, value: 160, unit: "点" }],
    steps: ["每秒均匀记录16000次。", "取出H个点，用H/16000算时间。", "真实设备重采样还要处理抗混叠，不能随意丢点。"],
    takeaway: "160/16000=0.01秒=10毫秒；采样率决定同一串数据的播放速度。"
  },
  gpu: { explanation: "教学grid是2×2个逻辑线程，对应矩阵课中的2×2输出。真实Metal matmul的线程坐标是(col,row)。逻辑线程由硬件分批调度，线程数量不等于物理核心数量。" }
};

const catalog = new Map([...originalLessons, ...additions].map(lesson => [lesson.id, lesson]));
const prerequisites = {
  ratio: [], square: ["ratio"], mean: ["ratio"], rms: ["square", "mean"],
  powers: ["ratio"], decibels: ["rms", "powers"], vector: ["ratio", "mean"], matmul: ["vector"],
  sine: ["square", "ratio"], sound: ["sine"], complex: ["square", "sine"],
  window: ["sound", "ratio"], dft: ["vector", "complex", "window"], erb: ["dft", "powers"], norm: ["mean", "erb"],
  conv: ["matmul"], activation: ["powers"], gru: ["matmul", "activation", "norm"], graphs: ["conv", "gru"],
  mask: ["erb", "complex", "graphs"], deep: ["complex", "gru", "mask"], istft: ["dft", "window", "deep"],
  tensor: ["matmul"], gpu: ["matmul", "tensor"], state: ["tensor", "gru"], quality: ["decibels", "state"]
};
export const lessons = modules.flatMap((module, moduleIndex) => module.ids.map(id => ({
  ...catalog.get(id), ...changes[id],
  moduleId: module.id,
  category: `${String(moduleIndex + 1).padStart(2, "0")} / ${module.title}`,
  prerequisites: prerequisites[id]
})));

export const lessonById = Object.fromEntries(lessons.map(lesson => [lesson.id, lesson]));

export default lessons;
