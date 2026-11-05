# Hush React Web

这是现有 `deploy/web` 的独立 React 重实现。旧页面代码保留；本目录生成可单独预览或部署的新版本，目前线上站点使用这里的构建产物。

技术栈：React 18、Vite、Mantine 7。Mantine 提供主题、可访问按钮和控件基础，页面外观继续使用当前 Trace Inspector 与动画课程的深色视觉规范。Canvas 教学动画由 `CanvasStage` 管理，Trace 回放由 React state/hooks 管理。

教程按数学先修关系重组为 7 个阶段、26 节课，默认从比例开始。原有18个主题的 URL hash 仍然有效，例如 `#sound` 现在直接选择采样课。

| 阶段 | 课程 |
| --- | --- |
| 1. 比例、平方与平均 | 比例；平方与开根号；求和与平均；RMS |
| 2. 指数、对数与分贝 | 指数与对数；分贝与12 dB混音 |
| 3. 向量、点积与矩阵 | 向量与点积；矩阵乘法 |
| 4. 波动、采样与复数 | 正弦与余弦；采样；复数 |
| 5. 从波形到频谱 | 窗口；傅里叶变换；ERB；归一化 |
| 6. 把数学拼成Hush | 卷积；激活；GRU；依赖图；Mask；DF；合成 |
| 7. runtime | Tensor地址；GPU；状态与内存复用；误差与性能验收 |

每课分成“问题 → 实验 → 公式 → 工程应用与小题”四步。公式再按“整句阅读 → 每个符号 → 数值示例 → 公式目的”逐段展开，完整模型公式放在折叠区。基础课提供可调振幅、面积分配、指数尺、点积与旋转投影实验；RMS与分贝使用独立图形推导，后续课程复用经过验证的Canvas计算。

课程、先修关系及核心公式分别在 `src/lib/curriculum.js`、`src/lib/formula-walkthroughs.js`。完成进度使用浏览器 `hush-math-completed-v2`，答对小题并点击“记为完成”才计入，访问历史不等于完成。所有课程可自由跳转，离开实验或页面隐藏时暂停播放。均方值作为相对功率指标，未校准的采样不当作瓦特；背景系数用于准备测试混音，不代表模型内部的增强强度。

## 本地运行

```sh
npm install
npm run dev
```

打开 Vite 输出的地址：

- `/`：真实 Hush Trace 回放
- `/tutorial.html`：数学与 GPU 动画教程

## 构建

```sh
npm run build
npm run preview
```

构建结果位于 `dist/`，不会覆盖 `deploy/web`，也不会修改现有 Docker/Nginx 配置。部署时可以把 `dist/` 作为新的静态站点根目录，并按现有服务配置 `/influx/` 反向代理。

`nginx.conf` 和 `docker-compose.production.yml` 是新站点的独立部署模板：它们只挂载 `dist/`，并保留 `/influx/` 代理。

`src/lib/` 中保留原版Trace解析、18课数据与教学数学实现作为兼容基线；新课程在 `curriculum.js` 中扩展与重排。页面实现修改限定在本目录，保留 `deploy/web` 作为可回退基线。

## 测试

```sh
npm ci
npm test
npm run build
```

测试包含固定 Trace fixture、26课课程顺序与先修关系、旧链接兼容、逐课公式资料、小题与完成进度、reducer状态、数学函数，以及972个桌面/移动Canvas帧与旧版绘制结果的逐帧对照。不会访问线上Influx，也不会修改 `deploy/web`。

RMS与分贝的数值回归检查：普通均值为零但RMS非零、幅度与均方值的平方关系、不均匀样本平均面积、重复样本时RMS不变、12 dB的幅度/功率比例，以及原始两源强度不等时的增益修正。

人工自测流程：从无hash的 `/tutorial.html` 确认进入比例课；走完四步并尝试错误/正确答案，正确完成后应进入平方课，刷新后仍显示1/26。打开 `#rms` 验证平方→平均→找边长；打开 `#decibels` 验证12 dB得到0.251189的振幅和约6.31%的功率。逐阶段抽查矩阵、DF与GPU实验，切换学习步骤确认动画暂停。用390px宽度检查课程目录、滑块和公式，确认无横向溢出。最后检查 `/` 的Trace入口与Influx代理。

需要查看旧页面与新页面的并排效果时，可以启动只读基线和新站点：

```sh
node tests/reference-server.js
npm run dev -- --port 5175
```

旧页面基线在 `http://127.0.0.1:5174/`，React 页面在 `http://127.0.0.1:5175/`。`HUSH_INFLUX_PROXY_TARGET` 可指定本地或远端 Influx 代理地址。
