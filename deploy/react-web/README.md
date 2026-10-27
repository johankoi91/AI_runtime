# Hush React Web

这是现有 `deploy/web` 的独立 React 重实现。旧页面和部署入口保持不变；本目录只负责生成可单独预览或部署的新版本。

技术栈：React 18、Vite、Mantine 7。Mantine 提供主题、可访问按钮和控件基础，页面外观继续使用当前 Trace Inspector 与动画课程的深色视觉规范。Canvas 教学动画由 `CanvasStage` 管理，Trace 回放由 React state/hooks 管理。

第 1 课顶部有独立的公式推导实验台：先对四个有正有负的样本执行平方、平均、开根号，再从功率比的 dB 定义逐步推出 `20=10×2`，最后用 12 dB 沿功率与幅度两条路线计算并反向验算。可切换两源 RMS 相等/不等的输入，查看简化公式的适用条件。推导播放与原有 Canvas 播放分开控制。

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

`src/lib/` 中的 Trace 解析、课程数据和教学数学实现是当前页面的只读副本；修改新站点时应只改本目录，保留旧站点作为可回退基线。

## 测试

```sh
npm ci
npm test
npm run build
```

测试包含固定 Trace fixture、reducer 状态、18 课课程数据、数学函数，以及 972 个桌面/移动 Canvas 帧与旧版绘制结果的逐帧对照。不会访问线上 Influx，也不会修改 `deploy/web`。

第 1 课的数值回归还检查：普通均值为零但 RMS 非零、幅度与均方值的平方关系、重复样本时 RMS 不变、12 dB 的幅度/功率比例，以及原始两源强度不等时的增益修正。

需要查看旧页面与新页面的并排效果时，可以启动只读基线和新站点：

```sh
node tests/reference-server.js
npm run dev -- --port 5175
```

旧页面基线在 `http://127.0.0.1:5174/`，React 页面在 `http://127.0.0.1:5175/`。`HUSH_INFLUX_PROXY_TARGET` 可指定本地或远端 Influx 代理地址。
