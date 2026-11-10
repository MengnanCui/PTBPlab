# PTBPlab 总计划

仓库里有两条开发线，各有自己的路线图。**目前的重点是业务线。**

| | 业务线（PTBP） | 引擎线（pydftb） |
|---|---|---|
| 回答的问题 | 用什么参数、什么数据与损失、什么优化策略，才能得到达到 DFT 精度的 DFTB 参数集 | DFTB 计算本身算得对不对、快不快、功能够不够 |
| 范围 | 参数模型（每个元素的 r0_w、r0_d、σ_rep，类型，随环境变化的参数，排斥势，Δ-ML）；损失与数据集；多参数的优化策略（BO、并行 BO、PSO、梯度法、σ_rep 内层搜索、策略比较）；运行方案与工具（并行、`ptbp plan`、TUI、GUI） | 与 DFTB+ 逐项对照的功能（M2、M6）；求解器与大体系（M7）；PyTorch 后端、自动求导与设备（M4a、M4b、M4d）；双引擎与打包（P1、P2） |
| 代码 | `cli/`、`utils/`、`tools/`、`gui/`、`parameterize.py`、`examples/` | `pydftb/` |
| 路线图 | [`docs/ROADMAP.md`](docs/ROADMAP.md) | [`pydftb/docs/ROADMAP.md`](pydftb/docs/ROADMAP.md) |
| 当前状态 | A0（精度计划）的代码已备好，计算未开始 | 功能基本齐全，以维护为主 |

## 两条线的分工

- 业务线向引擎线提需求：业务需要的引擎功能（例如能带损失要的本征值、随环境变化的参数、梯度），先在业务路线图里写明用途，再在引擎路线图里立项。
- 引擎线的改动不能改变业务线的结果：PTBP 默认路径的运行目录与损失值要逐字节不变（golden 与 G1 门槛）。
- 一个阶段跨两条线时（例如 M5：引擎的平滑切换 M5a + PTBP 的按类型拟合 M5b–M5d），引擎部分记在引擎路线图，业务部分记在业务路线图。

## 业务线的下一步

见 [`docs/ROADMAP.md`](docs/ROADMAP.md) 第二节，大致顺序：

1. A1.0 的便宜检验 K1–K7；
2. A1 的策略比较（`ptbp scan`，同一批 SK 点上比较 S1–S4）；
3. 排斥势：按原子对的标度与短程墙接到 `optimize`，σ_rep 内层搜索的边界问题；
4. A2：随环境变化的参数；
5. 最后 A3（排斥势）与 A4（Δ-ML）。

需要你决定的事项：

- DFT 数据 D2–D5 是否提交（粗估 450–750 元）；
- σ_rep 搜索的修正会改变默认结果，是否修正；
- M8 第三轮主动学习的 DFT 帧是否提交。

## 引擎线的待办（不急）

- 分治的其余 N² 数组与不建 γ 的 SCC 电势（M7d3 报告第八节）；
- 旧 SCC 路径最后一轮不收敛时不再混合（与 DFTB+ 一致）；
- 向 DFTB+ 报告对照中发现的问题（等你同意）。

## 约定

- 每个阶段开始前写设计文档，结束后写完成报告，都在 `pydftb/docs/design/` 与 `pydftb/docs/reports/`（两条线共用一个文档站）。
- 仓库里不写个人机器上的路径和同步盘信息；数据与大文件的位置用环境变量给出（见 `examples/08_accuracy/README.md`）。
- hotcent 的源码在 `third_party/hotcent`，离线安装：`pip install --no-build-isolation ./third_party/hotcent`。
