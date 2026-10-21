# pydftb 路线图与进度

规则：每个阶段（或阶段内的一块工作）开始前写一份**设计文档**（`design/`），结束后写一份**完成报告**（`reports/`）；新功能都要和 DFTB+ 对照，参考结果固化在 `pydftb/tests/data/ref/`。工作量单位为人周，均为估计值。

## 阶段状态

| 阶段 | 内容 | 状态 | 文档 |
|---|---|---|---|
| 原型 | SCC-DFTB 原型、加速实验、调研报告 | ✅ 完成 | `../RESEARCH.md` |
| M0 | 底座整理：ops 分发层、`System`/`ParamSet`/`SCC` 接口、四段式计算、冻结的 DFTB+ 参考数据、文档骨架 | ✅ 完成 | `design/M0-foundation.html`，`reports/M0-foundation.html` |
| M1 | 接入 PTBP：`--engine pydftb`、`--smearing` 开关、MP/Gaussian 展宽、Broyden 混合、排斥势开关、PTBP 损失函数等价性 | ✅ 完成（损失函数相对差 < 1e-7，DFTB 阶段快 1.8 倍） | `design/M1-ptbp-integration.html`，`reports/M1-ptbp-integration.html` |
| M2 | DFTB+ 功能补齐·第一波：应力、DFTB3、自旋、色散、能带/DOS、结构优化/MD、f 轨道 | ✅ 完成：M2a 应力 ✅（与 DFTB+ 差 ≤ 3e-9 Ha/Bohr³，顺带修正 SKF 表格末端）；M2b DFTB3 ✅（顺带修正 U 接近时 γ 的数值崩溃）；M2c 共线自旋 ✅（SCC 轮数与 DFTB+ 全部相同）；M2d 色散 ✅（UFF、SK、D3、SimpleD3、D4，色散部分与 DFTB+ 差 ≤ 1e-10；发现 DFTB+ 原生 D4 的三处问题；仓库改为 MIT 许可）；M2e 能带/DOS/PDOS ✅（周期体系本征值与 DFTB+ 差 ≤ 1.7e-11 Ha；顺带统一玻尔兹曼常数和短程 γ 截断；发现 DFTB+ middleGap 的舍入问题）；M2f 结构优化与 MD ✅（与 DFTB+ 自带驱动对照：优化终点能量差 ≤ 5e-11 Ha，MD 40 步位置差 ≤ 4.4e-8 Å、SCC 轮数逐步一致；力改用输出电荷，SCC 容差 1e-5 时力与 DFTB+ 一致到 ~1e-11）；M2g f 轨道 ✅（读 DFTB+ 的扩展 SKF 格式，两种格式可混用；Gd/O 测试参数下 8 个对照算例的能量差 ≤ 4.1e-11 Ha、SCC 轮数相同，用 DFTB+ 的差分时力差 ≤ 8.2e-13 Ha/Bohr；顺带去掉旋转中 einsum 的路径搜索，单点计算快 8–45%） | `design/M2a-stress.html`、`reports/M2a-stress.html`；`design/M2b-dftb3.html`、`reports/M2b-dftb3.html`；`design/M2c-spin.html`、`reports/M2c-spin.html`；`design/M2d-dispersion.html`、`reports/M2d-dispersion.html`；`design/M2e-bands.html`、`reports/M2e-bands.html`；`design/M2f-opt-md.html`、`reports/M2f-opt-md.html`；`design/M2g-f-orbitals.html`、`reports/M2g-f-orbitals.html` |
| M3a | 并行分发与 DFTB 阶段：CPU 预算与方案、fork 进程池（最大优先）、DFTB 阶段按结构并行、pydftb 的 k 点线程、`--ncores`（默认用满核心） | ✅ 完成（Cu 数据集 4 核：DFTB 阶段 pydftb 3.4 倍、DFTB+ 4.1 倍，整个 BO 步 1.2–1.5 倍；与串行的输出逐字节相同、损失值逐位相同；pydftb 的 k 点线程逐位相同；DFTB+ 默认的 OpenMP 线程在这些小体系上反而更慢） | `design/M3a-dispatch.html`，`reports/M3a-dispatch.html` |
| M3b | BO 步的其余部分：hotcent 按行分片并回放（逐字节相同）、physrep 向量化（逐位相同） | ✅ 完成（4 核一个 BO 步：pydftb 122 → 27 s、DFTB+ 132 → 40 s，对 M3 之前快 5.4 倍；运行目录、`log.out`、`par.out` 逐字节相同；hotcent 原来把单元素的表算了三遍；σ_rep 搜索的 74 次评估中 41 次白算，未改） | `design/M3b-hotcent-physrep.html`，`reports/M3b-hotcent-physrep.html` |
| M3c | 计算策略推荐器 v1（CPU）：成本模型、内存估计、`ptbp plan`、`plan.json` | ✅ 完成（成本模型按轨道数、k 点数和截断内的原子对估计 pydftb、DFTB+ 单点的耗时和内存：耗时中位误差 12%/7%，排序秩相关 0.99，pydftb 内存中位误差 2%；Cu 数据集一个 BO 步的预计耗时与实测相差 2–4%；方案只调不改变数值的设置，运行目录与 M3b 逐字节相同；发现大晶胞的耗时和内存主要在库仑部分：最小的 Hubbard U 决定截断，Cu 的 4p 为 96 Bohr，64 个原子的晶胞 pydftb 要 2.2 GB，DFTB+ 0.3 GB；DFTB+ 的 OpenMP 线程在 500 个轨道以下反而更慢） | `design/M3c-recommender.html`，`reports/M3c-recommender.html` |
| M3d | DFTB+ 配置顾问：HSD 读取器、规则目录、`python -m pydftb.advisor`、PTBP 钩子 | 计划 | — |
| M4 | PyTorch 后端：GPU、自动求导、批量 | 计划 | — |
| M5 | 同一元素多套参数（离散类型 → 平滑切换），固态电池示范 | 计划（离散类型的基础已在 M0 完成） | — |
| M6 | DFTB+ 功能补齐·第二波（按需）：自旋轨道耦合、DFTB+U、长程校正、溶剂、外场、TD-DFTB、XL-BOMD | 计划 | — |
| M7 | 大体系：GPU 混合精度 SP2、Chebyshev、分治多机 | 计划（CPU 原型见 `RESEARCH.md`） | — |
| M8 | AI 层：随环境连续变化的参数、ML 排斥势、主动学习 | 计划 | — |

## 与 DFTB+ 的功能对照清单

✅ 已实现并与 DFTB+ 对照；⬜ 未实现；— 不打算自己实现（调用现成库，或按需再说）

| 类别 | 功能 | 状态 |
|---|---|---|
| 哈密顿量 | 非 SCC / SCC (DFTB2) | ✅ |
| | 力：精确导数（默认），或模拟 DFTB+ 的有限差分（`fd_delta`） | ✅ M1 |
| | 轨道分辨 SCC | ✅ |
| | s/p/d 轨道 | ✅ |
| | SKF 表格末端和 1 Bohr 衰减尾巴（DFTB+ 只用表头行数的前 n−1 行） | ✅ M2a 修正 |
| | f 轨道（DFTB+ 的扩展 SKF 格式，可与简单格式混用；测试参数由 `validation/make_f_set.py` 用 hotcent 生成） | ✅ M2g |
| | DFTB3（完整三阶 / 只有原子自身部分，按壳层的 Hubbard 导数）、H–X 阻尼 | ✅ M2b |
| | U 几乎相等时 γ、Γ 的数值稳定计算（DFTB+ 在 3e-6 < \|ΔU\| ≲ 1e-3 时失效） | ✅ M2b（pydftb 独有） |
| | xTB (GFN1/GFN2) | — 调用 `tblite` |
| 周期体系 | k 点、时间反演约化、Ewald | ✅ |
| | 短程 γ 在 S < 1e-10 处截断（与 DFTB+ 相同） | ✅ M2e 统一 |
| | 应力（含 Ewald、排斥势、SCC 项）、ASE 晶格优化 | ✅ M2a |
| 电子结构分析 | 冻结电荷求任意 k 点本征态（DFTB+ `ReadInitialCharges` + `Klines`）、能带路径（Klines 规则或 ASE BandPath） | ✅ M2e |
| | 投影态（DFTB+ `ProjectStates`：按原子、壳层、轨道）、DOS/PDOS 曲线（dptools `dp_dos` 的展宽规则） | ✅ M2e |
| 占据 | Fermi 展宽（玻尔兹曼常数与 DFTB+ 25.1 相同） | ✅（常数 M2e 统一） |
| | Methfessel-Paxton、Gaussian 展宽（含 DFTB+ 的费米能级搜索与 TS 约定，另提供变分 TS） | ✅ M1 |
| 混合器 | Pulay 型（Anderson） | ✅ |
| | 改进 Broyden（DFTB+ 默认；SCC 迭代轮数与 DFTB+ 完全一致） | ✅ M1 |
| 排斥势 | 样条 | ✅ |
| | 多项式（`repulsive="polynomial"` = DFTB+ `PolynomialRepulsive`） | ✅ M1（M0 修复了系数错位） |
| 多类型 | 同一元素多个类型标签（`Type2FileNames` 约定） | ✅（M0，已与 DFTB+ 对照） |
| 自旋 | 共线自旋（固定或自由总自旋，每种原子一个 W 或 W 矩阵，含力、应力） | ✅ M2c |
| | 非共线自旋、自旋轨道耦合 | ⬜ M6 |
| 修正项 | 色散：Lennard-Jones/UFF、Slater-Kirkwood（原生，周期体系用 r⁻⁶ Ewald 求和）；D3、SimpleDftD3、D4（调用 `dftd3`/`dftd4` 参考库）；D3H5 的 H–H 排斥 | ✅ M2d |
| | 色散：TS/MBD、D4 自洽电荷（`ChargeModel = SelfConsistent`） | — 按需（libmbd） |
| | DFTB+U、onsite 修正、长程校正 | ⬜ M6 |
| | 溶剂、外场、QM/MM | ⬜ M6 |
| 驱动 | 结构优化（含晶格）、NVE 分子动力学（通过 ASE），与 DFTB+ 的 `GeometryOptimisation`、`VelocityVerlet` 对照 | ✅ M2f |
| | 几何步之间沿用上一步的电荷（`PyDFTB(reuse_charges=True)`，DFTB+ 驱动的做法） | ✅ M2f |
| | 力和应力用最后一轮的输出电荷（DFTB+ `RecomputeAfterDensity` 的默认做法） | ✅ M2f |
| | 热浴、压浴、约束 | — 用 ASE 的现成实现 |
| | XL-BOMD | ⬜ M6 |
| 激发态 | Casida | ⬜ M6 |
| | 实时 TD、REKS、pp-RPA | — 按需 |
| 输运 | NEGF | — 调用 libNEGF / 按需 |
| 求解器 | LAPACK 本征分解、批量 | ✅ |
| | SP2 纯化（稠密、混合精度、稀疏线性标度） | ✅ 原型 |
| | GPU、分治、多机 | ⬜ M4/M7 |
