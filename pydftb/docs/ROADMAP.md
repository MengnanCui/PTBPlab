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
| M3d | DFTB+ 配置顾问：HSD 读取器、规则目录、`python -m pydftb.advisor`、PTBP 钩子 | ✅ 完成（18 条规则，各有出处；11 次冻结的 DFTB+ 25.1 运行上结论全部符合预期；PTBP 的历史输入被查出 5 条：ParserVersion 1 带来的壳层分辨 SCC 和 0.01 Bohr 差分步长、写在 Filling 外被忽略的展宽（实际 T = 0）、8 MB 的栈、默认 Ewald 容差，前两项标为预期；PTBP 用 DFTB+ 时自动检查第一个运行和失败的运行，只报告，结果逐字节相同） | `design/M3d-advisor.html`，`reports/M3d-advisor.html` |
| M4a | PyTorch 后端（前向）：k 点折叠、Cholesky、本征分解、密度矩阵、布居、能量的迹和力用到的实空间密度在 torch 设备上算；`backend="torch"`、`device`、`dtype`；NumPy 路径逐字节不变（golden 工具） | ✅ 完成（74 个单点和 11 个优化/MD 的 DFTB+ 参考在 torch 上全部通过，SCC 轮数与 DFTB+ 相同；与 NumPy 的最大差能量 3.6e-14 Ha、力 4.1e-14 Ha/Bohr、电荷 1.4e-13 e；float32 下能量差 ≤ 4.3e-6 Ha；发现 threadpoolctl 管不到 torch 的 MKL，fork 后子进程会卡死，已修正） | `design/M4a-torch-backend.html`，`reports/M4a-torch-backend.html` |
| M4b | 自动求导：两种后端通用的核函数、`pydftb.autodiff`（可微参数、密度矩阵的差商反向、SCC 隐式微分、HF 替代能量），与有限差分对照 | ✅ 完成（`evaluate` 的 `mode="hf"` 与 `mode="implicit"`：SK 表、onsite、U、Ud、自旋常数、排斥势系数和坐标、晶胞的导数，返回值即引擎的数；与引擎有限差分对照 346 项全部通过（导数 > 1e-4 的项：hf 最大 7.6e-7，implicit 最大 2.8e-6，中位数 2e-9）；hf 与 implicit 逐项 ≤ 8.7e-10，implicit 与展开的 SCC ≤ 4e-12；以力为损失用 PTBP 的 0.01 Bohr 差分核；发现 DFTB+ 的带隙中点费米能级与 MP 的 TS 使 F 不是电子数固定的 Mermin 自由能，新增 `Smearing(fermi_level="exact")`） | `design/M4b-autodiff.html`，`reports/M4b-autodiff.html` |
| M4c | PTBP 损失对 r0_w、r0_d 的梯度（自动求导 × hotcent 有限差分），可选的梯度优化器，与 BO 对比 | ✅ 完成（hotcent 的中心差分切向 × pydftb 的导数 × 损失的解析偏导，σ_rep 用包络定理，含折点和边界；torch 与独立的 NumPy 收缩差 1e-15，沿真实切向的参数差分 2e-11–5e-10，dataset 模式整条流程的差分 2e-4；默认 BO 逐字节不变；一次“值 + 梯度”约 1.4–2.6 个 BO 步；与 BO 对比（Cu，4 核）：energygeometry 模式梯度法 635 s 把损失从 1.899 降到 0.760，3 次 BO（各 11 次）最好 0.855，用同样时间的 BO 没有比种子好；dataset 模式打平（209.9 对 208.5），梯度法受 T = 0 下 SCC 不收敛拖累，BO 被失败值（97–99，低于这个模式的真实损失）引到失败区；发现 hotcent 2.1 下 r0_d 不改变 SKF、energygeometry 损失在发表点附近不连续、PTBP 默认的 BO 是种子加 10 个随机点；pydftb 的重叠矩阵不正定改报 `CalculationFailed`，与 DFTB+ 一致） | `design/M4c-ptbp-gradient.html`，`reports/M4c-ptbp-gradient.html` |
| M4d | 设备选项（CUDA/ROCm、MPS、XPU，按设备能力把部分步骤留在 CPU，CUDA 上确定性）、GPU 自检脚本（`python -m pydftb.validation.gpu_check`）、PTBP 的 `--pydftb_backend`/`--device`/`--dtype` | ✅ 完成（本机没有 GPU，GPU 的正确性和速度未实测，由用户运行自检脚本；在 CPU 上用 meta 设备检查查出并改正自动求导中约 50 处建在 CPU 上的张量，MPS/XPU 布局的退回 CPU 分支与正常路径逐位相同，自检脚本 155 项检查全部通过；PTBP 用 GPU 时 DFTB 阶段在主进程串行，与 parallel_bo/pso 同用时拒绝，梯度优化器只用 NumPy 后端，`plan.json` 记录后端；顺带修正自动求导的两处色散错误；默认 NumPy 路径逐字节不变） | `design/M4d-devices.html`，`reports/M4d-devices.html` |
| M4e | 恢复 r0_d 的作用：hotcent 原子计算默认用自洽约束（hotcent 1.x 的做法），`--confinement perturbative` 保留旧行为；参数化坚持每个元素三个参数（r0_w、r0_d、σ_rep） | ✅ 完成（r0_d 不起作用的原因是 hotcent 在 2023-02 把微扰约束改成默认，而 PTBP 是按 1.x 写的；自洽约束下 Cu 的 r0_d 从 12.036 改到 13 时 SK 表变化 0.011；发表参数下 energygeometry 损失 1.899 → 0.920（论文 0.288），μ_V 0.120 → 0.0129（论文 0.0182），μ_E 剩下的差距最可能来自 hotcent 2021-10 修正的 GGA 错误；两个引擎仍一致；`--confinement perturbative` 时与 M4 之前逐字节相同） | `design/M4e-r0d-confinement.html`，`reports/M4e-r0d-confinement.html` |
| M5 | 同一元素多套参数（离散类型 → 平滑切换），固态电池示范 | 计划，排在 M6 之后（离散类型的基础已在 M0 完成） | — |
| M6 | DFTB+ 功能补齐·第二波，按顺序：M6a 自旋轨道耦合与非共线自旋 → M6b DFTB+U 与 onsite 修正 → M6c 外电场、点电荷与隐式溶剂 → M6d 激发态 Casida → M6e 激发态的力与 XL-BOMD（M6d 原计划包括后两者，按“量大时单列”拆出）→ M6f 常用缺口与分析输出 → M6g 响应性质 → M6h 长程校正 LC/CAM（含 TD-LC）；新功能先只做 NumPy 引擎（torch 后端、自动求导遇到时报错）；PTBP 的参数化不变（每个元素 r0_w、r0_d、σ_rep 三个参数） | 进行中：M6a ✅（两分量计算：非共线自旋，onsite 与 dual 两种 SOC 模型，可与 DFTB3、壳层分辨 SCC、k 点、力、应力、能带/DOS 同用；16 个 DFTB+ 参考算例全部通过，能量差 ≤ 4.9e-11 Ha，SCC 轮数全部相同（最长 166 轮），磁矩向量、自旋与轨道角动量 ≤ 7.6e-9，dual 模型的力与能量差分一致到 6e-9；旧路径逐位不变；发现磁矩方向是 SCC 的软模，周期体系倾斜的磁矩或很弱的 SOC 下 DFTB+ 的 Broyden 混合发散（pydftb 相同，Pulay 混合可收敛），以及 SOC 下约化的 k 点网格让 DFTB+ 算错磁矩）；M6b ✅（DFTB+U 的 FLL 与 pSIC、onsite 修正，两者共用块布居、½(SV + VS) 的块势和 DFTB+ 的混合向量（被修正的轨道各自混合，块的非对角元也混合）；可与无自旋、共线、非共线自旋、dual SOC（虚部块势）、DFTB3、k 点、力、应力、能带同用；18 个 DFTB+ 参考算例全部通过，能量差 ≤ 5.6e-11 Ha，逐轨道布居 ≤ 8e-11，17 个的 SCC 轮数相同、另一个差 1 轮；水的 onsite MD 与 DFTB+ 一致到 4.5e-8 Å；发现只有 s 壳层的 +U 组让 DFTB+ 25.1 崩溃，以及带隙中点费米能级在近简并能级对下让力与能量差分相差 2.7e-7）；M6c ✅（外电场（周期体系为锯齿势）、外部点电荷（高斯展宽、周期体系 Ewald 求和，给出点电荷受的力，实现 ASE 的 QM/MM 嵌入接口）、Gross/Net 原子格点势、分子的隐式溶剂（广义 Born 的 Still、P16 核与 ALPB，SASA，氢键修正，CM5，溶剂常数表与参考态，xtb 格式参数文件）；可与自旋、非共线自旋、SOC、+U、onsite、DFTB3、k 点、力、应力、能带同用；22 个 DFTB+ 参考算例全部通过，能量差 ≤ 1.0e-10 Ha，SCC 轮数全部相同，点电荷受的力 ≤ 5e-12，溶剂中的 MD 一致到 3.9e-8 Å；发现 DFTB+ 25.1 在 GB 中把 CM5 修正算了两次、CM5 与氢键修正同用时力与能量不一致、外场的应力未对称化且在非立方晶胞中算错）；M6d ✅（激发态 Casida 线性响应：激发能、振子强度、跃迁偶极、主要跃迁与权重；单重态、三重态、共线自旋极化（Δ⟨S²⟩），分数占据、onsite 修正、Γ 点周期体系、能量窗口与振子窗口、指定态与最亮态，可与外电场、点电荷、色散同用；稠密、分块 Davidson、ARPACK 三种求解器，默认跃迁超过 500 个时用 Davidson；DFTB+ 格式的 EXC.DAT、SPX.DAT；18 个 DFTB+ 参考算例全部通过（三种求解器分别对照），激发能差 ≤ 1.8e-11 Ha，振子强度 ≤ 6.1e-10，打印的表全部一致；C₆₀ 的激发部分 0.46 s，DFTB+ 的 Stratmann 1.46 s、ARPACK 9.1 s；发现 DFTB+ 在指定态时输出的“总能量”仍是基态能量、两个求解器在部分占据 d 能级的 Cu₄CO 上都失败、Stratmann 使简并态的振子强度只准到约 5e-7）；M6e ✅（激发态的力：Z 向量方法求 E₀ + ω 的完整导数，单重态、三重态、共线自旋极化，含外场、点电荷、格点势、色散、H–X 阻尼与能量窗口（正则轨道乘子）；激发态的 Mulliken 电荷与偶极（XCH.DAT、XREST.DAT 逐字节相同）；ASE 中的激发态优化与 MD；XL-BOMD：DFTB+ 的 `Xlbomd`/`XlbomdFast` 积分器、影子能量与它的力和应力、`ForceEvaluation`；11 个单点与 9 个动力学的 DFTB+ 参考算例全部通过：用 DFTB+ 的差分步长时激发态力差 ≤ 1.8e-12 Ha/Bohr，7 条 XL 轨迹每步的 SCC 轮数相同、位置速度能量一致到 DFTB+ 打印的位数；发现 DFTB+ 的激发态力在外场、点电荷、H–X 阻尼、能量窗口、按壳层自旋常数五种组合下不是能量的导数、`DynamicsT0` 在自旋极化时给出错误的力，以及验证工具的差分步长应为 DFTB+ 的 2⁻¹³ 而不是 1.22e-4）；M6f 下一步（常用缺口与分析输出），之后 M6g（响应性质）、M6h（长程校正 LC/CAM，含 TD-LC） | `design/M6a-spin-orbit.html`，`reports/M6a-spin-orbit.html`；`design/M6b-dftbu-onsite.html`，`reports/M6b-dftbu-onsite.html`；`design/M6c-fields-solvation.html`，`reports/M6c-fields-solvation.html`；`design/M6d-casida.html`，`reports/M6d-casida.html`；`design/M6e-excited-forces-xlbomd.html`，`reports/M6e-excited-forces-xlbomd.html` |
| M7 | 大体系：GPU 混合精度 SP2、Chebyshev、分治多机 | 计划，排在 M6 之后（CPU 原型见 `RESEARCH.md`） | — |
| M8 | AI 层：随环境连续变化的参数、ML 排斥势、主动学习 | 计划，排在 M6 之后 | — |

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
| | DFTB+ 的 Simple、Anderson、DIIS 混合器与 Broyden 的权重选项（SCC 轮数逐轮一致） | ⬜ M6f |
| | 改进 Broyden（DFTB+ 默认；SCC 迭代轮数与 DFTB+ 完全一致） | ✅ M1 |
| 排斥势 | 样条 | ✅ |
| | 多项式（`repulsive="polynomial"` = DFTB+ `PolynomialRepulsive`） | ✅ M1（M0 修复了系数错位） |
| 多类型 | 同一元素多个类型标签（`Type2FileNames` 约定） | ✅（M0，已与 DFTB+ 对照） |
| 自旋 | 共线自旋（固定或自由总自旋，每种原子一个 W 或 W 矩阵，含力、应力） | ✅ M2c |
| | 非共线自旋（三维磁矩向量）、自旋轨道耦合（onsite 与 dual 模型，s/p/d/f），含力、应力、能带/DOS、轨道角动量；SOC 要求完整的 k 点网格 | ✅ M6a（只有 NumPy） |
| 修正项 | 色散：Lennard-Jones/UFF、Slater-Kirkwood（原生，周期体系用 r⁻⁶ Ewald 求和）；D3、SimpleDftD3、D4（调用 `dftd3`/`dftd4` 参考库）；D3H5 的 H–H 排斥 | ✅ M2d |
| | 色散：TS/MBD、D4 自洽电荷（`ChargeModel = SelfConsistent`） | — 按需（libmbd） |
| | H5、卤素修正（HalogenXCorr）、偶极修正 | ⬜ M6f |
| | DFTB+U（FLL、pSIC，可按组耦合多个壳层）、onsite 修正（附 DFTB+ 手册的 PBE 常数），含力、应力、能带；可与自旋、SOC（dual）、DFTB3 同用 | ✅ M6b（只有 NumPy） |
| | 长程校正 LC/CAM（含 TD-LC） | ⬜ M6h（先解决带 RangeSep 的测试参数） |
| 环境 | 外电场（分子；周期体系为锯齿势，含应力）、外部点电荷（静电嵌入、高斯展宽、周期 Ewald，点电荷受的力，ASE 的 QM/MM 接口）、Gross/Net 原子格点势 | ✅ M6c（只有 NumPy） |
| | 隐式溶剂：广义 Born（Still、P16 核，ALPB）、SASA、氢键修正、CM5、参考态、xtb 格式参数文件；溶剂中外场按 3ε/(2ε+1) 放大 | ✅ M6c（只有 NumPy，只用于分子，与 DFTB+ 相同） |
| | COSMO（区域分解）、含时外场（DFTB+ 自己的 MD 驱动） | — 按需 |
| 响应性质 | 静态与频率相关的极化率、响应核、Born 电荷、对点电荷的导数（DFTB+ 的 `Analysis` 块）；数值 Hessian（`SecondDerivatives`） | ⬜ M6g |
| 驱动 | 结构优化（含晶格）、NVE 分子动力学（通过 ASE），与 DFTB+ 的 `GeometryOptimisation`、`VelocityVerlet` 对照 | ✅ M2f |
| | 几何步之间沿用上一步的电荷（`PyDFTB(reuse_charges=True)`，DFTB+ 驱动的做法） | ✅ M2f |
| | 力和应力用最后一轮的输出电荷（DFTB+ `RecomputeAfterDensity` 的默认做法） | ✅ M2f |
| | 热浴、压浴、约束 | — 用 ASE 的现成实现 |
| | XL-BOMD（DFTB+ 的 `Xlbomd`/`XlbomdFast`：K = 5、6、7 的积分器、影子能量与它的力和应力，`ForceEvaluation` 的 DynamicsT0/Dynamics；限制与 DFTB+ 相同） | ✅ M6e（只有 NumPy） |
| 激发态 | Casida 线性响应：激发能、振子强度、跃迁偶极、主要跃迁与权重；单重态、三重态、共线自旋极化（Δ⟨S²⟩）；分数占据、onsite 修正、Γ 点周期体系、能量窗口与振子窗口、指定态与最亮态；稠密、分块 Davidson、ARPACK 求解器；EXC.DAT、SPX.DAT | ✅ M6d（只有 NumPy） |
| | 激发态的力（Z 向量，含外场、点电荷、格点势、色散、H–X 阻尼、能量窗口的完整导数）、激发态的 Mulliken 电荷与偶极（XCH.DAT、XREST.DAT）、激发态的结构优化与 MD | ✅ M6e（只有 NumPy；分子、整数占据，与 DFTB+ 相同） |
| | 非绝热耦合、锥形交叉优化、激发态的自然轨道 | — 按需 |
| | 实时 TD、REKS、pp-RPA | — 按需 |
| 输运 | NEGF | — 调用 libNEGF / 按需 |
| 求解器 | LAPACK 本征分解、批量 | ✅ |
| | SP2 纯化（稠密、混合精度、稀疏线性标度） | ✅ 原型 |
| | PyTorch 后端（CPU 已验证）；设备选项（CUDA/ROCm、MPS、XPU）与 GPU 自检脚本 | ✅ M4a、M4d |
| | GPU 实测（等用户运行自检脚本）、跨结构批量、分治、多机 | ⬜ M7 |
| 自动求导 | 对 SK 表、onsite、U、Ud、自旋常数、排斥势系数及坐标、晶胞的导数（hf：变分的 dF/dθ；implicit：电荷、本征值、力等的全导数） | ✅ M4b |
| | PTBP 损失对 r0 的梯度、梯度优化器（`--optimizer gradient`、`bo_grad`） | ✅ M4c |
