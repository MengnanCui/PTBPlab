# PTBP 业务路线图：参数化与多参数的优化策略

这是**业务线**的路线图：PTBP 用什么参数、拿什么数据和损失来拟合、用什么策略优化这些参数、怎样把一次拟合跑起来。DFTB 计算本身（pydftb 引擎与 DFTB+ 的对照）是另一条线，见 [`../pydftb/docs/ROADMAP.md`](../pydftb/docs/ROADMAP.md)；两条线的关系见总计划 [`../PLAN.md`](../PLAN.md)。**目前的重点是这一条线。**

设计文档与报告仍放在 `pydftb/docs/design/`、`pydftb/docs/reports/`（文件名不变，链接不断）。

## 一、目标与验收（A0，用户 2026-11-09 定）

- 一套 DFTB 参数集，至少在能量上（成键、断键）达到 DFT 精度。
- 验收线（A 级）：测试集上能量差 MAE ≤ 10 meV/原子、力 MAE ≤ 0.1 eV/Å，决定性的能量差符号全部正确，300–600 K 的 MD 稳定。
- 能量按配平的能量差算，绝对原子化能只作诊断；拟合损失的默认仍是绝对原子化能。
- 试用体系：Li、Li | Li₂O 界面、Pb–Sn 钙钛矿前驱体溶液（晶体只作参考）。
- 路线按顺序走，每一阶段结束时按家族外测试集上的数字决定下一步（关口 G1–G4）：
  - A1：单一参数集，比较几种拟合策略；
  - A2：参数随环境变化，排斥势冻结；
  - A3：排斥势这一维；
  - A4：最后才做 Δ-ML。
- 详细计划：[`A0-accuracy-programme.html`](../pydftb/docs/design/A0-accuracy-programme.html)。

## 二、下一步（建议顺序，待确认）

| 序 | 工作 | 状态 | 需要什么 |
|---|---|---|---|
| 1 | A1.0 便宜的检验 K1–K7（本机或服务器，不花钱） | ⬜ 代码已备好（`examples/08_accuracy/`） | 在服务器上运行 |
| 2 | A1 的策略比较：同一批 SK 点上比较 S1 能量 + 力、S2 EOS、S3 先能带后排斥势、S4 能量 + 力加能带正则，并与对照行 R₂b、oracle 比较（`ptbp scan`） | ⬜ 代码已备好，计算未开始 | D1 能带参考；D2–D5 的 DFT 数据（粗估 450–750 元，等批准） |
| 3 | physrep 的按原子对排斥势与短程墙接到 `optimize` | ⬜ 部分完成（见第五节） | — |
| 4 | 内层 σ_rep 搜索的边界问题（见第五节）：决定是否改变默认结果 | ⬜ | 用户决定 |
| 5 | dataset/reaction 损失可选的“按元素能量偏移” | ⬜ 可选 | — |
| 6 | 随环境变化的参数（A2；M8 的 Cu 自适应模型是起点）：第三轮主动学习的 DFT 帧 | ⬜ 输入已写好，未提交 | 用户批准作业 |
| 7 | PTBP 的 DFTB 调用改用 `pydftb.engines`（结果须逐字节不变） | ⬜ 待定 | — |

## 三、四个方向

### 1. 参数模型：拟合什么

| 层级 | 内容 | 状态 |
|---|---|---|
| 每个元素三个参数 | r0_w（波函数约束）、r0_d（密度约束）、σ_rep（排斥势）；hotcent 默认用自洽约束，r0_d 才起作用 | ✅ M4e |
| 多元素各一套 | 多元素数据集默认每个元素一套（2N 维 + σ_rep 的 N 维内层搜索），`--shared-parameters` 复现共用一套，`--multi_element` 只拟合列出的元素 | ✅ M5b |
| 同一元素的几个类型 | 按配位数平滑切换（引擎机制 M5a），PTBP 按类型拟合 | ✅ M5b；Li \| Li₂O 上结论否定（M5c、M5d） |
| 随环境连续变化 | on-site、跃迁积分与排斥势的标度是配位数（及角度描述符 q）的光滑函数；Cu 上能量 + 力的模型训练 24.2 meV/原子、MD 稳定 | ✅ 框架 M8a–M8d；推广到业务体系是 A2 |
| 排斥势 | physrep（σ_rep、kxc）；按原子对的标度与短程墙；样条排斥势 `repfit`；排斥势这一维是 A3 | 部分完成 |
| Δ-ML | 最后才做 | ⬜ A4 |

### 2. 损失与数据：拿什么来比

| 内容 | 状态 |
|---|---|
| 四种模式：energygeometry（EOS）、dataset（能量与力）、reaction（多组成）、bandstructure（能带） | ✅（bandstructure 模式在 eloquent-archimedes 合并后能从头跑通） |
| energygeometry 的 Boltzmann 因子在每种组成内计算 | ✅ M5b-4 |
| 失败点的损失 = 本次运行至今最差成功损失的 2 倍，失败序号记在检查点 | ✅ M5b-2 |
| 能带参考与能带损失（`pydftb.bandref`、`pydftb.bandloss`）、评测工具（`pydftb.assess`） | ✅ 代码（A0） |
| 数据批 D1–D5（能带参考、Li、界面、钙钛矿前驱体溶液） | ⬜ 输入已写，未提交 |
| 可选：按元素能量偏移的损失；DFT 侧 E0（最小二乘拟合）与 DFTB 侧 E0（孤立原子）不一致 | ⬜ 见第五节 |

### 3. 多参数的优化策略：怎样找参数

| 策略 | 内容 | 状态 |
|---|---|---|
| BO（默认） | `gp_minimize`，种子点加随机点；默认 `--n-calls`：维数 ≤ 2 时 11，以上 10 + 5 × 维数；`--seed` 设随机种子（默认 123） | ✅ |
| 并行 BO、PSO | `--optimizer parallel_bo`、`pso`，按方案并行评估 | ✅ M3 |
| 梯度法 | `--optimizer gradient`、`bo_grad`：损失对 r0 的梯度（hotcent 差分切向 × pydftb 自动求导 × 损失的解析偏导）；Cu 的 energygeometry 上比 BO 好（0.760 对 0.855），dataset 模式打平 | ✅ M4c |
| σ_rep 内层搜索 | 共用 σ 的一维搜索 → 各参数集的格子 → 联合 L-BFGS-B；`--sigma-range` 限定范围 | ✅ M5b、M5c；边界问题见第五节 |
| 策略比较 | `ptbp scan`：同一批 SK 点上比较 S1–S4，消除“优化器不同”这个差别 | ✅ 代码，⬜ 计算（A1） |
| 高维参数（A2） | 随环境变化的参数维数高，要线性/最小二乘部分与非线性部分分开（`pydftb.adaptive_fit` 已用这个办法） | ⬜ A2 |

### 4. 运行与工具：怎样跑起来

| 内容 | 状态 |
|---|---|
| 并行方案：DFTB 阶段按结构并行、hotcent 分片、physrep 向量化，一个 BO 步快 5.4 倍；成本模型与 `ptbp plan` | ✅ M3a–M3c |
| DFTB+ 配置顾问的 PTBP 钩子（只报告） | ✅ M3d |
| GPU：PTBP 的小结构上 GPU 慢于 CPU 进程池，PTBP 默认用 CPU 进程池 | ✅ M7e-2 的结论 |
| 引擎：`--engine dftbplus`（默认）或 `pydftb`，损失函数相对差 < 1e-7 | ✅ M1 |
| 界面：命令行子命令、TUI、桌面 GUI（`ptbp gui`） | ✅ |
| hotcent 随仓库提供（`third_party/hotcent`，离线安装） | ✅ |

## 四、已完成阶段的记录

| 阶段 | 内容 | 状态 | 文档 |
|---|---|---|---|
| M1 | 接入 PTBP：`--engine pydftb`、`--smearing` 开关、MP/Gaussian 展宽、Broyden 混合、排斥势开关、PTBP 损失函数等价性 | ✅ 完成（损失函数相对差 < 1e-7，DFTB 阶段快 1.8 倍） | `../pydftb/docs/design/M1-ptbp-integration.html`，`../pydftb/docs/reports/M1-ptbp-integration.html` |
| M3a | 并行分发与 DFTB 阶段：CPU 预算与方案、fork 进程池（最大优先）、DFTB 阶段按结构并行、pydftb 的 k 点线程、`--ncores`（默认用满核心） | ✅ 完成（Cu 数据集 4 核：DFTB 阶段 pydftb 3.4 倍、DFTB+ 4.1 倍，整个 BO 步 1.2–1.5 倍；与串行的输出逐字节相同、损失值逐位相同；pydftb 的 k 点线程逐位相同；DFTB+ 默认的 OpenMP 线程在这些小体系上反而更慢） | `../pydftb/docs/design/M3a-dispatch.html`，`../pydftb/docs/reports/M3a-dispatch.html` |
| M3b | BO 步的其余部分：hotcent 按行分片并回放（逐字节相同）、physrep 向量化（逐位相同） | ✅ 完成（4 核一个 BO 步：pydftb 122 → 27 s、DFTB+ 132 → 40 s，对 M3 之前快 5.4 倍；运行目录、`log.out`、`par.out` 逐字节相同；hotcent 原来把单元素的表算了三遍；σ_rep 搜索的 74 次评估中 41 次白算，未改） | `../pydftb/docs/design/M3b-hotcent-physrep.html`，`../pydftb/docs/reports/M3b-hotcent-physrep.html` |
| M3c | 计算策略推荐器 v1（CPU）：成本模型、内存估计、`ptbp plan`、`plan.json` | ✅ 完成（成本模型按轨道数、k 点数和截断内的原子对估计 pydftb、DFTB+ 单点的耗时和内存：耗时中位误差 12%/7%，排序秩相关 0.99，pydftb 内存中位误差 2%；Cu 数据集一个 BO 步的预计耗时与实测相差 2–4%；方案只调不改变数值的设置，运行目录与 M3b 逐字节相同；发现大晶胞的耗时和内存主要在库仑部分：最小的 Hubbard U 决定截断，Cu 的 4p 为 96 Bohr，64 个原子的晶胞 pydftb 要 2.2 GB，DFTB+ 0.3 GB；DFTB+ 的 OpenMP 线程在 500 个轨道以下反而更慢） | `../pydftb/docs/design/M3c-recommender.html`，`../pydftb/docs/reports/M3c-recommender.html` |
| M4c | PTBP 损失对 r0_w、r0_d 的梯度（自动求导 × hotcent 有限差分），可选的梯度优化器，与 BO 对比 | ✅ 完成（hotcent 的中心差分切向 × pydftb 的导数 × 损失的解析偏导，σ_rep 用包络定理，含折点和边界；torch 与独立的 NumPy 收缩差 1e-15，沿真实切向的参数差分 2e-11–5e-10，dataset 模式整条流程的差分 2e-4；默认 BO 逐字节不变；一次“值 + 梯度”约 1.4–2.6 个 BO 步；与 BO 对比（Cu，4 核）：energygeometry 模式梯度法 635 s 把损失从 1.899 降到 0.760，3 次 BO（各 11 次）最好 0.855，用同样时间的 BO 没有比种子好；dataset 模式打平（209.9 对 208.5），梯度法受 T = 0 下 SCC 不收敛拖累，BO 被失败值（97–99，低于这个模式的真实损失）引到失败区；发现 hotcent 2.1 下 r0_d 不改变 SKF、energygeometry 损失在发表点附近不连续、PTBP 默认的 BO 是种子加 10 个随机点；pydftb 的重叠矩阵不正定改报 `CalculationFailed`，与 DFTB+ 一致） | `../pydftb/docs/design/M4c-ptbp-gradient.html`，`../pydftb/docs/reports/M4c-ptbp-gradient.html` |
| M4e | 恢复 r0_d 的作用：hotcent 原子计算默认用自洽约束（hotcent 1.x 的做法），`--confinement perturbative` 保留旧行为；参数化坚持每个元素三个参数（r0_w、r0_d、σ_rep） | ✅ 完成（r0_d 不起作用的原因是 hotcent 在 2023-02 把微扰约束改成默认，而 PTBP 是按 1.x 写的；自洽约束下 Cu 的 r0_d 从 12.036 改到 13 时 SK 表变化 0.011；发表参数下 energygeometry 损失 1.899 → 0.920（论文 0.288），μ_V 0.120 → 0.0129（论文 0.0182），μ_E 剩下的差距最可能来自 hotcent 2021-10 修正的 GGA 错误；两个引擎仍一致；`--confinement perturbative` 时与 M4 之前逐字节相同） | `../pydftb/docs/design/M4e-r0d-confinement.html`，`../pydftb/docs/reports/M4e-r0d-confinement.html` |
| M5 | 同一元素多套参数（离散类型 → 平滑切换），固态电池示范 | 进行中，与 M6 并行（git worktree）：M5a ✅（同一元素多套参数的平滑切换，引擎机制：被切换元素的每个原子按配位数 CN（紧支撑的五次台阶计数函数）在它的类型之间取权重（C² 的切换窗口，窗口外恰为 0 或 1），H0、S 与排斥能按权重混合，力与应力含权重的导数；要求各类型的 on-site 能、U、占据数、基组相同；可与周期体系、应力、DFTB3、自旋、SOC、+U、外场、溶剂等同用，只有 NumPy；离散极限的 2 个 DFTB+ 参考算例（Cu 平板与 Cu₁₃）能量差 ≤ 2.2e-11 Ha、SCC 轮数相同，切换与显式标签逐位相同；切换区内力与能量差分一致到 5e-10，MD 中守恒量的漂移与不切换时同一量级；发现异标签 SKF 中多出的 on-site 行使 DFTB+ 静默地算错 3.1e-2 Ha，hotcent 2.1 不能处理同一元素的两种约束）→ M5b ✅（PTBP 按类型拟合：参数按类型记账，类型默认就是元素，多元素数据集默认每个元素各一套 r0_w、r0_d、σ_rep（优化器 2N 维，σ_rep 的内层搜索推广到 N 维：共用 σ 的一维搜索 → 各参数集的格子 → 联合 L-BFGS-B，格子还能改进时再来一轮），`--shared-parameters` 复现共用一套（逐字节相同），单元素不变；reaction 示例（B₂N₂ + N₃Na）按元素的损失比共用一套低 60%；可声明同一元素的几个类型，用 M5a 的平滑切换（hotcent 两个方向的交叉表、无 on-site 行、按标签命名，physrep 按标签并按权重混合，只用 pydftb）；罚项为各参数集罚项的几何平均，损失版本 3；梯度优化器支持按元素的参数（2N 个方向），切换时报错；修正 `--multi_element`（现为只拟合列出的元素，其余用 PTBP 发表的参数）与 `dftb_full.xyz` 的 Hill 顺序）→ M5b-2 ✅（PTBP 后续修正：失败点的损失 = 本次运行至今最差成功损失的 2 倍，第一个成功点之前 energygeometry 1000、dataset/reaction 每个结构 1000；有结构没有 DFTB 结果也算失败（旧路径原来在 dataset/reaction 中让运行停止，按类型的路径原来给出偏低的损失）；失败的序号记在检查点元数据 `failed_evaluations`，续跑时从加载的成功点初始化，旧检查点按 97–99 识别；多类型 + energygeometry 打印警告；默认 `--n-calls` 在维数 > 2 时为 10 + 5 × 维数；没有失败的运行逐字节不变）→ M5b-3 ✅（PTBP 的两个旧错误：`--shared-parameters` 的 energygeometry 支持含几个元素的 EOS 组，组内每个元素用共用的 σ_rep，`loss_grad` 的副本同样修改，这时不画单元素的两张图；`get_elements_from_dataset` 接受元素数不同的结构，原来这样的数据集在任何参数布局下都在开始时停止；原来能运行的运行逐字节不变）→ M5b-4 ✅（PTBP 后续修正：energygeometry 的 Boltzmann 因子在每种组成（约化化学式）内分别计算并归一，原来只有每原子总能量最低的组成有权重，单一组成逐位不变，多组成运行的检查点元数据记 `boltzmann`，旧检查点不能续跑；`--shared-parameters` 加列出全部元素的 `--multi_element` 在开始时报错；σ_rep 搜索中 L-BFGS-B 的 σ* 与 μ(σ*) 配对，两条路径；PTBP 测试 286 项））→ M5c ✅（固态电池示范 Li | Li₂O，结论否定：VASP 76 个结构，reaction 损失不改；一套 Li（b30）与按 O 配位切换的两种 Li（c60）把训练集原子化能 MAE 从 0.79 降到 0.60、0.54 eV/原子，但 σ_rep 每一步都落在下界 0.1（排斥势只到 0.8 Å），EOS 最低点被推到 0.88 V₀；80 个原子的界面上三套参数（含发表的）都塌缩，DFT 检验（3 个结构，实收 4.49 元）显示塌缩结构高 1.3–1.4 eV/原子而 DFTB 给出低 0.7–3.4 eV/原子；根源是电子部分 Li₂ 在 1.4 Å 的假势阱加上训练集没有短程；新增 `optimize --sigma-range`，σ ≥ 0.65 时界面稳定、符号正确，但 Li₂O 少结合 2.2 eV/原子；physrep 的 Li–Li 短程墙在不改训练集的情况下把界面符号改对） | `../pydftb/docs/design/M5c-battery-demo.html`，`../pydftb/docs/reports/M5c-battery-demo.html`；`../pydftb/docs/design/M5a-type-switching.html`，`../pydftb/docs/reports/M5a-type-switching.html`；`../pydftb/docs/design/M5b-ptbp-types.html`，`../pydftb/docs/reports/M5b-ptbp-types.html`；`../pydftb/docs/reports/M5b2-ptbp-followups.html`；`../pydftb/docs/reports/M5b3-ptbp-fixes.html`；`../pydftb/docs/reports/M5b4-ptbp-fixes.html` |
| M8 | AI 层：随环境连续变化的参数、ML 排斥势、主动学习 | 进行中：M8a ✅（随环境连续变化的参数：`DFTB(adaptive=AdaptiveParams(...))`，每个原子的 on-site 能量、跃迁积分的标度、排斥势的标度是配位数 CN 的光滑函数（RBF 基，对系数线性），加可拟合的对排斥势与固定的短程墙；力与应力精确（周期金属对差分 2.4e-9 Ha/Bohr，应力 4e-11），每次计算给出 ∂F/∂θ（相对误差 1e-9），`pydftb.adaptive_fit` 用它做最小二乘；系数为零时逐位不变，golden 1523 个计算逐位相同；Cu 的 6 个相 × 11 个体积：拟合 244（原来的表）→ 15.6（固定）→ 1.4 meV/atom（自适应），留一相交叉验证 117 → 40 meV/atom（6 个相里 5 个改进，diamond 没有）；只用理想结构能量拟合的自适应模型在 Cu13 的 MD 中不稳定（团簇重构、原子蒸发），固定模型稳定（2 ps 总能量范围 2.2 meV），要有力与畸变结构的数据）；M8b 框架 ✅（力进入拟合：`Sample(forces=)`、`fit_adaptive(force_weight=)`，力的 Jacobian 用系数的中心差分，与解析 ∂F/∂θ 的坐标导数一致，用力拟合恢复已知模型到 1e-4；角度描述符 q = Σ f f P3(cosθ)/(CN²+ε)，fcc/bcc 0、diamond 5/9、石墨 0.62，dq/dR 对差分 < 1e-8；在 6 个理想相上过拟合：拟合 1.4 → 0.5 meV/atom，但留出 diamond 76 → 224、留出 β-Sn 19 → 61，没有解决 diamond，要畸变结构的数据；Cu 的 VASP 训练集 176 个结构、估计 1403 核时、约 140 元，只生成未提交，等用户决定），之后是带力的真实拟合与神经网络的 Φ；M8c 框架 ✅（`pydftb.active`：bootstrap 委员会、锚点扰动（anchored ensemble）、σ_E 与 σ_F、按 CN 与 q 的指纹去重挑结构、写 VASP 输入；校准：不含 diamond 的委员会，锚定在基础拟合时 diamond 的 σ 反而小（对数相关 0.00），锚点扰动 0.02 后 diamond 的 σ 是见过的相的 17 倍（相关 0.75），但仍低估误差约 9 倍；M8a 的 Cu13 轨迹上 σ 随重构上升，挑出 10 帧（200–380 fs），VASP 输入只生成未提交）；M8d ✅（精简版 Cu 训练集 142 个结构在电光云 dg83 算完、全部收敛，扣费 7.98 元；`pydftb.bench.cu_dft_fit`：核心墙移到 1.6 Å、对排斥势先做线性最小二乘；只拟合能量的自适应模型训练 39.6 meV/atom，测试：团簇 65、理想相 43 meV/atom，力 0.3–0.4 eV/Å（团簇、体相），Cu13 600 K MD 2 ps 稳定（漂移 0.10 meV/ps），是 M8 第一个 MD 稳定的自适应模型；带力的拟合（权重 0.01）没收敛，228 meV/atom，MD 中原子飞出；委员会在出界轨迹上半数帧 SCC 不收敛，第二轮 10 帧（约 1.5 元）只生成未提交；修了拟合的鲁棒性：SCC 失败的试探步被拒绝、不收敛时用更慢的混合重试、macOS 上 fork 子进程在 Accelerate 里段错误（用 VECLIB_MAXIMUM_THREADS=1 与 ProcessPoolExecutor）；后续：从能量模型出发、力权重 0.001 再拟合，训练 44.9 meV/atom、力 0.91 eV/Å，测试集重构的 Cu13 168 → 75 meV/atom，MD 2 ps 稳定、漂移 0.00 meV/ps，是目前最好的模型；从 DFT 二聚体拟合的排斥势样条（`pydftb.bench.cu_dimer_repulsive`）让训练降到 22.8，;但压缩的 bcc 过度结合 879 meV/atom、MD 中原子飞出；二聚体、团簇、体相一起只用能量拟合 B 样条排斥势（`pydftb.bench.cu_joint_repulsive`）并让 CN 标度在数据范围外取常数（`RBF(clamp=)`、`cu_dft_fit --cn-clamp`），训练 17.2 meV/atom，但测试集多数变差，Cu2 在 1.8 Å 出现假势阱，MD 1.8 ps 后失守；排斥势的线性拟合加上力（`--force-weight 0.1`）与单调约束（`--monotonic`）、再做力权重 0.001 的精修：训练 24.2 meV/atom、0.32 eV/Å，测试集重构的 Cu13 41 meV/atom、0.53 eV/Å，理想相 39，Cu2 只有 2.2 Å 一个极小，MD 2 ps 稳定、漂移 −0.02 meV/ps，是目前最好的模型；围绕它的纯 bootstrap 委员会（`--perturb 0`）在 149 帧上 SCC 全部收敛（原来 72 帧失败），挑出 1130–1410 fs 的 10 帧替换 round2；第二轮 DFT 7 帧收敛、3 帧超时，2.95 元（累计约 10.9 元）；`cu_dft_fit` 认 `round2` 组；加进训练集重新拟合后 round2 帧 143 → 79 meV/atom，但原来的测试集变差（Cu13 帧 41 → 104），这些帧最短键 1.55–1.72 Å，来自能量加力模型崩掉的轨迹，jfm 自己的 MD 到不了那里；jfm 仍是最好的模型；第三轮在 jfm 自己的高温 MD 上筛（`active_cu` 多段轨迹、`--stride`、`--time-limit`），委员会分歧与训练集同量级，写了 round3 / round3_mixed 的输入，没有提交；压力测试：Cu13 900/1200 K、Cu38/Cu55 600 K、fcc32 300 K 全部稳定，`adaptive_md --structure --kpts`） | `../pydftb/docs/design/M8a-adaptive-params.html`，`../pydftb/docs/reports/M8a-adaptive-params.html`，`../pydftb/docs/design/M8b-forces-angles.html`，`../pydftb/docs/reports/M8b-forces-angles.html`，`../pydftb/docs/design/M8c-active-learning.html`，`../pydftb/docs/reports/M8c-active-learning.html`，`../pydftb/docs/reports/M8d-cu-dft-fit.html` |
| A0–A4 | DFTB 能做到多准：一套参数到环境相关参数再到 Δ-ML，试用 Li、Li \| Li₂O 界面与钙钛矿前驱体溶液 | A0 进行中：计划 ✅（用户 2026-11-09 定：验收线 ≤ 10 meV/原子、≤ 0.1 eV/Å，能量按配平的能量差，钙钛矿以前驱体溶液为主，本轮只写代码不提交作业）；代码已备好（pydftb 的 `bandref`、`bandloss`、`repfit`、`assess` 四个模块，24 项测试；PTBP 的 `ptbp scan`（`utils/skgrid.py`、`utils/accuracy.py`、`cli/cmd_scan.py`），在同一批 SK 点上比较 S1–S4 与 R₂b、oracle 两个对照；`utils/runlog.py` 给每个命令写日志；hotcent 积分网格选项 `hotcent_parallel.set_grid`，默认逐字节不变；`examples/08_accuracy/` 的数据准备、D1 能带参考（GPAW）、D2–D5 的 VASP 输入（不提交）、K1–K7 检验与远程执行手册；M5d 的短程 43 点与界面 3 点 DFT 数据存进 `examples/07_battery_li_li2o/data/`；PTBP `loss.py` 的新选项推迟，optimize 路径不动），计算未开始；A1–A4 ⬜ | `../pydftb/docs/design/A0-accuracy-programme.html` |

## 五、开放事项与结论记录（原在 pydftb 的功能清单里）

| | 内容 | 状态 |
|---|---|---|
| | PTBP 损失对 r0 的梯度、梯度优化器（`--optimizer gradient`、`bo_grad`） | ✅ M4c |
| | PTBP 改用 `pydftb.engines`（PTBP 的结果须逐字节不变） | ⬜ 待定（P2 报告第七节） |
| | PTBP 的 dataset/reaction 损失可选“按元素拟合的能量偏移”（每次评估先对 E_DFTB − E_DFT 按元素做最小二乘的常数偏移，只比与偏移无关的部分）；默认仍比较绝对原子化能（结合能），因为不同结合状态的偏移并不相同，这部分差异正是构型之间的结合能（用户 2026-11-07 的意见）；另记：PTBP 默认的 DFT 侧 E0 是数据集最小二乘拟合的，DFTB 侧却是孤立原子，两者不一致 | ⬜ 可选（M5c 设计文档） |
| | M5d ✅（结论否定）：平滑切换的定位（`../pydftb/docs/design/M5d-switching-landscape.html`：与 FHI 的 DOVE、Bremen 的 TBMaLT、DSKO、GFN-xTB 比较，没有找到对位置可微、带力与 MD 的同类工作）；Li 带 p 后 PTBP 的 Li₂ 在 1.2 Å 假结合 14.9 eV，故 `max_l` 默认跟 PTBP 的拟合基组（占据壳层）并对未用的壳层警告；短程 DFT 43 个单点（作业 981219/981265，实收约 0.60 元）；按 DFT 二聚体拟合的 Li–Li 墙（WF2）、加短程数据的重拟合（b30s4 σ_rep 0.373，不再在下界）与两种 Li 的切换（c60s4）都不改界面塌缩结构的符号（−0.6 到 −2.0 eV/原子，DFT +1.29/+1.44）；WF2 的界面 MD 继续塌缩（NVE 中 500 → 1600 K），c60s4 的弛豫中 Li_ox–Li_ox 熔合到 0.77 Å（Li_ox 的 σ_rep 在下界、没有该类型对的短程数据）后 SCC 不收敛 | `../pydftb/docs/design/M5d-switching-landscape.html`，`../pydftb/docs/reports/M5d-short-range-switching.html` |
| | M5c/M5d 后续（待用户决定）：(1) 训练集加入做过 DFT 的界面快照与更多界面类结构；(2) 类型之间允许 on-site 能与 U 不同；(3) 用 Li p 重新拟合；带切换的拟合要有每个类型对自己环境下的短程数据，拟合后按类型对检查二聚体排斥 | ⬜ 并入 A 系列（`../pydftb/docs/design/A0-accuracy-programme.html`）：界面结构进训练集 → A1 的数据批 D3；类型之间不同的 on-site/U → A2 的通道 E1；Li p → A1 的基组因素（M5d 报告第六节） |
| | physrep 的排斥势按原子对分开：`physrep(pair_scale=, pair_wall=)` 已实现，墙可带指数 n（默认 4）；按 DFT 二聚体拟合的 Li–Li 墙（1.61 eV @ 1.4 Å，r_c 1.6 Å，n = 2）去掉 Li₂ 的假势阱、训练集不变，但不改界面的符号，Li–O、O–O 的偏差是少结合，墙修不了（M5d 报告第二、三节）；未接到 `optimize` | ⬜ 部分完成（M5c 报告第七节，M5d 报告第三节） |
| | PTBP 内层 σ 搜索：`brute` 默认用无约束的 `fmin` 精修，返回的 σ ≈ 1e-16 在边界外，L-BFGS-B 从下界开始；改为 `finish=None` 或有界精修会改变默认结果 | ⬜（M5c 报告第八节） |
