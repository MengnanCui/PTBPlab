# A 系列开发日志

按时间记录每一步做了什么、为什么、怎样验证的、还剩什么问题。新的一轮写在最后。运行日志（`logs/…/run.log`、`meta.json`、`events.jsonl`、`status.json`）由脚本自动写，见 `README.md` 第 2 节。

---

## 2026-11-09 计划（A0）

用户的目标是一套至少在能量上（成键、断键）达到 DFT 精度的 DFTB 参数集，按以下顺序推进：

1. 一套参数，比较几种策略：直接拟合能量与力、EOS、先能带再排斥势；
2. 随环境变化的参数，描述符从配位数开始，依次是 2/3 体、SOAP、ACE；
3. 最后才是 Δ-ML。

试用体系是 Li、Li | Li₂O 界面和钙钛矿。

用户的决定（2026-11-09）：

- 钙钛矿以 `../Perovskite` 的前驱体溶液为主，晶体只作参考。
- 验收线：测试集上能量差 MAE ≤ 10 meV/原子、力 MAE ≤ 0.1 eV/Å，决定性符号全对，300–600 K MD 稳定。
- 能量按配平的能量差算，绝对原子化能只作诊断，拟合的默认仍是绝对原子化能。
- 本轮只写计划，不提交作业，可以本地计算。

计划经用户批准，见 `pydftb/docs/design/A0-accuracy-programme.html`。

之后用户要求：执行计划中不涉及计算的部分，代码准备好拿到远程服务器上运行，全程保留日志。

## 2026-11-10 代码准备

### 1. 数据保全（不花钱）

- M5d 的 43 个短程 DFT 点（作业 981219/981265）和 3 个界面 DFT 点原来只在会话的临时目录里，现在存进了仓库：`examples/07_battery_li_li2o/data/short_range.xyz` 与 `interface.xyz`，附 `data/README.md`（设置、来源、费用）。
- 已核对：分别读出 43 个和 3 个结构，都带能量和力；能量范围 −38.3 到 3.6 eV、−277.5 到 −162.5 eV。

### 2. pydftb 的新模块（只依赖 NumPy、SciPy、ASE）

| 模块 | 要点 | 测试 |
|---|---|---|
| `bandref.py` | BandRef JSON（path / ibz / molecule）；特殊点、相邻线段 5% 和 10% 处的点（第一级）与整条线段（第二级）；VASP 的 vasprun、EIGENVAL、OUTCAR 费米能级；ASE BandStructure；ZVAL 与半芯态计数 | `test_bandref.py`，8 项 |
| `bandloss.py` | 窗口带 logistic 边缘；自由平移（闭式解，与导带配对一起迭代）；价带按序号配对，导带单向就近配对，空态权重 0.4；带隙项；第一级 RMS、逐段 MAE、DOVE 𝒜、占据带宽；`dftb_bands` 在 SCC 电荷下、在参考的 k 点上算，不做时间反演约化 | `test_bandloss.py`，7 项，含 Cu 金属路径与 H₂O 分子的 DFTB 自洽对照（损失为 0、平移恢复到 1e-9） |
| `repfit.py` | 多个原子对的 B 样条，线性最小二乘；单调非负（c = U p, p ≥ 0）；二阶差分光滑项；按元素、全局或不加偏移；r_lo 以下用指数头；写 SKF 的 Spline 块（替换已有块或追加） | `test_repfit.py`，5 项：恢复已知模型（能量 < 1e-6 eV/原子，力 < 1e-5 eV/Å），偏移恢复，力与差分一致 |
| `assess.py` | 按角色/家族的原子化能、组内相对能、按元素偏移；力；EOS（Birch–Murnaghan）；二聚体曲线；反应（检查元素配平，`family:` 取 EOS 最低点）；A/B/C 级；NVE 检验 | `test_assess.py`，4 项 |

合计 24 项，全部通过（本机 0.3 s）。

### 3. PTBP 一侧

- **`utils/runlog.py`：** 运行日志，tee stdout/stderr，记录 meta、events 与 status。
- **`utils/skgrid.py`：** 参数空间，box 有三种：`ptbp`、`around`、显式。SK 缓存按内容取键，带文件锁，过期锁可接管。
- **`utils/accuracy.py` 与 `cli/cmd_scan.py`：** `ptbp scan points/run/losses/report/refine/export/status`。
  - 对每个点：SK（缓存）→ 电子部分的能量（E(σ→0)）与力（`repulsive="none"`），SCC 失败时用更慢的混合重试两次 → 孤立原子 → 能带。
  - 每个策略各自做 σ 内层搜索：共用的一维网格，逐元素扫一遍，再做有界的 L-BFGS-B。
  - S2 与 PTBP 的 energygeometry 定义一致：组权重取组中心结构的 Boltzmann 因子，在组成内归一，丢掉小于 0.1 的组。
  - R₂b 与 oracle 用 `repfit` 拟合。
  - 报告按策略取最好的点；一维元素的扫描另画损失地形图；给出 Spearman 相关。
- **`utils/hotcent_parallel.set_grid`** 与 hotpy 参考路径：hotcent 积分网格默认为空，调用与原来完全相同。PTBP 的 hotcent 测试 10 项通过，4 项在 macOS 上照常跳过；`tests/test_cli_subcommands.py` 11 项通过。

### 4. 冒烟测试（本机，小规模，只为验证代码）

数据为 Li，2 个点，9 个结构；hotcent 用默认网格，每个 SK 集 3.7 s。

| 阶段 | 结果 |
|---|---|
| `points`、`run` | 2 个进程，0 个失败结构；孤立 Li −2.875 eV（2s 本征值） |
| `losses`、`report` | 6 个策略、R₂b 与 oracle 都有结果，报告能写出 |
| `export` | 导出的完整 SKF（电子部分加 physrep 的 Spline 块）用 `repulsive="skf"` 重算，与扫描的“电子部分 + physrep”相差 ≤ 1e-8 eV、≤ 3e-6 eV/Å |
| `refine` | 3 次 Nelder–Mead 评估，每次都生成新的 SK 集 |
| 能带链路 | 用 DFTB 在发表点自己造一个 BandRef（加 0.5 eV）：该点损失 0、平移 −0.500，另一点损失 0.012，S3 与 S4 都能算出 |

冒烟测试修掉的问题：

- 名字重复的检查起了作用：冒烟数据本身有重复。
- 本机缺 `threadpoolctl`，线程限制改为可选，进程池在启动前设好 BLAS 线程环境变量。
- 罚项不再导入 `utils.loss`，避免依赖 skopt。
- 单个结构的家族没有相对能（NaN），不能凭空达到 A 级。
- 再次运行 `refine` 时编号不再复用，缓存的 x 不同时会重新计算。

### 5. K1 的本机部分结果

PTBP 发表的 Li、O 半径，hotcent 积分网格 nr 100、ntheta 300。

| 结构 | A 系列设置对收敛参考 | PTBP 默认（k 点密度 5、T = 0）对收敛参考 |
|---|---|---|
| rat_li_bcc16 | −0.66 meV/原子，0.005 eV/Å | +4.04 meV/原子，0.030 eV/Å |
| rat_li2o12 | +0.01 meV/原子，0.006 eV/Å | −0.02 meV/原子，0.002 eV/Å |
| eos_li2o2_05 | 0.00 | 0.00 |
| eos_li_bcc_05 | +0.77 meV/原子 | −4.43 meV/原子 |

- A 系列设置（KSPACING 0.15、Gaussian 0.05 eV）对收敛参考（KSPACING 0.08）全部通过：≤ 0.8 meV/原子、≤ 0.0055 eV/Å。
- PTBP 默认在 Li 金属上差约 4 meV/原子、0.03 eV/Å，占验收线的 40%，所以 A 系列必须用对齐 DFT 的设置。
- 精确力对差分力 7.7e-5 eV/Å，通过。
- (b) SK 表噪声还没跑，交给远程（`slurm/kchecks.sh`）。

### 6. 并行写的脚本（三个分支，各管各的目录）

- **`li_o/` 的数据生成器：**
  - `bands_gpaw.py`：D1，30 个结构；
  - `vasp_bands.py`、`collect_bands.py`；
  - `make_d2.py`：79 个单点，约 37 元；
  - `make_d3.py`：sample、select、write 三个阶段，约 120 元；
  - `README_data.md`。
  - 都编译过；dry run 和假数据上的 select/write/retag 都跑过。
- **`pvk_ink/`：**
  - `prepare_structures.py`：148 个结构，已生成并提交；
  - `orbitals_gpaw.py`、`make_d4.py`（201 个算例，约 51 元）、`collect_d4.py`、`make_d5.py`、`k5_fa_iodide_gap.py`；
  - `reactions_ink.yaml`：28 个反应，都能配平；
  - `scan_ink.yaml`：8 维，40 个对样条；
  - `vasp_ink.py`、`README.md`。
- **文档：**
  - `pydftb/docs/design/A0-accuracy-programme.html`，计划全文加已实现的代码与执行顺序；
  - ROADMAP 加了 A0–A4 一行，M5c/M5d 的后续改为并入 A 系列；
  - index.html 加了 A0 一行。
- **本线程写的：**
  - `li_o/prepare_data.py`、四个 `scan_*.yaml`、`reactions_li*.yaml`；
  - `k1_numerics.py`、`k4_dftb3_u.py`、`k6_per_structure_r0.py`、`k7_mace.py`、`md_check.py`；
  - `slurm/` 模板、`README.md`、本日志。

### 7. 与计划不同之处

- **PTBP 的 loss.py 新选项推迟：** 指 `--energy-reduction`、`--force-weight`、`--energy-offset`、`--band-weight`、`--fixed-r0`。A1 全部通过 `ptbp scan` 在同一批点上计算，PTBP 的 `optimize` 路径一行没改，所以逐字节不变。
- **界面的 k 网格显式写成 4×4×1：** M5c 的记录是 4×4×1，而 KSPACING 规则沿 c 给出 2（2π|b₃|/0.25 = 1.013）。数据里 `info['kpts']` 优先。

### 8. 尚未解决的问题

1. GPAW 26.7 的 `fixed_density` 调用（`bands_gpaw.py`、`orbitals_gpaw.py`）还没实跑。远程先 `--only li_bcc` 试一个。
2. KSPACING 规则需要用一个 OUTCAR 核对一次：`grep "generate k-points"`。
3. D2 里 53 和 54 原子的空位晶胞约 127 核时，是 D2 中最贵的部分。批准时可以先不算，`make_d2.py` 目前还没有 `--only`。
4. 前驱体溶液的离子簇与质子转移路径是手搭的结构。D4 会先优化它们；反应取每个家族最低的结构。
5. 前驱体溶液的 R₂b 要列出出现的全部 40 个原子对。“physrep 打底加少数对的样条修正”可以以后再加。
6. 费用都是粗估，每批先用一个试算校准。

### 9. 下一步（远程）

按 `README.md` 第 3 节的步骤 1–7：D1 → K1(b) → Li 网格（K2）→ Li–O Sobol（K3）→ K4/K6/K7 → 精修、导出、MD → 传回结果，然后写 G1 报告。D2–D5 等用户批准费用后再算。
