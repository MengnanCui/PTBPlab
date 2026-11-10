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
- **界面的 k 网格显式写成 4×4×1：** M5c 的记录是 4×4×1，而 KSPACING 规则沿 c 给出 2（2π|b₃|/0.25 = 1.013）。数据里 `info['kpts']` 优先。（这条后来证明是错的，已改正，见下一轮。）

### 8. 尚未解决的问题

1. GPAW 26.7 的 `fixed_density` 调用（`bands_gpaw.py`、`orbitals_gpaw.py`）还没实跑。远程先 `--only li_bcc` 试一个。
2. KSPACING 规则需要用一个 OUTCAR 核对一次：`grep "generate k-points"`。（已解决，见下一轮：界面的 OUTCAR 是 4×4×2，与规则一致。）
3. D2 里 53 和 54 原子的空位晶胞约 127 核时，是 D2 中最贵的部分。批准时可以先不算，`make_d2.py` 目前还没有 `--only`。
4. 前驱体溶液的离子簇与质子转移路径是手搭的结构。D4 会先优化它们；反应取每个家族最低的结构。
5. 前驱体溶液的 R₂b 要列出出现的全部 40 个原子对。“physrep 打底加少数对的样条修正”可以以后再加。
6. 费用都是粗估，每批先用一个试算校准。

### 8b. 与远程分支合并

push 时远程分支已经多了其他会话合并进来的提交，最新是 5b03338，内容有：
- 桌面 GUI（`ptbp gui`）与 `--seed`；
- `bandstructure` 模式修好（cd3cbdf）；
- scikit-optimize 0.10 的 BO 后处理修复；
- 失败的 BO 步排在最后；
- pydftb M6f-1b：最后一轮 SCC 之后不再混合；
- 两原子距离小于 0.01 Bohr 时停止；
- hotcent 参考路径不再吞掉错误。

本轮把它们合并进来，没有强推。冲突只在两个文件，都是双方各加了一部分，所以两边都保留：
- `cli/config.py`：本轮的 A 系列配置项与远程的 `seed`。
- `cli/run.py`：BO 作图改用远程的 `save_bo_plots`，`--fixed-r0` 的结果仍不画；先做 `--sigma-finish`，再执行远程改写的备份与打包（shutil/tarfile）。

前面说的“与 66a6498 逐位相同”，比较的是本轮改动前后的代码。合并之后，基准变成合并后的主线：远程的有些改动本来就会改变结果，例如 M6f-1b。本轮的改动在默认设置下仍不改变任何结果。

合并后的测试：
- **pydftb：** 23 个测试文件，269 项通过，5 项因缺 dscribe、mace、torch 跳过。
- **PTBP `tests/`：** 331 项通过，63 项跳过，5 项失败。这 5 项在远程提交 5b03338 上单独跑也同样失败，那里是 294 项通过；多出的通过项是本轮新增的测试。5 项中：
  - 4 项因为缺 torch；
  - 1 项是远程新加的 `test_logging_setup.py::test_install_handler_swallows_benign_logs_real`。它单独跑也失败：错误记录打印出来了，但 caplog 没有捕获到，问题在 GUI 日志的设置，与本轮无关。

### 9. 下一步（远程）

按 `README.md` 第 3 节的步骤 1–7：D1 → K1(b) → Li 网格（K2）→ Li–O Sobol（K3）→ K4/K6/K7 → 精修、导出、MD → 传回结果，然后写 G1 报告。D2–D5 等用户批准费用后再算。

## 2026-11-10 代码准备（二）：A2–A4 与其余部分

用户要求：“继续跑完补全剩下的计算部分，然后 push（不做计算哦）”。即写完计划里剩下的全部代码，只做小规模冒烟测试，不提交作业，不做正式计算。

### 1. 做法

- 先写了三个共用接口并测过：
  - `pydftb/descriptors.py`：`Features` 带“中心–邻居”向量对的梯度，`Features.chain` 给出力和维里；
  - `pydftb/regressors.py`：线性回归器；
  - `pydftb/envmodel.py`：环境模型的统一协议，含 `engine_kwargs`、`model_gradient` 和保存/读取。
- 然后分五个分支并行写，每个分支只改自己的文件。这一轮的 A2 驱动、整合、文档和测试由主线程写。

### 2. pydftb（A2 引擎）

| 部分 | 内容 | 测试 |
|---|---|---|
| 描述符 | `Radial`、`Angular`（G4/G5）、`Expand`、`Affine.standardize`/`pca`、`SOAP`（dscribe）、`MACEFeatures`（mace-torch，无导数）；可按邻居元素分开 | `test_descriptors.py`：22 项通过，2 项因缺 dscribe/mace 跳过 |
| 回归器 | `Kernel`（GPR 式，对系数线性，`from_features` 用最远点取诱导点）、`MLP`（tanh，解析导数） | `test_regressors.py`：17 项 |
| 能带导数 | `bands.eigenstates(vectors=True)`；`banddiff`：冻结电荷下的 H、S，通用的 Hellmann–Feynman 差分（简并子空间单独处理），adaptive 的解析式 | `test_banddiff.py`：8 项 |
| E3 | `AdaptiveParams(descriptor=, regressor=, channels=)`：力与应力经 `Features.chain`；`AdaptiveResult.dedp` 让非线性回归器也有 dF/dθ | `test_adaptive.py`：29 项，新增 6 项 |
| 拟合器 | `fit_adaptive`：`BandSample`、`band_weight`、`band_settings`、`offsets=global\|none\|per_element`、`band_jacobian=frozen\|scc`；可只拟合能带（`samples=[]`）；k 点接受 Γ 心网格；其他模型的能量梯度可用时取解析值，否则用差分 | `test_adaptive_fit_bands.py`：6 项 |
| E2 | `r0mix.R0Switch`：r0 节点之间三次 B 样条（C²）混合，`around()` 以 A1 点为中心建节点；力、应力、dF/dθ 精确 | `test_r0mix.py`：10 项 |
| E1 | `typeswitch.OnsiteSwitch`：按类型的 on-site 与 U 平移，可拟合；`Coulomb.hubbard_gradient`（E3 以后的 U 通道也可以用） | `test_typeswitch_unequal.py`：12 项 |
| 其他 | `active.write_candidates(writer=)`；`active.py` 的 threadpoolctl 改为可选（原来在没有它的机器上 2 项测试失败） | `test_active.py`：7 项 |

- **与差分的一致性：** 力、应力、dF/dθ、dF/d(平移) 与差分一致到 2e-9 以内，应力到 5e-8 以内。
- **逐位相同：** 以下与 66a6498 的引擎逐位相同：
  - `eigenstates` 的重构；
  - CN 模式的 adaptive；
  - 默认参数的 `fit_adaptive`；
  - 普通 SCC、M5a、M5a + DFTB3 + 自旋、M5a + DFTB+U、周期体系的应力。
- **冻结电荷的能带雅可比不含电荷响应：** 去掉平移后，水分子差约 21%，Cu 金属差约 0.5%。需要精确值时用 `band_jacobian="scc"`。

### 3. PTBP

- **`ptbp optimize` 的新选项**（A0 第八节，上一轮推迟的）：
  - 能量与力：`--energy-reduction`、`--force-weight`、`--energy-offset`；
  - 能带：`--band-weight` 加 `--band-refs`，新的 `--mode bands`；
  - `--fixed-r0`、`--sigma-finish`；
  - 对齐 DFT：`--kspacing`、`--smearing-width`、`--energy-kind sigma0`、`--third-order`、`--hubbard`。
  - 默认时与 66a6498 逐位相同。`tests/test_loss_options.py`（31 项）把 66a6498 的 `loss.py`/`loss_grad.py` 与现在的代码放在一起比较；另用 `tools/engine_parity` 做了端到端核对，μ 与写出的文件都逐字节相同。
- **`ptbp scan refine --method bo --seeds 3`：** 用 scikit-optimize，可选依赖；Nelder–Mead 仍是默认。测试 `tests/test_scan_refine.py`。
- **`utils/skgrid.TypedSKCache`：** E2 的 r0 节点 SK 表，按标签对写出全部文件，同元素不同标签用异核格式，`plan()` 给出费用。测试 `tests/test_skgrid_typed.py`，含一次 2 个标签的真实 hotcent 运行，约 25 s。

### 4. `examples/08_accuracy/`

- **`a2/a2.py`（主线程写）：** 阶段有 `plan`、`prepare`、`fit`、`transfer`、`summary`、`status`，配置是 `a2_li.yaml`、`a2_li_o.yaml`、`a2_ink.yaml`。
  - 拟合只用能带，没有排斥势。
  - 每次运行都做一次全量拟合，再让每个能带组轮流留出一次。
  - 传递检验：把 A2 模型在全部结构上的电子能量与力写成一个“伪扫描点”，再用扫描本身的 `score_point` 打分（physrep σ、R₂b、oracle）。这个做法先核对过：把一个扫描点复制到别处重新打分，损失逐位相同。
  - 输出 `a2-result/1` 文件，交给 G2。
  - Slurm 模板：`a2_prepare.sh`、`a2_fit.sh`（数组）、`a2_post.sh`。
- **其他分支写的：**
  - `common/models.py`：模型目录 `a-series-model/1`，含 D3(BJ) 选项与 DFTB 层的选项；
  - `common/baselines.py`：发表的 PTBP、GFN1-xTB、MACE-MP-0/OFF/OMOL；
  - `common/gates.py`：G0–G4；
  - `a3/a3_repulsive.py`：R1–R4，加 MD；
  - `a4/delta_ml.py`：Δ1、Δ3 与学习曲线；
  - `pvk_ink/md_topology.py`；
  - `li_o/d0_ibz_bands.py`；
  - `li_o/s5_sk_tables.py`：S5，梯度用 pydftb.autodiff 的 implicit 模式，需要 torch；
  - Slurm 模板 `a3_node.sh`、`a4_delta_gpu.sh`、`baselines.sh`、`s5.sh`。

### 5. 本轮发现并改正的问题

- **界面的 k 网格：** 3 个界面 OUTCAR 的 KSPACING 都是 0.25，生成的网格都是 “4 4 2”（20 个 k 点），与 `kpoints_for` 的规则一致。M5c 记的 4×4×1 是错的。
  - `prepare_data.py` 不再写 `kpts`，`li_o_ood.xyz` 已重新生成，只有这一个文件变了。
  - 同时改了 `make_iface_dft.py` 和 `k4_dftb3_u.py` 的注释。
  - M5c 报告里加了一条勘误。
- **Slurm 模板：** `sbatch` 运行的是脚本的暂存副本，所以上一轮的 6 个模板里 `source "$(dirname "$0")/env.sh"` 在作业里找不到文件。现在所有模板都依次在 `$SLURM_SUBMIT_DIR/slurm`、`$SLURM_SUBMIT_DIR`、`$PTBP_REPO/examples/08_accuracy/slurm` 里找，也可以用 `PTBP_ENV_SH` 直接指定。
  - 另外加了 `logs/.gitkeep`：`#SBATCH -o logs/...` 指向的目录必须已经存在，Slurm 不会自己创建。
- **远程环境的依赖：** 补上 `pylibxc`、`scikit-optimize`。可选依赖见 README 第 1 节。
- **只拟合能带时的进度行：** 原来打印 `rmse nan meV/atom`，现在不打印这一项。

### 6. 冒烟测试（本机，只验证代码）

- **A2：** 数据是上一轮的 Li 冒烟扫描。参考能带用 DFTB 在另一个 r0 点造（bcc、fcc、Li₂），从发表点出发去拟合。
  - E3 L1 linear：只拟合 bcc 与 Li₂ 两组，能带损失从 26 meV 降到 0.02 meV。
  - 留一组拟合也跑通。E1 留出的组没有拟合过自己的类型，所以它的误差等于起点，符合预期。
  - E2 的 5 节点 SK 表 hotcent 用 136 s。E2 只沿 r0_w 移动，而目标点主要差在 r0_d，所以降不到 0（19 meV），也符合预期。
  - E3 L2 kernel/mlp、传递检验（9 个结构，无失败）、汇总和 G2 判定都跑通。
- **A3：** R3（`--offsets none`）、R4（带能带正则）都跑通。
- **其他分支报告：**
  - A4 用 2 轮训练的 MACE 跑通全程；
  - 基线：PTBP、xTB、MACE-MP 加 D3；
  - 关口在 K1 结果和假 A2–A4 文件上跑通；
  - `md_topology.py` 在发表的 PTBP 下 20 fs 内抓到 FA⁺·I⁻ 的 N→I 质子转移，与 M409 上看到的一致；
  - S5：梯度与差分相差 ≤ 5e-9，导出的表与内存中的修正相差 1.8e-15 eV。

### 7. 测试（2026-11-10，本机）

- **pydftb：** 新增与受影响的 21 个测试文件，228 项通过，3 项因缺 dscribe/mace/torch 跳过。另有 2 项失败，原因是缺 threadpoolctl，改了 `active.py` 之后 `test_active.py` 7 项全过。
- **PTBP `tests/`：** 在有 skopt 的环境里，271 项通过，60 项跳过，4 项失败。这 4 项都因为缺 torch，66a6498 上也是这 4 项。

### 8. 尚未解决的问题

1. 冻结电荷的能带雅可比是近似值，拟合不收敛时改用 `band_jacobian: scc`。
2. E2 的 B 样条不插值：θ = 0 时不完全等于 A1 点，所以节点间距要小。
3. 引擎仍拒绝 E2 与 E3 同时使用，等 G2 显示两者各自有用时再做。
4. L4（MACE 特征）没有导数，只能拟合能带；周期体系的 SOAP 没有应力。
5. 本机没有 dscribe 和 mace，SOAP 与 MACE 特征的封装还没实跑。
6. E3 的 U 平移通道还没做。`Coulomb.hubbard_gradient` 已经可以用。
7. `TypedSKCache` 不复用 A1 缓存里相同的表。
8. D0 的下载、D2–D5 的计算都要你批准。费用都是粗估。
9. S5 远程需要 torch，CPU 版即可。

### 8b. 与远程分支合并

push 时远程分支已经多了其他会话合并进来的提交，最新是 5b03338，内容有：
- 桌面 GUI（`ptbp gui`）与 `--seed`；
- `bandstructure` 模式修好（cd3cbdf）；
- scikit-optimize 0.10 的 BO 后处理修复；
- 失败的 BO 步排在最后；
- pydftb M6f-1b：最后一轮 SCC 之后不再混合；
- 两原子距离小于 0.01 Bohr 时停止；
- hotcent 参考路径不再吞掉错误。

本轮把它们合并进来，没有强推。冲突只在两个文件，都是双方各加了一部分，所以两边都保留：
- `cli/config.py`：本轮的 A 系列配置项与远程的 `seed`。
- `cli/run.py`：BO 作图改用远程的 `save_bo_plots`，`--fixed-r0` 的结果仍不画；先做 `--sigma-finish`，再执行远程改写的备份与打包（shutil/tarfile）。

前面说的“与 66a6498 逐位相同”，比较的是本轮改动前后的代码。合并之后，基准变成合并后的主线：远程的有些改动本来就会改变结果，例如 M6f-1b。本轮的改动在默认设置下仍不改变任何结果。

合并后的测试：
- **pydftb：** 23 个测试文件，269 项通过，5 项因缺 dscribe、mace、torch 跳过。
- **PTBP `tests/`：** 331 项通过，63 项跳过，5 项失败。这 5 项在远程提交 5b03338 上单独跑也同样失败，那里是 294 项通过；多出的通过项是本轮新增的测试。5 项中：
  - 4 项因为缺 torch；
  - 1 项是远程新加的 `test_logging_setup.py::test_install_handler_swallows_benign_logs_real`。它单独跑也失败：错误记录打印出来了，但 caplog 没有捕获到，问题在 GUI 日志的设置，与本轮无关。

### 9. 下一步（远程）

1. 先按 README 第 3 节完成 A1，跑完基线和 `common/gates.py`，传回结果写 G1 报告。
2. 再按 G1 的结论走 A2（`a2/README.md`）或 A3（`a3/README.md`）。
3. A4 和 S5 在需要时做。
