# A 系列：DFTB 能做到多准（执行手册）

设计文档：`pydftb/docs/design/A0-accuracy-programme.html`。开发日志：本目录 `WORKLOG.md`。

本目录是 A0 计划全部阶段的执行代码：A1（一套参数能走多远）、A2（随环境变化的电子部分）、A3（排斥势）、A4（Δ-ML），以及 S5 诊断、基线和关口判定。代码已在本机备好并做过小规模冒烟测试，正式计算在远程服务器上做。

- **不提交作业：** 没有任何脚本会自动提交作业。Slurm 模板在 `slurm/`。要花钱的 DFT 批次（D2–D5）只写输入和费用估计，等你批准后再手动 `sbatch`。
- **日志：** 每个命令都写一份完整日志（见第 2 节）。

```
examples/08_accuracy/
  common/paths.py         路径（PTBP_REPO、PTBP_EX07_DATA、PTBP_PEROVSKITE、PTBP_OFFLOAD 可覆盖；钙钛矿项目的位置没有默认，大文件默认放 `large/`）
  common/models.py        模型目录（a-series-model/1）：加载、预测（可续跑）、打分、MD；DFTB 层的选项与 Γ 心 k 点
  common/baselines.py     基线：发表的 PTBP、GFN1-xTB、MACE-MP-0（溶液另加 MACE-OFF/OMOL）
  common/gates.py         关口 G0–G4：读各阶段的结果文件，写 gates.md / gates.json
  li_o/                   Li 与 Li–O：数据、扫描配置、能带参考、K1/K4/K6/K7、MD 检验、D0/D2/D3、S5
  pvk_ink/                钙钛矿前驱体溶液：结构、轨道参考、D4/D5、K5、扫描配置、MD 拓扑检查（见其 README.md）
  a2/                     A2 驱动 a2.py 与配置（Li、Li–O、溶液），见 a2/README.md
  a3/                     A3 驱动 a3_repulsive.py（R1–R4），见 a3/README.md
  a4/                     A4 驱动 delta_ml.py（Δ1、Δ3 与学习曲线），见 a4/README.md
  slurm/                  Slurm 模板与 env.sh（dg83 CPU、dg50 GPU），见 slurm/README.md
  logs/                   Slurm 的 stdout/stderr（#SBATCH -o logs/...；不进 git）
```

核心代码（都有单元测试）：

| 模块 | 作用 |
|---|---|
| `pydftb/bandref.py` | DFT 参考能带的格式（BandRef，每个结构一个 JSON）、特殊点与连线的 k 点、VASP/GPAW/ASE 读取、半芯态计数 |
| `pydftb/bandloss.py` | 能带损失：VBM−3 到 CBM+1 eV 的窗口、自由平移、价带按序号配对、导带单向就近配对、带隙项、两级（特殊点附近、连线）、DOVE 的 𝒜；最小二乘用的残差与雅可比 |
| `pydftb/repfit.py` | 多原子对的 2 体样条排斥势（线性最小二乘，单调、光滑、可带按元素偏移），写 SKF 的 Spline 块：R₂b 对照与 oracle |
| `pydftb/assess.py` | 评测：按角色/家族的能量、力、EOS、二聚体、配平的反应、A/B/C 级、NVE 稳定性 |
| `pydftb/descriptors.py` | A2 的描述符阶梯：配位数（可按邻居元素分开）、径向、角度（G4/G5）、RBF 展开、标准化/PCA、SOAP（dscribe）、MACE 特征；梯度按“中心–邻居”向量对给出，力和应力精确 |
| `pydftb/regressors.py` | 描述符 → 参数：线性、核回归（GPR 式）、小型 MLP |
| `pydftb/envmodel.py` | 环境模型的统一协议（拟合、保存、读取） |
| `pydftb/banddiff.py` | 冻结电荷下的能带本征值导数（Hellmann–Feynman；通用差分与 adaptive 的解析式） |
| `pydftb/adaptive.py`、`adaptive_fit.py` | E3：on-site 平移与跃迁标度随描述符变化；拟合器加了能带目标（`BandSample`、`band_weight`）、按元素或不加偏移（`offsets`）、只拟合能带、Γ 心 k 点 |
| `pydftb/r0mix.py`、`typeswitch.py`、`coulomb.py` | E2：r0 随环境连续变化（r0 节点的 SK 表按三次 B 样条混合）；E1：按类型的 on-site 与 U 平移（`OnsiteSwitch`）；`Coulomb.hubbard_gradient` |
| `utils/skgrid.py` | 约束半径的参数空间与 SK 缓存；`TypedSKCache`：E2 的 r0 节点 SK 表（带类型标签） |
| `utils/accuracy.py`, `cli/cmd_scan.py` | `ptbp scan`：同一批点上算所有策略（S1、S1o、S1eos、S2、S3、S4）、R₂b 与 oracle；`refine --method nm\|bo --seeds N` |
| `utils/loss.py`、`loss_grad.py`、`engines.py` | PTBP 自己的 `optimize` 的新选项（第 8 节），默认与 66a6498 逐位相同 |
| `utils/runlog.py` | 运行日志 |

## 1. 远程环境

```bash
git clone -b claude/inspiring-ride-zax2o7 https://github.com/MengnanCui/PTBPlab.git && cd PTBPlab
conda create -n ptbp python=3.11 numpy scipy ase pyyaml matplotlib threadpoolctl pylibxc scikit-optimize -c conda-forge
conda activate ptbp
pip install --no-build-isolation ./third_party/hotcent   # hotcent 2.1.0 的源码在仓库里，不用下载；PTBP 的分片 hotcent 按 2.1.0 校验（E2 的带类型 SK 表必须用分片驱动）
pip install -e pydftb
```

可选依赖，用到的阶段才需要：

| 包 | 用途 |
|---|---|
| `gpaw` | D1 能带与轨道参考 |
| `torch`（CPU 版即可）、`mace-torch` | K7、A4、S5 的自动微分、A2 的 L4 特征、D3 的 MACE 采样 |
| `dscribe` | A2 的 L3（SOAP） |
| `tblite` | GFN1-xTB 基线 |
| `simple-dftd3`（`dftd3.ase`） | 溶液的 D3(BJ)：只在使用模型时两边都加 |

装好后先检查（本机 2026-11-10 的结果写在后面）：

```bash
python -m pytest -q pydftb/tests/test_bandref.py pydftb/tests/test_bandloss.py pydftb/tests/test_repfit.py \
       pydftb/tests/test_assess.py pydftb/tests/test_descriptors.py pydftb/tests/test_regressors.py \
       pydftb/tests/test_banddiff.py pydftb/tests/test_adaptive_fit_bands.py pydftb/tests/test_r0mix.py \
       pydftb/tests/test_typeswitch_unequal.py pydftb/tests/test_adaptive.py pydftb/tests/test_type_switching.py
python -m pytest -q tests/                                          # PTBP（含 test_loss_options、test_scan_refine、test_skgrid_typed）
python parameterize.py scan --help
```

然后编辑 `slurm/env.sh`：conda 环境名，以及 SK 缓存 `PTBP_SKCACHE`。缓存会很大，放在本地盘或 scratch 盘上。所有模板都从 `examples/08_accuracy/` 提交，并先 `mkdir -p logs`（仓库里已有这个目录）。

## 2. 日志（全程保留）

每个命令都在 `<out>/logs/<名字>/<时间>-<主机>-<pid>/` 下写这些文件：

- `run.log`：stdout、stderr、Python 警告，出错时还有完整 traceback；
- `meta.json`：命令行、参数、工作目录，git 提交号及是否有未提交的改动，主机、CPU 数，Slurm 作业号，线程变量，Python 与各包的版本；
- `events.jsonl`：每一步一行，带耗时；
- `status.json`：ok 或 failed，用时，错误信息。

另外几个位置：
- `ptbp scan` 的日志在 `<scan out>/logs/scan-<stage>/` 下。每个点另有 `points/pNNNNN/result.json`，内容是半径、SK 键、每个结构的 SCC 尝试次数和错误；出错的点还有 `error.txt`。
- A2 的日志在 `<out>/logs/a2-<阶段>-<运行>/`，A3、A4、S5、基线与关口的日志在各自输出目录的 `logs/` 下。
- Slurm 自己的 stdout/stderr 在 `examples/08_accuracy/logs/`。

跟进度用 `tail -f <...>/events.jsonl`。

**需要我分析时，把这些传回来：**

```bash
tar czf a_results.tgz li_o/runs/*/report.* li_o/runs/*/losses.csv li_o/runs/*/maps.png \
    li_o/runs/*/logs li_o/runs/k*/ li_o/bands/*.json li_o/runs/gates li_o/runs/baselines \
    li_o/runs/a2_*/result_*.json li_o/runs/a2_*/summary.md a3/runs/*/result.json a3/runs/*/summary.md \
    a4/runs/*/result.json a4/runs/*/*.md li_o/runs/s5_*/s5_report.* pvk_ink/runs/*/report.* 2>/dev/null
```

`points/` 不必都传。它只有每点几十 KB，用 rsync 同步也很快。

## 3. Li 与 Li–O（A1.1、A1.2）：执行顺序

所有命令在 `examples/08_accuracy/` 下执行。`PTBP="python ../../parameterize.py"`，或者 `cd li_o` 后用 `python ../../../parameterize.py`。

| 步 | 命令 | 产出 | 耗时与费用 | 状态 |
|---|---|---|---|---|
| 0 | `python li_o/prepare_data.py` | `li_o/data/`：li、li_o 两组的 train/test/ood，E0s，manifest | 秒 | 已做，已提交（2026-11-10 改：界面帧不再写 4×4×1，见下） |
| 1 | D1 能带参考：`python li_o/bands_gpaw.py --out li_o --dry-run`，再去掉 `--dry-run` 运行（工作站 GPAW，或 `sbatch slurm/gpaw_bands.sh`） | `li_o/bands/*.json`：15 个晶体在 3 个体积，15 个二聚体；`bands/energies.json` 用于与 VASP 对照 | 数核时，0 元 | 待运行；GPAW 26.7 的调用还没实跑，先用 `--only li_bcc` 试一个 |
| 2 | K1：`python li_o/k1_numerics.py --out li_o/runs/k1` | `k1_summary.md` | 几分钟 | (a)(c) 已在本机跑过，通过（见 WORKLOG）；(b) SK 噪声待跑 |
| 3 | K2 = A1.1，Li 的 12×12 网格：`sbatch slurm/scan_node.sh li_o/scan_li.yaml`，Li 带 p 的版本用 `scan_li_sp.yaml` | `li_o/runs/li_s_grid/{report.md, maps.png, losses.csv}` | 约 1 节点时，每个版本 ≤ 7.6 元 | 待运行 |
| 4 | K3 = A1.2，Li–O 的 Sobol 256 点：`sbatch --array=0-31 slurm/scan_array.sh li_o/scan_li_o.yaml 32`，之后 `sbatch --dependency=afterok:<id> slurm/scan_losses.sh li_o/scan_li_o.yaml`；Li 带 p 用 `scan_li_o_sp.yaml` | `li_o/runs/li_o_s_sobol/report.md`；oracle 一列就是 K3 | 约 10–17 核时，每个版本 1–2 元 | 待运行 |
| 5 | K4、K6：`sbatch slurm/kchecks.sh` | `runs/k4/k4_summary.md`、`runs/k6/k6_summary.md` | < 1 元 | 待运行 |
| 6 | K7：`sbatch slurm/mace_gpu.sh k7` | `runs/k7/k7_summary.md` | 1–3 GPU 时，6–18 元 | 待运行 |
| 7 | 精修、导出、MD：`$PTBP scan refine li_o/scan_li_o.yaml --strategy S1 --method bo --seeds 3`（或默认的 Nelder–Mead）；`$PTBP scan export li_o/scan_li_o.yaml --point I --strategy S1 --out li_o/models/S1`；`python li_o/md_check.py --model li_o/models/S1 --out li_o/runs/md_S1` | `refine_S1_bo.json`、导出的完整 SKF、`md_summary.md` | 精修每次评估约数分钟；MD 约 1 小时 | 待运行 |
| 8 | 基线：`sbatch slurm/baselines.sh li_o/scan_li_o.yaml li_o/runs/baselines` | `baselines_summary.md`：发表的 PTBP、GFN1-xTB、MACE-MP-0 | < 1 节点时，< 8 元 | 待运行 |
| 9 | 关口：`python common/gates.py --root li_o/runs` | `li_o/runs/gates/gates.md`：G0、G1 的每条规则、数字与结论 | 秒 | 待运行 |
| 10 | G1：把 `report.md`、`k*_summary.md`、`md_summary.md`、`gates.md` 传回，我写 A1 报告 | — | — | — |

**要你批准后才提交的 DFT：**

| 批 | 生成命令 | 内容 | 估计 |
|---|---|---|---|
| D0 | `python li_o/d0_ibz_bands.py --help`（只转换已下载的 vasprun/EIGENVAL/OUTCAR；下载命令写在脚本里） | 已有 SCF 的 IBZ 本征值作能带数据 | 0 元，下载要你同意 |
| D2 | `python li_o/make_d2.py --out ~/a1_d2` | hcp EOS、表面、空位、Li₂O 缺陷与表面、Li₂O₂ 随机位移，79 个单点 | 约 370 核时，约 37 元 |
| D3 | `python li_o/make_d3.py sample --out li_o/runs/d3`（GPU），然后 `select`、`write` | 界面帧：MACE-MP-0 采样，最远点选 80 帧 | 约 120 元 |
| VASP 能带（可选，与 GPAW 对照） | `python li_o/vasp_bands.py --out ~/a1_vasp_bands` | 非自洽能带 | 约 13 核时，1.3 元 |

- **算完之后：** 用 `python examples/07_battery_li_li2o/collect.py RUNS --out DIR` 收集，`python li_o/make_d2.py retag ...` 设标签，再 `python li_o/prepare_data.py --d2 DIR/train.xyz`。细节见 `li_o/README_data.md`。
- **k 网格：** `accuracy.kpoints_for` 按 VASP 的 KSPACING 规则取 N = ceil(2π|b|/KSPACING)。已用界面的 3 个 OUTCAR 核对（2026-11-10）：VASP 生成的是 4×4×2，与规则一致；M5c 记录的 4×4×1 是错的，所以数据里不再显式写网格。新结构的网格仍可用 `grep "generate k-points"` 抽查。

## 4. 前驱体溶液（A1.3）

见 `pvk_ink/README.md`。

1. `prepare_structures.py` 生成 148 个结构，已提交在 `pvk_ink/data/`。
2. D1 轨道参考：`orbitals_gpaw.py`，用 GPAW。
3. D4 气相单点、优化与孤立原子：`make_d4.py` 生成 201 个算例，约 508 核时、约 51 元，等批准；算完后用 `collect_d4.py` 收集。
4. K5：`k5_fa_iodide_gap.py`，看 FA⁺ π* 与 I⁻ 5p 的间隙能不能单靠 r0 打开。
5. 扫描：`scan_ink.yaml`。
6. D5 团簇与 M409 帧：`make_d5.py`，需要 MACE 轨迹（钙钛矿项目里的大文件，要先拷到服务器）。
7. MD 拓扑检查：`python pvk_ink/md_topology.py --help`（默认加 PBE-D3(BJ)；报告断键、成键与质子转移）。
8. 色散：参考数据是纯 PBE，拟合时比纯 PBE 与纯 DFTB；使用模型时两边加同一套 PBE-D3(BJ)（模型目录的 `dispersion`），不用 3ob 的 D3 参数。

## 5. A2、A3、A4 与 S5（G1 之后）

G1 决定先走哪条路（规则见第 7 节）。命令都已备好：

| 阶段 | 入口 | 远程模板 | 粗估费用 |
|---|---|---|---|
| A2 电子部分随环境变化（E1/E2/E3 × L1–L4 × 线性/核/MLP） | `a2/a2.py plan/prepare/fit/transfer/summary`，配置 `a2/a2_li.yaml`、`a2_li_o.yaml`、`a2_ink.yaml`；说明见 `a2/README.md` | `slurm/a2_prepare.sh`、`a2_fit.sh`（数组：全量 + 每组留出）、`a2_post.sh` | 每次拟合 0.2–0.8 元；Li–O 一次运行（8 个拟合 + 传递检验）约 3–8 元；E2 的 SK 表约 1 核时 |
| A3 排斥势（R1–R4） | `a3/a3_repulsive.py score/r3/r4/md`；说明见 `a3/README.md` | `slurm/a3_node.sh` | Li–O R3 约 15–23 元（粗估） |
| A4 Δ-ML（Δ1、Δ3、学习曲线 N = 25/50/100/全部） | `a4/delta_ml.py prepare/train/evaluate`；说明见 `a4/README.md` | `slurm/a4_delta_gpu.sh`（dg50） | 2–4 GPU 时，12–24 元（粗估） |
| S5 诊断：直接拟合 SK 表（两中心形式不变） | `li_o/s5_sk_tables.py dry-run/gradcheck/fit/report/export`；说明见 `li_o/README_s5.md` | `slurm/s5.sh` | 一个 dg83 节点，按 dry-run 的估计 |
| 关口 G2–G4 | `python common/gates.py --root li_o/runs --a2 '...' --a3 '...' --a4 '...'` | — | 秒 |

本机冒烟测试（2026-11-10，小数据，只验证代码）：
- A2：E3、E1、E2 的全量拟合、留一组拟合、传递检验、汇总与 G2 都跑通；E2 的 5 节点 SK 表 hotcent 用了 136 s。
- A3：R3、R4（带能带正则，`--offsets none`）都跑通。
- A4：用 2 轮训练的 MACE 跑通全程，Δ 模型目录与 DFTB+MACE 的和相差 1e-14 eV。
- S5：自动微分的梯度与差分相差 ≤ 5e-9。

## 6. PTBP 自己的 optimize：新选项

`ptbp scan` 仍是 A1 的主路径。PTBP 原有的 `ptbp optimize` 新增了以下选项，全部取默认值时与 66a6498 逐位相同（`tests/test_loss_options.py` 把 66a6498 的代码与现在的代码放在一起比较，另有端到端核对）：

- dataset/reaction 损失：`--energy-reduction sum|mean|group`、`--force-weight`（默认 0.1）、`--energy-offset none|per-element`；
- 能带：`--band-weight W --band-refs GLOB`、`--mode bands`（旧的 `bandstructure` 不动）；
- `--fixed-r0`（只拟合 σ，即 S3 的路线）、`--sigma-finish`；
- 对齐 DFT：`--kspacing`、`--smearing-width`、`--energy-kind sigma0`、`--third-order`、`--hubbard`。

## 7. 关口（事先定好，见设计文档第三到第七节；`common/gates.py` 自动判定）

- **G0：** K1 通过（k 网格与展宽 ≤ 2 meV/原子、≤ 0.01 eV/Å；SK 噪声 ≤ 0.02 eV/Å；精确力）。
- **G1（A1 之后）：**
  - 有策略在家族外测试上达到 A 级且 MD 稳定：这个体系用一套参数就够。
  - oracle 在 B 级以内，而 physrep 各行差得多：瓶颈是排斥势的形式，先做 A3。
  - oracle 差于 C 级，或界面符号在所有点上都错：单一参数集做不了断键。
  - Spearman(能带, R₂b) < 0.3：能带只当正则项。
  - K7 的 MACE 也达不到 A 级：限制在数据，先补数据。
- **G2（A2）：**
  - 留一环境的能带误差降 ≥ 30%，且 R₂b 降 ≥ 20%：保留这个电子模型。
  - 能带变好但 R₂b 降 < 10%：环境相关性应放进排斥势（A3）。
  - 训练能带误差比逐结构最优（K6）高 ≥ 20%，且训练–测试差 < 30%：才升到下一级描述符。
- **G3（A3）：** 达到 A 级且 MD 稳定就结束，否则进入 A4。
- **G4（A4）：** 在 N ≤ 100 或家族外测试上，Δ-ML 要比同样大小的纯 MACE 好 ≥ 30%。
- **验收线（A 级，用户 2026-11-09 定）：** 测试集上能量差 MAE ≤ 10 meV/原子、力 MAE ≤ 0.1 eV/Å，决定性的能量差符号全对，300–600 K MD 稳定。能量按配平的能量差算，绝对原子化能只作诊断。

## 8. 常见问题

- **SCC 不收敛：** 每个结构先用默认混合，失败后依次改用更慢的 Broyden 和 Anderson。仍失败就记录在 `result.json` 的 `errors`（A2/A3/A4 是 `predict.jsonl` 的 `error`）里，这个点的损失为 NaN，扫描继续。
- **中断后续跑：** 重跑同一个命令即可，已完成的部分会跳过，`--force` 重算。SK 缓存按内容寻址，键包括半径、p、xc、网格和 hotcent 版本，可以跨扫描共享。
- **Slurm 找不到 env.sh：** `sbatch` 运行的是脚本的暂存副本。模板会依次在 `$SLURM_SUBMIT_DIR/slurm`、`$SLURM_SUBMIT_DIR`、`$PTBP_REPO/examples/08_accuracy/slurm` 里找，也可以用 `PTBP_ENV_SH` 直接指定。所以要从 `examples/08_accuracy/` 提交。
- **大文件：** 超过 20 MB 的轨迹与缓存不进仓库，放到 `PTBP_OFFLOAD`（默认 `large/`，git 忽略）或服务器本地盘。
- **hotcent 版本：** 分片驱动校验 hotcent 2.1.0 的源码哈希，其他版本自动走参考实现，结果相同，只是慢。例外是 E2 的带类型 SK 表，它必须用分片驱动。加密积分网格（`sk.grid`）需要 hotcent ≥ 2。
