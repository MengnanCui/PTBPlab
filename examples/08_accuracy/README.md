# A 系列：DFTB 能做到多准（执行手册）

设计文档：`pydftb/docs/design/A0-accuracy-programme.html`。开发日志：本目录 `WORKLOG.md`。

本目录是计划第一阶段（A1：一套参数能走多远）的全部执行代码。代码已在本机备好并做过小规模冒烟测试，正式计算在远程服务器上做。

- **不提交作业：** 没有任何脚本会自动提交作业。Slurm 模板在 `slurm/`。要花钱的 DFT 批次（D2–D5）只写输入和费用估计，等你批准后再手动 `sbatch`。
- **日志：** 每个命令都写一份完整日志（见第 2 节）。

```
examples/08_accuracy/
  common/paths.py         路径（PTBP_REPO、PTBP_EX07_DATA、PTBP_PEROVSKITE、PTBP_OFFLOAD 可覆盖；钙钛矿项目的位置没有默认，大文件默认放 `large/`）
  li_o/                   Li 与 Li–O：数据、扫描配置、能带参考、K1/K4/K6/K7、MD 检验、D2/D3 输入
  pvk_ink/                钙钛矿前驱体溶液：结构、轨道参考、D4/D5 输入、K5、扫描配置（见其 README.md）
  slurm/                  Slurm 模板与 env.sh（dg83 CPU、dg50 GPU）
```

核心代码（都有单元测试）：

| 模块 | 作用 |
|---|---|
| `pydftb/bandref.py` | DFT 参考能带的格式（BandRef，每个结构一个 JSON）、特殊点与连线的 k 点、VASP/GPAW/ASE 读取、半芯态计数 |
| `pydftb/bandloss.py` | 能带损失：VBM−3 到 CBM+1 eV 的窗口、自由平移、价带按序号配对、导带单向就近配对、带隙项、两级（特殊点附近、连线）、DOVE 的 𝒜 |
| `pydftb/repfit.py` | 多原子对的 2 体样条排斥势（线性最小二乘，单调、光滑、可带按元素偏移），写 SKF 的 Spline 块：R₂b 对照与 oracle |
| `pydftb/assess.py` | 评测：按角色/家族的能量（原子化能只作诊断、组内相对能、按元素偏移后）、力、EOS、二聚体、配平的反应、A/B/C 级、NVE 稳定性 |
| `utils/skgrid.py` | 约束半径的参数空间（网格、Sobol）与 SK 缓存（加锁，可多任务共享） |
| `utils/accuracy.py`, `cli/cmd_scan.py` | `ptbp scan`：同一批点上算所有策略（S1、S1o、S1eos、S2、S3、S4），加 R₂b 与 oracle |
| `utils/runlog.py` | 运行日志 |
| `utils/hotcent_parallel.set_grid` | hotcent 积分网格（默认为空，与原来逐字节相同） |

## 1. 远程环境

```bash
git clone -b claude/inspiring-ride-zax2o7 https://github.com/MengnanCui/PTBPlab.git && cd PTBPlab
conda create -n ptbp python=3.11 numpy scipy ase pyyaml matplotlib threadpoolctl -c conda-forge
conda activate ptbp
pip install hotcent==2.1.0          # PTBP 的分片 hotcent 按 2.1.0 校验；其他版本走参考实现（较慢）
pip install -e pydftb
# 可选：D1 用 GPAW；D3 与 K7 用 mace-torch（GPU）；钙钛矿的 D3 扣除用 simple-dftd3 或 dftd3
```

装好后先检查：

```bash
python -m pytest -q pydftb/tests/test_bandref.py pydftb/tests/test_bandloss.py \
       pydftb/tests/test_repfit.py pydftb/tests/test_assess.py            # 24 passed
python -m pytest -q tests/test_cli_subcommands.py                          # 11 passed
python parameterize.py scan --help
```

然后编辑 `slurm/env.sh`：conda 环境名，以及 SK 缓存 `PTBP_SKCACHE`。缓存会很大，放在本地盘或 scratch 盘上。

## 2. 日志（全程保留）

每个命令都在 `<out>/logs/<名字>/<时间>-<主机>-<pid>/` 下写这些文件：

- `run.log`：stdout、stderr、Python 警告，出错时还有完整 traceback；
- `meta.json`：命令行、参数、工作目录，git 提交号及是否有未提交的改动，主机、CPU 数，Slurm 作业号，线程变量，Python 与各包的版本；
- `events.jsonl`：每一步一行，带耗时；
- `status.json`：ok 或 failed，用时，错误信息。

`ptbp scan` 的日志在 `<scan out>/logs/scan-<stage>/` 下。每个点另有 `points/pNNNNN/result.json`，内容是半径、SK 键、每个结构的 SCC 尝试次数和错误；出错的点还有 `error.txt`。

跟进度用 `tail -f <...>/events.jsonl`。

**需要我分析时，把这些传回来：**

```bash
tar czf a1_results.tgz li_o/runs/*/report.* li_o/runs/*/losses.csv li_o/runs/*/maps.png \
    li_o/runs/*/logs li_o/runs/k*/ li_o/bands/*.json pvk_ink/runs/*/report.* 2>/dev/null
```

`points/` 不必都传。它只有每点几十 KB，用 rsync 同步也很快。

## 3. Li 与 Li–O（A1.1、A1.2）：执行顺序

所有命令在 `examples/08_accuracy/` 下执行。`PTBP="python ../../parameterize.py"`，或者 `cd li_o` 后用 `python ../../../parameterize.py`。

| 步 | 命令 | 产出 | 耗时与费用 | 状态 |
|---|---|---|---|---|
| 0 | `python li_o/prepare_data.py` | `li_o/data/`：li、li_o 两组的 train/test/ood，E0s，manifest | 秒 | 已做，已提交 |
| 1 | D1 能带参考：`python li_o/bands_gpaw.py --out li_o --dry-run`，再去掉 `--dry-run` 运行（工作站 GPAW，或 `sbatch slurm/gpaw_bands.sh`） | `li_o/bands/*.json`：15 个晶体在 3 个体积，15 个二聚体；`bands/energies.json` 用于与 VASP 对照 | 数核时，0 元 | 待运行；GPAW 26.7 的调用还没实跑，先用 `--only li_bcc` 试一个 |
| 2 | K1：`python li_o/k1_numerics.py --out li_o/runs/k1` | `k1_summary.md` | 几分钟 | (a)(c) 已在本机跑过，通过（见 WORKLOG）；(b) SK 噪声待跑 |
| 3 | K2 = A1.1，Li 的 12×12 网格：`sbatch slurm/scan_node.sh li_o/scan_li.yaml`，Li 带 p 的版本用 `scan_li_sp.yaml` | `li_o/runs/li_s_grid/{report.md, maps.png, losses.csv}` | 约 1 节点时，每个版本 ≤ 7.6 元 | 待运行 |
| 4 | K3 = A1.2，Li–O 的 Sobol 256 点：`sbatch --array=0-31 slurm/scan_array.sh li_o/scan_li_o.yaml 32`，之后 `sbatch --dependency=afterok:<id> slurm/scan_losses.sh li_o/scan_li_o.yaml`；Li 带 p 用 `scan_li_o_sp.yaml` | `li_o/runs/li_o_s_sobol/report.md`；oracle 一列就是 K3 | 约 10–17 核时，每个版本 1–2 元 | 待运行 |
| 5 | K4、K6：`sbatch slurm/kchecks.sh` | `runs/k4/k4_summary.md`、`runs/k6/k6_summary.md` | < 1 元 | 待运行 |
| 6 | K7：`sbatch slurm/mace_gpu.sh k7` | `runs/k7/k7_summary.md` | 1–3 GPU 时，6–18 元 | 待运行 |
| 7 | 精修、导出、MD：`$PTBP scan refine li_o/scan_li_o.yaml --strategy S1`；`$PTBP scan export li_o/scan_li_o.yaml --point I --strategy S1 --out li_o/models/S1`；`python li_o/md_check.py --model li_o/models/S1 --out li_o/runs/md_S1` | `refine_S1.json`、导出的完整 SKF、`md_summary.md` | 精修每次评估约数分钟；MD 约 1 小时 | 待运行 |
| 8 | G1：把 `report.md`、`k*_summary.md`、`md_summary.md` 传回，我写 A1 报告与关口结论 | — | — | — |

**要你批准后才提交的 DFT：**

| 批 | 生成命令 | 内容 | 估计 |
|---|---|---|---|
| D2 | `python li_o/make_d2.py --out ~/a1_d2` | hcp EOS、表面、空位、Li₂O 缺陷与表面、Li₂O₂ 随机位移，79 个单点 | 约 370 核时，约 37 元 |
| D3 | `python li_o/make_d3.py sample --out li_o/runs/d3`（GPU），然后 `select`、`write` | 界面帧：MACE-MP-0 采样，最远点选 80 帧 | 约 120 元 |
| VASP 能带（可选，与 GPAW 对照） | `python li_o/vasp_bands.py --out ~/a1_vasp_bands` | 非自洽能带 | 约 13 核时，1.3 元 |

- **算完之后：** 用 `python examples/07_battery_li_li2o/collect.py RUNS --out DIR` 收集，`python li_o/make_d2.py retag ...` 设标签，再 `python li_o/prepare_data.py --d2 DIR/train.xyz`。细节见 `li_o/README_data.md`。
- **核对 k 网格：** 新结构的 k 网格要和 DFT 一致。`accuracy.kpoints_for` 按 VASP 的 KSPACING 规则取 N = ceil(2π|b|/KSPACING)。界面的 M5c 记录是 4×4×1，而规则给出 4×4×2，所以界面帧在数据里显式写了 `kpts: 4x4x1`。有 OUTCAR 时用 `grep "generate k-points"` 核对一次。

## 4. 前驱体溶液（A1.3）

见 `pvk_ink/README.md`。

1. `prepare_structures.py` 生成 148 个结构，已提交在 `pvk_ink/data/`。
2. D1 轨道参考：`orbitals_gpaw.py`，用 GPAW。
3. D4 气相单点、优化与孤立原子：`make_d4.py` 生成 201 个算例，约 508 核时、约 51 元，等批准；算完后用 `collect_d4.py` 收集。
4. K5：`k5_fa_iodide_gap.py`，看 FA⁺ π* 与 I⁻ 5p 的间隙能不能单靠 r0 打开。
5. 扫描：`scan_ink.yaml`。
6. D5 团簇与 M409 帧：`make_d5.py`，需要 MACE 轨迹（钙钛矿项目里的大文件，要先拷到服务器）。

## 5. 关口（事先定好，见设计文档第三、四节）

- **G0：** K1 通过（k 网格与展宽 ≤ 2 meV/原子、≤ 0.01 eV/Å；SK 噪声 ≤ 0.02 eV/Å；精确力）。
- **G1（A1 之后）：**
  - 有策略在家族外测试上达到 A 级且 MD 稳定：这个体系用一套参数就够。
  - oracle 在 B 级以内，而 physrep 各行差得多：瓶颈是排斥势的形式。
  - oracle 差于 C 级，或界面符号在所有点上都错：单一参数集做不了断键。
  - Spearman(能带, R₂b) < 0.3：能带只当正则项。
- **验收线（A 级，用户 2026-11-09 定）：** 测试集上能量差 MAE ≤ 10 meV/原子、力 MAE ≤ 0.1 eV/Å，决定性的能量差符号全对，300–600 K MD 稳定。能量按配平的能量差算，绝对原子化能只作诊断。

## 6. 常见问题

- **SCC 不收敛：** 每个结构先用默认混合，失败后依次改用更慢的 Broyden 和 Anderson，仍失败就记录在 `result.json` 的 `errors` 里，这个点的损失为 NaN。扫描继续进行。
- **中断后续跑：** 重跑同一个命令即可，已完成的点会跳过；`--force` 重算。SK 缓存按内容寻址（半径、p、xc、网格、hotcent 版本），可以跨扫描共享。
- **大文件：** 超过 20 MB 的轨迹与缓存不进仓库，放到 `PTBP_OFFLOAD`（默认 `large/`，git 忽略）或服务器本地盘。
- **hotcent 版本：** 分片驱动校验 hotcent 2.1.0 的源码哈希；其他版本自动走参考实现，结果相同，只是慢。加密积分网格（`sk.grid`）需要 hotcent ≥ 2。
