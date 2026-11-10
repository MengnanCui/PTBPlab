# A2：随环境变化的电子部分（执行说明）

设计见 `pydftb/docs/design/A0-accuracy-programme.html` 第四节。关口 G2 由 `common/gates.py` 判定，阈值在 A0 里事先定好。所有命令都在 `examples/08_accuracy/` 下执行。

A2 从 A1 扫描的一个点出发（默认是能带损失最好的 S3 点），只让电子部分随环境变化：
- **拟合时只拟合能带**，没有排斥势。这是计划的“先从能带做起，忽略 rep”。
- **每拟合完一个模型，做一次传递检验**：在 A1 的全部结构上算这个电子部分，再重拟合 physrep 的 σ 和 R₂b 样条（与 A1 扫描完全相同的打分）。这样能看出“能带变好”是否带来“能量和力变好”。

## 通道

| 通道 | 做法 | pydftb |
|---|---|---|
| E3 | on-site 平移、按壳层的跃迁标度，作为描述符的函数（M8a 的推广） | `adaptive.AdaptiveParams(descriptor=, regressor=, channels=)` |
| E2 | 每个原子的约束半径 r0 随环境变化：在 A1 点附近的一排 r0 节点之间按 C² 三次 B 样条混合 SK 表（DOVE 的思路，但带力） | `r0mix.R0Switch` |
| E1 | M5a 的 CN 平滑切换，两个类型拷贝 A1 的表，按类型平移 on-site（可选 U） | `typeswitch.OnsiteSwitch` |

## 描述符阶梯与回归器

| 级 | 描述符（`pydftb.descriptors`） | 导数 |
|---|---|---|
| L1 | `cn`：配位数，可按邻居元素分开（`split`） | 有，力和应力精确 |
| L2 | L1 + `radial`（每种邻居几个高斯）+ `angular`（G4/G5） | 有 |
| L3 | `soap`（dscribe，可选），再做 PCA | 视 dscribe 而定，周期体系没有应力 |
| L4 | `mace`：MACE 的不变节点特征（mace-torch，可选），再做 PCA | 没有：只能拟合能带，不能算力、跑 MD |

- **变换：** 描述符先标准化（`transform: standardize`），也可以做 PCA（`transform: pca`）。两者都在拟合用的结构上拟合。
- **回归器：**
  - `linear`：线性模型；
  - `kernel`：GPR 式的核回归，诱导点用最远点采样，对系数是线性的；
  - `mlp`：小型神经网络，tanh 激活。
- **正则：** 岭回归 `fit.l2` 把系数拉向 0，即拉回 A1 点本身。

## 命令

```bash
python a2/a2.py plan     a2/a2_li_o.yaml                         # 列出各次运行、能带组、参数个数、E2 的 SK 费用
python a2/a2.py prepare  a2/a2_li_o.yaml [--run R] [--dry-run]   # E1：类型拷贝（秒级）；E2：带类型的 SK 表（hotcent）
python a2/a2.py fit      a2/a2_li_o.yaml --run E3-L1-linear --holdout all   # 全量拟合 + 每个能带组轮流留出
python a2/a2.py transfer a2/a2_li_o.yaml --run E3-L1-linear --jobs 32       # 传递检验
python a2/a2.py summary  a2/a2_li_o.yaml                         # result_<通道>.json（a2-result/1）+ summary.md
python a2/a2.py status   a2/a2_li_o.yaml
python common/gates.py --root li_o/runs --a2 'li_o/runs/a2_li_o/result_*.json'   # G2
```

远程用 Slurm（先 `mkdir -p logs`）：

```bash
sbatch slurm/a2_prepare.sh a2/a2_li_o.yaml                       # 只有 E2 需要，hotcent
sbatch --array=0-7 slurm/a2_fit.sh a2/a2_li_o.yaml E3-L1-linear # 0 = 全量拟合，k = 第 k 个组留出（组的顺序见 plan）
sbatch --dependency=afterok:<数组作业号> slurm/a2_post.sh a2/a2_li_o.yaml E3-L1-linear
```

**数组大小：** `--array=0-N`，N 是不作测试的能带组数（`plan` 会打印出来）。Li–O 的 D1 有 li_bcc、li_fcc、li2o、li2o2 和三个二聚体，li_hcp 作测试，所以 N = 7。

## 输出

在配置的 `out` 下（Li–O 是 `li_o/runs/a2_li_o/`）：

| 文件 | 内容 |
|---|---|
| `plan.json` | 起点、能带组、各次运行 |
| `runs/<运行>/full/` | 全量拟合：`fit.json`（系数、历史、各角色的能带损失）、`env_models.json`、`bands.json`（每个参考的比较结果），以及模型目录 `model/`（a-series-model/1，`repulsive: none`） |
| `runs/<运行>/loeo-<组>/` | 该组留出时的拟合，`band.holdout` 就是这个组的留出误差 |
| `runs/<运行>/transfer/` | `predict.jsonl`（每个结构的电子能量与力，可续跑）、伪扫描点 `points/p00000/`（含 `losses.json`：S1、S1o、S2、S3、S4、R₂b、oracle）、`transfer.json`（与 A1 起点对比） |
| `result_E1.json` 等 | G2 的输入（schema `a2-result/1`，见 `common/gates.py`） |
| `summary.md` | 一张总表 |
| `logs/a2-<阶段>-<运行>/` | 每次运行的日志：run.log、meta.json、events.jsonl、status.json |

## 约定与注意

- **k 点与展宽：** 与 A1 扫描相同，即 Γ 心的 KSPACING 网格与 Gaussian 0.05 eV（`common/models.engine_kpoints`、`engine_options`）。不要传裸的 (n1, n2, n3)，pydftb 会把它当成 Monkhorst–Pack 网格。
- **能带雅可比：** 默认用冻结电荷下的 Hellmann–Feynman 导数（`fit.band_jacobian: frozen`），不含电荷响应。在 1a 分支的测试里，去掉平移后，水分子差约 21%，Cu 金属差约 0.5%。拟合不收敛时改用 `scc`，这是精确值，每个参考要多算 2 n_x 次 SCC。
- **E2 的 B 样条不插值：** θ = 0 时每个原子拿到的是 (T₋₁ + 4T₀ + T₊₁)/6，不完全等于 A1 点本身的表，所以节点间距要小（默认 0.3 Bohr）。E2 至少要 5 个节点，两端各一个保护节点。
- **L4（MACE 特征）没有导数：** 只能拟合能带。传递检验要算力，会报错并写进 `predict.jsonl` 的 error。L4 的用途是判断描述符本身能不能把能带拟合好。
- **E2 与 E3 不能同时用**（引擎拒绝），这是下一步的引擎工作，G2 显示两者各自有用时再做。
- **续跑：** 每个阶段都可以重跑。配置改了（按内容哈希判断）就重新拟合；`--force` 强制重算。
- **冒烟测试（本机，2026-11-10）：** 数据是 Li 的冒烟扫描，参考能带用 DFTB 在另一个 r0 点造。E3（L1 linear、L2 kernel/mlp）、E1、E2（5 个节点，hotcent 136 s）的全量拟合、留一组拟合、传递检验、汇总与 G2 都跑通了。
