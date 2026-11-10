# A4：Δ-ML（执行说明）

设计见 `pydftb/docs/design/A0-accuracy-programme.html` 第六节；关口 G4 由 `common/gates.py` 判定。所有命令在 `examples/08_accuracy/` 下执行。

| 编号 | 做法 | 在哪里 |
|---|---|---|
| Δ1 | 在最好的 DFTB 模型上，用 MACE 学 DFT − DFTB 的能量与力残差 | `delta_ml.py`，`--model` 取 A1/A3 的模型目录 |
| Δ2 | 哈密顿量修正用神经网络，即 E3 配 MLP 回归器 | A2 的驱动，不在这里 |
| Δ3 | A2 的环境相关电子部分加 Δ1 | `delta_ml.py`，`--model` 取 A2 的模型目录 |

对比对象是同一批结构、同样架构的纯 MACE。学习曲线取 N = 25、50、100、全部，并看家族外（ood）测试。

## 命令

```bash
python a4/delta_ml.py prepare  --config li_o/scan_li_o.yaml --model li_o/models/S1 --out a4/runs/d1_S1 --jobs 20
python a4/delta_ml.py train    --out a4/runs/d1_S1 --device cuda            # 8 次训练，已完成的跳过
python a4/delta_ml.py evaluate --config li_o/scan_li_o.yaml --out a4/runs/d1_S1 --device cuda
python a3/a3_repulsive.py md --config li_o/scan_li_o.yaml --model a4/runs/d1_S1/models/delta_all --out a4/runs/d1_S1/md
python common/gates.py --root li_o/runs --a4 'a4/runs/**/result.json'        # G4
```

远程一次做完用 `slurm/a4_delta_gpu.sh CONFIG MODEL OUT`（dg50，1 张卡，20 个 CPU）。

## 约定

- **残差：**
  - REF_energy = E_DFT − E_DFTB，两边都是 E(σ→0)；REF_forces = F_DFT − F_DFTB。
  - 残差的 E0 = E0_DFT − E0_DFTB(模型的孤立原子)，纯 MACE 的 E0 = E0_DFT（与 K7 相同）。
  - 基础模型上 SCC 失败的结构不进训练；在评测里，它们记为 Δ 模型的失败结构。
- **子集：**
  - 训练集中固定 10% 作验证集，Δ 与纯 MACE 共用；
  - 其余按固定种子排列，取前 N 个，各 N 嵌套；
  - N 不少于训练集大小时记为 all。
- **架构：** 与 `li_o/k7_mace.py` 相同（r_max 5 Å、64 通道、L 1、correlation 3、400 epoch、SWA），Δ 与纯 MACE 一样大。`mace_args/*.json` 里的参数请对照远程的 mace-torch 版本检查。
- **色散：** 前驱体溶液的模型拟合时用无 D3 的 PBE。基础模型 `model.json` 的 `dispersion` 会传给 Δ 模型目录，只在使用时（MD）加上。
- **G4：**
  - 误差 = 相对能 MAE / 10 meV + 力 MAE / 0.1 eV/Å；
  - 在某个 N ≤ 100 的测试集上，或在最大 N 的 ood 上，Δ 比纯 MACE 好 ≥ 30%，就算通过。

## 输出（`--out`）

| 文件 | 内容 |
|---|---|
| `base/predictions.jsonl` | 基础 DFTB 模型在每个结构上的结果，按结构缓存 |
| `data/{delta,pure}/{train_N,valid,test,ood}.xyz` | 训练与评测数据（REF_energy、REF_forces） |
| `mace_args/<kind>_<N>.json`、`manifest.json` | 训练参数、子集名单、E0、架构 |
| `mace_models/`、`train/<name>.log` | 训练好的 MACE 与训练日志 |
| `eval/<name>/summary.md`、`summary.json`、`predictions.jsonl` | 每个模型的 pydftb.assess 结果 |
| `result.json` | G4 的输入（schema a4-result/1） |
| `learning_curve.md`、`learning_curve.png` | 学习曲线 |
| `models/delta_<N>/` | DFTB + Δ 的模型目录（common/models.py），可直接用于 MD |
| `logs/a4-<阶段>/…` | run.log、meta.json、events.jsonl、status.json |

## 费用（粗估，先用第一次训练校准）

| 步 | 估计 |
|---|---|
| prepare | 基础 DFTB 算一遍全部结构，与 A1 扫描的一个点相当，用 dg50 的 20 个免费 CPU |
| train | 每次训练约 15–30 分钟，共 8 次，约 2–4 卡时，约 12–24 元 |
| evaluate | 几分钟 |

## 本机验证（只为验证代码，2026-11-10）

冒烟测试的 Li 数据（9 个结构），2 个 epoch，CPU：

- prepare → train → evaluate 全流程跑通，result.json 与学习曲线都写出了；
- 用 `models/delta_all` 模型目录重算，与“缓存的 DFTB + MACE”之差为 1e-14 eV。

数字本身没有意义（数据太少、epoch 太少）。
