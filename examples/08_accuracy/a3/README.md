# A3：排斥势这一维（执行说明）

设计见 `pydftb/docs/design/A0-accuracy-programme.html` 第五节；关口 G3 由 `common/gates.py` 判定。本目录只有一个脚本 `a3_repulsive.py`，所有命令在 `examples/08_accuracy/` 下执行。

| 编号 | 做法 | 本目录的命令 |
|---|---|---|
| R1 | physrep，每个元素一个 σ（A1 的形式） | `score --variant R1`，对 A1 导出的模型打分 |
| R2 | 每个原子对一条 2 体样条（R₂b） | `score --variant R2`，对 `ptbp scan export ... --repulsive R2b` 的模型打分 |
| R3 | 随环境变化的排斥势：M8a 的排斥势标度通道 V_AB·exp((p_A+p_B)/2)，p = Φ(CN)·θ，可另加对多项式 | `r3` |
| R4 | 电子部分（on-site 平移、按壳层的跃迁标度）与排斥势一起在能量 + 力上精调，能带作正则 | `r4` |

## 命令

```bash
# 父模型：A1 某个策略最好的点导出的完整 SK 文件
ptbp scan export li_o/scan_li_o.yaml --point I --strategy S1 --out li_o/models/S1

python a3/a3_repulsive.py score --config li_o/scan_li_o.yaml --model li_o/models/S1 --out a3/runs/r1_S1 --variant R1
python a3/a3_repulsive.py r3 --config li_o/scan_li_o.yaml --parent li_o/models/S1 --out a3/runs/r3_S1 --jobs 64 \
    [--centres 3] [--pair Li-O:3.2:4] [--offsets none|per_element|global] [--max-evaluations 60]
python a3/a3_repulsive.py r4 --config li_o/scan_li_o.yaml --parent li_o/models/S1 --out a3/runs/r4_S1 --jobs 64 \
    [--channels onsite hopping repulsive] [--band-weight 1.0] [--atom-iterations 2]
python a3/a3_repulsive.py md --config li_o/scan_li_o.yaml --model a3/runs/r3_S1/model --out a3/runs/r3_S1/md
python common/gates.py --root li_o/runs --a3 'a3/runs/**/result.json'        # G3
```

远程用 `slurm/a3_node.sh`（dg83 一个节点；`A3_MD=1` 时接着做 MD）。

## 约定

- **能量零点：** 按用户的决定（2026-11-09），拟合默认用绝对原子化能。目标放在模型的零点上：E_target = E_DFT − ΣE0_DFT + ΣE0_模型，E0_模型 取父模型的孤立原子。
- **F 与 E(σ→0)：** fit_adaptive 拟合 Mermin 自由能 F，A 系列的能量是 E(σ→0)。所以每个目标再加上父模型在这个结构上的 F − E(σ→0)。R3 的电子部分不变，这一项是精确的。
- **偏移：**
  - `--offsets none`（默认）需要 A2 这一轮给 `fit_adaptive` 加的 `offsets` 选项；
  - `per_element` 是按元素偏移的对照；
  - `global` 是现有的行为（所有原子一个偏移），旧版 pydftb 只能用这个。
- **权重：**
  - 每个家族的总权重相同，与 L_EF 一致；
  - `--force-weight auto` = (0.010/0.10)² / ⟨3N⟩，使 10 meV/原子 与 0.1 eV/Å 等权。
- **基函数：** CN 的 RBF，中心铺满训练集的 CN 范围，并按 M8d 的做法夹在这个范围内，外推时参数不再变化。
- **R4：**
  - 能带正则需要 `fit_adaptive(band_samples=..., band_weight=...)` 和 `BandSample`，都是 A2 这一轮加的；没有时用 `--band-weight 0`。
  - on-site 通道会改变孤立原子，所以 `--atom-iterations` 次重拟合时，目标按新的孤立原子移动。F − E(σ→0) 仍取父模型的值，这是近似。
- **R3 的另一种做法没有做：** 按 M5a 的类型混合排斥势需要 E2 的类型对 SK 集。如果 G2 需要，再加。

## 输出（`--out`）

| 文件 | 内容 |
|---|---|
| `model/` | 模型目录：`model.json`（a-series-model/1）和 `env_models.json`（拟合的 AdaptiveParams） |
| `parent/summary.md`、`parent/predictions.jsonl` | 父模型在同一套打分下的结果 |
| `summary.md`、`summary.json`、`predictions.jsonl` | 新模型的 pydftb.assess 结果，按结构缓存 |
| `result.json` | G3 的输入（schema a3-result/1，见 `common/gates.py`），含父模型的结果作对照 |
| `result.md` | 一页小结 |
| `md/md_results.json`、`md/md_summary.md`、`md/*.traj` | MD 检验，同时写进 `result.json` 的 `md` 字段 |
| `logs/a3-<阶段>/<时间>-<主机>-<pid>/` | run.log、meta.json、events.jsonl、status.json |

## 续跑

- 预测按结构缓存，重跑只算缺的。
- 已有 `model/` 时跳过拟合，`--force` 重新拟合，`--force-predict` 重新预测。
- MD 已完成的 (体系, 温度) 也会跳过。

## 本机验证（只为验证代码，2026-11-10）

数据是冒烟测试的 Li，9 个结构：

- `r3` 做了 3 次 TRF 评估（`--offsets global`，因为本机 pydftb 还没有 offsets 选项），写出了模型目录、result.json 和 result.md；
- `score` 和 `md` 都跑了（6 步 NVE）；
- 加载模型目录后，环境模型确实参与计算。

R4 还没有实跑，它要等 A2 的能带接口。
