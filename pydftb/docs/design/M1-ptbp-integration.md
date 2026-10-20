# M1 设计：把 pydftb 接入 PTBP

## 目标

在 PTBP 中加一个开关 `--engine pydftb`。打开后，参数优化的每一步不再调用 DFTB+，而是调用 pydftb。在默认设置下，损失函数值要与 DFTB+ 一致。

## 先弄清 PTBP 实际让 DFTB+ 算了什么

PTBP 并不自己写 DFTB+ 的输入，而是通过 ASE 的 `Dftb` 计算器。ASE 写的输入带有 `ParserVersion = 1`，这时 DFTB+ 会按“老版本兼容规则”悄悄改写好几处设置。从 DFTB+ 处理后的输入 `dftb_pin.hsd` 和它的源码中确认了以下几点：

| 设置 | PTBP 代码里写的 | DFTB+ 实际使用的 | 依据 |
|---|---|---|---|
| 电子温度 | 2000 K、MP 2 阶 | **Fermi，T = 0**（关键字放错了位置，被忽略） | `test/orbit/Li2.out`，`dftb_pin.hsd` |
| SCC 类型 | 未写 | **按壳层分辨（ShellResolvedSCC = Yes）** | `oldcompat.F90:182`：老版本输入自动加入 `OrbitalResolvedSCC = Yes` |
| 力的数值求导 | 未写 | **有限差分，步长 0.01 Bohr**（现在的默认是 1.2e-4） | `oldcompat.F90:368` |
| 排斥势 | `PolynomialRepulsive = Yes` | 多项式系数全为 0，所以**排斥能为 0**（排斥能在 Python 里由 physrep 另算） | SKF 文件 |
| 混合器 | 未写 | 改进 Broyden，参数为默认值 | `dftb_pin.hsd` |
| k 点 | PTBP 的 `monkhorstpack2kptdensity`（向下取整） | `SupercellFolding`，偶数网格偏移 0.5，并按时间反演约化 | `dftb_pin.hsd` |
| 最高角动量 | ASE 自动选取 | 占据数不为 0 的最高 l | ASE 源码 |

**结论：已发表的 PTBP 参数是在“按壳层分辨的 SCC、T = 0、力用 0.01 Bohr 差分”下拟合的。** 这一点 PTBP 自己的文档里没有写，ASE + DFTB+ 的其他用户也同样会受影响。

## pydftb 侧（本阶段第一部分，已完成）

| 功能 | 作用 |
|---|---|
| `repulsive="polynomial"` | 对应 DFTB+ 的 `PolynomialRepulsive = Yes` |
| `shell_resolved=True` | 已有，与 DFTB+ 对照验证过 |
| `fd_delta=0.01` | 按 DFTB+ 的方式用有限差分求力，可以与 DFTB+ 逐位对照；不设时用精确导数 |
| Fermi / MP / Gaussian 展宽 | 照搬 DFTB+ 的费米能级搜索（先试带隙中点，再按 DFTB+ 的方式二分）和 TS 约定；另提供变分版 TS |
| 改进 Broyden 混合 | 照搬 DFTB+ 的默认混合器，SCC 迭代轮数与 DFTB+ 完全一致 |
| 不收敛时的行为 | 照常返回结果并发出警告，对应 `ConvergentSCCOnly = No` |
| `PyDFTB`（ASE 计算器） | MP 网格（与 `SupercellFolding` 一致）、显式 k 点、自动最高角动量、线程限制、类型标签；提供能带工具需要的 ASE 访问器；单位换算与 ASE 的 `Dftb` 相同 |
| γ 函数按原子做邻居搜索 | 壳层分辨时，Cu 的 p 壳层 U 很小，短程项的截断距离长达约 126 Bohr。改为“按原子搜邻居、每一对 U 用各自的截断”后，PTBP 最大 k 网格的结构从 7.9 秒降到 2.0 秒 |

## PTBP 侧（本阶段第二部分）

- 新增 `utils/engines.py`，提供 `make_calculator(mole, dftb_args, engine=None)`：
  - `engine="dftbplus"` 时走原来的代码，原样不动；
  - `engine="pydftb"` 时返回 `PyDFTB`，设置与上表一致：`shell_resolved=True`、`repulsive="polynomial"`、`scc_tol=1e-5`、`max_scc=1000`、T = 0、PTBP 的 k 网格、每进程 1 个线程。
- `dftbplus_calculator` 保留为原名，内部转交给 `make_calculator`。
- 新增选项：
  - `--engine {dftbplus,pydftb}`，默认 dftbplus；
  - `--smearing {none,fermi,mp,gaussian}`、`--smearing-temperature`、`--mp-order`，两个引擎上都真正生效；DFTB+ 一侧改为正确写入 `Filling`，并在运行后检查实际的电子温度；
  - `--pydftb-forces {dftbplus,exact}`，默认 dftbplus，即按 0.01 Bohr 差分求力，与已发表的流程逐位一致；选 exact 时用精确导数。
- 损失函数等价性检验：`tools/engine_parity.py` 在固定的几组参数下调用 `Loss.EnergyGeometry`，比较两个引擎的损失值。

## 验收

- 19 个 DFTB+ 对照算例全部通过（已完成）。
- PTBP 实际路径（ASE + DFTB+，ParserVersion 1）与 `PyDFTB` 在 `dft/dft.xyz` 上对比：能量差 ≤ 2e-7 eV，力差 ≤ 3e-6 eV/Å（已完成抽查）。
- 在 dataset 和 energygeometry 两种模式下，损失函数相对差异 < 1e-4。
- 端到端 `ptbp optimize --engine pydftb` 可以完整跑通。
