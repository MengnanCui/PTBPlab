# M0 设计：底座整理

## 目标

原型阶段已经证明 pydftb 能和 DFTB+ 算得一样。M0 不加新物理，而是为后面所有阶段打底座，满足三点：

1. **加新功能有固定的位置**，不必每次都改同一个大函数。
2. **以后换成 GPU / PyTorch 时不需要重写物理代码。**
3. **每个功能都能和 DFTB+ 对照**，而且测试不依赖 DFTB+。

## 设计决定

| 问题 | 决定 | 理由 |
|---|---|---|
| 目录结构 | 保持扁平（`pydftb/*.py`），只新增文件 | 代码只有约 2 千行，搬进 `core/`、`physics/` 只会打断引用，没有收益（采纳计划审查意见） |
| 与硬件相关的操作 | 新增 `backend/ops.py`，只放 `scatter_add`、`segment_sum`、本征分解、Cholesky、线性求解；`scatter_add` 是**函数式**的（返回新数组） | PyTorch 的自动求导要求不原地修改；其余标准操作直接用数组库 |
| 一个大函数 `calculate()` | 拆成 `_setup`、`_scc`、`_energies`、`_gradient` 四段，数学完全不变 | DFTB3、自旋、应力等功能各有落脚点；拆分后结果逐位相同 |
| “元素”与“参数”混在一起 | `System` 同时记录**类型标签**和**元素**；`ParamSet` 负责“标签 → SKF 文件” | 为“同一元素多套参数”（M5）打基础；标签约定与 DFTB+/ASE 的 `Type2FileNames` 相同（`O2-H.skf`），可以直接和 DFTB+ 对照 |
| 对外接口 | PySCF 风格的 `pydftb.SCC(system, params).kernel()` | 状态保存在对象里，用户熟悉；原有的 `DFTB`、`SKSet`、`PyDFTB` 保持不变 |
| 邻居表 | 整数配对表（i, j, 平移）与键矢量分开（`pair_vectors`） | 以后求应力或换后端时，只需重新计算矢量 |
| 与 DFTB+ 一致的数值约定 | 新增 `constants.py` 中的 `MIN_HUB_DIFF` 等容差，γ 函数中“两个 U 相等”的判据与 DFTB+ 相同 | 原判据（\|Δτ\|<1e-8）与 DFTB+ 不同，在 M5 的平滑切换中会造成不连续 |
| 测试数据 | 两套 SKF（H/C/O/Cu、H/O/Li）以 gzip 格式放进 `tests/data/skf`，并附来源和 sha256；pydftb 直接读取 `.skf.gz` | 原先依赖 `/tmp` 里的临时文件，无法复现 |
| 与 DFTB+ 对照 | 新增 `pydftb.validation` 子包：`cases.py`（算例登记）、`dftbplus.py`（运行 DFTB+）、`refgen.py`（`--regenerate` / `--check` / `--live`）；参考结果写成 JSON，记录精确坐标、设置、HSD 文本和 DFTB+ 版本 | 测试只需 pydftb，CI 可以直接运行 |

## 测试计划

- 重构前后，3 个代表体系的结果**逐位相同**（能量、电荷完全一致，力只因求和顺序不同而相差 1e-17）。
- 12 个对照算例全部在容差内：
  - 原型阶段的 8 个体系；
  - 新增 4 个：非 SCC、金属 T=0、Li 参数集，以及**多类型标签（`O2`）经 DFTB+ `Type2FileNames`**。
- 新接口与底层 `DFTB` 的结果一致；带类型标签的计算与普通计算逐位相同。
