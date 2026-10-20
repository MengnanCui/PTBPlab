# pydftb 路线图与进度

规则：每个阶段（或阶段内的一块工作）开始前写一份**设计文档**（`design/`），结束后写一份**完成报告**（`reports/`）；新功能都要和 DFTB+ 对照，参考结果固化在 `pydftb/tests/data/ref/`。工作量单位为人周，均为估计值。

## 阶段状态

| 阶段 | 内容 | 状态 | 文档 |
|---|---|---|---|
| 原型 | SCC-DFTB 原型、加速实验、调研报告 | ✅ 完成 | `../RESEARCH.md` |
| M0 | 底座整理：ops 分发层、`System`/`ParamSet`/`SCC` 接口、四段式计算、冻结的 DFTB+ 参考数据、文档骨架 | ✅ 完成 | `design/M0-foundation.md`，`reports/M0-foundation.md` |
| M1 | 接入 PTBP：`--engine pydftb`、`--smearing` 开关、MP/Gaussian 展宽、Broyden 混合、排斥势开关、PTBP 损失函数等价性 | 进行中 | — |
| M2 | DFTB+ 功能补齐·第一波：应力、DFTB3、自旋、色散、能带/DOS、结构优化/MD、f 轨道 | 计划 | — |
| M3 | 计算策略推荐器 v1（CPU）+ 结构/k 点并行分发 + DFTB+ 配置顾问 | 计划 | — |
| M4 | PyTorch 后端：GPU、自动求导、批量 | 计划 | — |
| M5 | 同一元素多套参数（离散类型 → 平滑切换），固态电池示范 | 计划（离散类型的基础已在 M0 完成） | — |
| M6 | DFTB+ 功能补齐·第二波（按需）：自旋轨道耦合、DFTB+U、长程校正、溶剂、外场、TD-DFTB、XL-BOMD | 计划 | — |
| M7 | 大体系：GPU 混合精度 SP2、Chebyshev、分治多机 | 计划（CPU 原型见 `RESEARCH.md`） | — |
| M8 | AI 层：随环境连续变化的参数、ML 排斥势、主动学习 | 计划 | — |

## 与 DFTB+ 的功能对照清单

✅ 已实现并与 DFTB+ 对照；⬜ 未实现；— 不打算自己实现（调用现成库，或按需再说）

| 类别 | 功能 | 状态 |
|---|---|---|
| 哈密顿量 | 非 SCC / SCC (DFTB2) | ✅ |
| | 轨道分辨 SCC | ✅ |
| | s/p/d 轨道 | ✅ |
| | f 轨道（扩展 SKF 格式） | ⬜ M2 |
| | DFTB3、H–X 阻尼 | ⬜ M2 |
| | xTB (GFN1/GFN2) | — 调用 `tblite` |
| 周期体系 | k 点、时间反演约化、Ewald | ✅ |
| | 应力、晶格优化 | ⬜ M2 |
| 占据 | Fermi 展宽 | ✅ |
| | Methfessel-Paxton、Gaussian 展宽 | ⬜ M1 |
| 混合器 | Pulay 型（Anderson） | ✅ |
| | 改进 Broyden（DFTB+ 默认） | ⬜ M1 |
| 排斥势 | 样条 | ✅ |
| | 多项式 | ✅ 解析（M0 修复了系数错位）；与 DFTB+ 对照在 M1 |
| 多类型 | 同一元素多个类型标签（`Type2FileNames` 约定） | ✅（M0，已与 DFTB+ 对照） |
| 自旋 | 共线自旋 | ⬜ M2 |
| | 非共线自旋、自旋轨道耦合 | ⬜ M6 |
| 修正项 | 色散 D3/D4/UFF/SK | ⬜ M2 |
| | DFTB+U、onsite 修正、长程校正 | ⬜ M6 |
| | 溶剂、外场、QM/MM | ⬜ M6 |
| 驱动 | 结构优化、MD（通过 ASE） | ⬜ M2（`ase_calc.PyDFTB` 已经可以给 ASE 优化器用） |
| | XL-BOMD | ⬜ M6 |
| 激发态 | Casida | ⬜ M6 |
| | 实时 TD、REKS、pp-RPA | — 按需 |
| 输运 | NEGF | — 调用 libNEGF / 按需 |
| 求解器 | LAPACK 本征分解、批量 | ✅ |
| | SP2 纯化（稠密、混合精度、稀疏线性标度） | ✅ 原型 |
| | GPU、分治、多机 | ⬜ M4/M7 |
