# M0 完成报告：底座整理（含原型阶段回顾）

## 一、原型阶段回顾（已完成的物理）

| 步骤 | 实现方式 | 与 DFTB+ 的关系 |
|---|---|---|
| SKF 读取与插值 | 8 点多项式插值，尾部 5 阶多项式衰减；每个窗口的多项式系数预先算好 | 与 DFTB+ `slakoeqgrid.F90` 逐点一致 |
| Slater-Koster 旋转 | 用实球谐旋转矩阵统一处理 s/p/d，不手写 100 多个表项 | 结果与教科书公式（以及 DFTB+ `sk.F90`）相同，单元测试逐项核对 |
| H0/S 导数 | 复步长法（解析精度） | **比 DFTB+ 更准**：DFTB+ 用数值差分，误差约 1e-6 Ha/Bohr |
| SCC + Ewald | γ 函数短程部分 + Ewald 长程部分，按原子或按壳层 | 与 DFTB+ 一致 |
| 力 | 能带项 + SCC 项 + 排斥项，全部解析 | 与能量的数值差分相差约 3e-9 |

## 二、本阶段完成的内容

- **`backend/ops.py`**：分发层，函数式的 `scatter_add`；所有 `np.add.at`、`bincount`、本征分解调用都已改为经过这一层。
- **四段式计算**：`_setup`、`_scc`、`_energies`、`_gradient`。结果新增 DFTB+ 风格的能量名称（`e_total`、`e_mermin`、`e_zero`、`terms`）以及壳层和轨道布居。
- **`System` / `ParamSet` / `SCC`**：PySCF 风格接口；类型标签可以不同于元素（`O2`、`Li1`、`Li_el`、`Li@cat`）。
- **`write_typed_copies`**：为新类型生成 SKF 副本，并自动处理“同一份同核文件用于两个不同类型时要改成异核格式”。这是 M5 的核心文件操作。
- **`max_l_from_skf`**：与 ASE/PTBP 完全相同的最高角动量规则，在所有测试文件上与 ASE 的结果一致。
- **SKF 读取**：支持 `.skf.gz`；按（路径、修改时间、大小）缓存，PTBP 每步重写文件后会自动重新读取。
- **`pydftb.validation`**：12 个算例，参考结果冻结为 JSON（124 KB）。
- **测试**：移到 `pydftb/tests/`，共 27 个，全部通过；没有 dftb+ 时，需要它的测试会自动跳过。
- **CI**：新增 GitHub Actions 工作流 `.github/workflows/pydftb-tests.yml`，只装 numpy/scipy/ase 即可跑全部离线测试。
- **打包**：`pyproject.toml` 增加可选依赖组 `torch`、`dispersion`、`mpi`、`dev`，`threadpoolctl` 改为核心依赖。
- **文档**：`WHY.md`、`ROADMAP.md`、`glossary.md`、`dftbplus_map.md` 和本阶段的设计文档。

## 三、验证结果（`python -m pydftb.validation.refgen --check`）

| 算例 | 原子数 | 能量差 (Ha) | 最大力差 (Ha/Bohr) | 电荷差 (e) | SCC 轮数 pydftb / DFTB+ |
|---|---|---|---|---|---|
| h2o | 3 | 1.8e-11 | 1.2e-06 ※ | 3.8e-12 | 10 / 8 |
| h2o_nonscc | 3 | 9.7e-12 | 6.9e-07 ※ | 2.2e-15 | 1 / 1 |
| ch3oh | 6 | 9.3e-12 | 1.4e-08 | 6.4e-11 | 15 / 12 |
| cu4co_300K | 6 | 1.6e-11 | 1.6e-08 | 2.5e-09 | 11 / 11 |
| cu4co_shell_resolved | 6 | 1.6e-08 | 1.5e-08 | 2.4e-08 | 27 / 19 |
| cu_fcc_k6 | 1 | 4.6e-10 | — | 1.6e-15 | 1 / 1 |
| cu8_distorted_k355 | 8 | 8.3e-10 | 4.7e-07 ※ | 2.5e-09 | 11 / 9 |
| water_box_gamma | 24 | 1.3e-07 | 4.3e-08 | 1.8e-07 | 23 / 19 |
| cuo_chain_k422 | 4 | 3.8e-09 | 8.9e-09 | 4.5e-08 | 18 / 16 |
| **water_dimer_typed**（第二个 O 用类型 `O2`） | 6 | 1.8e-11 | 1.0e-08 | 2.6e-11 | 15 / 11 |
| li2h2_orbit（第二套参数，Li 只用 s） | 4 | 4.8e-11 | 2.3e-09 | 1.4e-11 | 13 / 10 |
| cu_fcc_k6_T0（金属，T=0） | 1 | 1.3e-10 | — | 3.6e-14 | 1 / 1 |

※ 这几项的力差来自 DFTB+ 自身的数值求导。pydftb 的力与它自己能量的数值差分只差约 3e-9。

**换算成常用单位**：最大能量差 1.3e-7 Ha ≈ 3.5e-6 eV，最大力差 1.2e-6 Ha/Bohr ≈ 6e-5 eV/Å，两个程序的答案可以认为相同。

## 四、发现的问题和学到的东西

1. **修复了一个潜伏的 bug。** 多项式排斥势的系数错位了一列。SKF 格式是 `mass c2…c9 rcut`，原代码取成了 c3…c9 加上 rcut。PTBP 的多项式系数全为零，而且测试文件都有样条，所以一直没有暴露。M1 的“多项式排斥势”开关依赖这一项，已改正，将在 M1 与 DFTB+ 对照。
2. **DFTB+ 的 ParserVersion 陷阱。** 版本低于 14 时，MP 展宽的阶数会被悄悄减 1。原型阶段写的是 12，ASE 写的是 1。现在固定写 14。
3. **Richardson 求导反而更差。** 计划审查建议参考数据用 DFTB+ 的 Richardson 求导，实测在这些 SKF 上误差反而变大（水分子力误差从 1.2e-6 升到 1.2e-5），因此改回默认差分，并在代码里记录了原因。
4. **γ 函数判据与 DFTB+ 对齐。** “两个 U 相等”的判据改为 DFTB+ 的 \|ΔU\| < 3.125e-6，并在相等时用平均 τ。原判据在两种公式的切换处不连续，M5 的平滑切换会碰到这个问题。
5. **DFTB+ 新版输入格式的几处变化**：`Analysis{CalculateForces}` 改名为 `PrintForces`；非 SCC 计算不能出现 `SCCTolerance` 等关键字；numpy 2 的浮点数 `repr` 形如 `np.float64(…)`，DFTB+ 读不了。这些都已在对照工具里处理。
6. **SCC 轮数不同。** pydftb 用 Pulay 型混合，DFTB+ 用改进 Broyden，收敛结果一致但轮数不同。PTBP 设 `ConvergentSCCOnly = No`，结果在理论上依赖混合路径，所以 M1 会补上 Broyden。

## 五、留给用户的问题

- **仓库没有 LICENSE 文件。** 如果要像 PySCF 一样开源，需要选一个许可证（常见的有 Apache-2.0、BSD-3、MIT）。注意：DFTB+ 是 LGPL-3，pydftb 是按公式独立实现、没有复制它的代码，所以选择不受它约束。（2026-10 更新：已选定 MIT，覆盖整个仓库，见根目录 `LICENSE`。）

## 六、下一步（M1）

- pydftb 侧：排斥势开关与 DFTB+ `PolynomialRepulsive` 对照；显式 k 点；`max_scc` 和不收敛告警；ASE 访问器；按 DFTB+ 的方式实现 Fermi / MP / Gaussian 展宽和费米能级搜索；改进 Broyden 混合。
- PTBP 侧：`--engine`、`--smearing` 开关，修正 DFTB+ 的 Filling 写法，损失函数等价性检验，端到端运行。
