# M2c 完成报告：共线自旋极化

## 一句话结论

pydftb 现在支持共线自旋极化（DFTB+ 的 `SpinPolarisation = Colinear`），覆盖以下情形：

- 固定未配对电子数（两个费米能级）和总自旋自由变化（共用一个费米能级）；
- 每种原子一个自旋常数，或按壳层的自旋常数矩阵；
- 原子分辨与壳层分辨 SCC；
- 能量、力、应力，以及与 DFTB3 组合。

新增 7 个与 DFTB+ 对照的算例全部通过，**SCC 迭代轮数全部与 DFTB+ 相同**。分子的能量差 ≤ 5e-11 Ha，周期 CuO 链的能量差为 5e-9 Ha，应力差为 2e-10 Ha/Bohr³。

## 一、做了什么

| 位置 | 内容 |
|---|---|
| `solvers.py` | 拆出 `eig(h)`（只对角化）；填充可以用自旋因子 1，以便两个自旋通道共用或分用费米能级 |
| `calculator.py` | 新参数 `spin`、`unpaired_electrons`、`relax_total_spin`、`spin_constants`。两个自旋通道沿 k 点方向“堆叠”，能带能量、Mulliken 布居和力的大部分代码不变。力中的自旋势项用“磁化密度” ρ↑ − ρ↓，通过带符号的权重求得。结果新增 `e_spin`、每个原子的磁矩 `magmoms`、↑/↓ 布居 `q_spin` 和两个费米能级 `fermi_spin` |
| `occupations.py` | 修正“所有能级都已占满”的判断（见第三节第 3 条） |
| `ase_calc.py`、`api.py` | ASE 的 `magmom`/`magmoms`、`get_number_of_spins`，按自旋取本征值和占据数，`get_fermi_levels`；`SCC` 重启时带上自旋布居 |
| `validation/` | 自动写入 DFTB+ 的 `SpinPolarisation` 和 `SpinConstants`（仅在原子分辨 SCC 时写 `ShellResolvedSpin`，否则 DFTB+ 会报错）；7 个新参考算例 |
| `tests/test_spin.py` | 8 个测试，不需要 DFTB+ |

用法：

```python
atoms.calc = PyDFTB("path/to/skf", spin=True, unpaired_electrons=2,
                    spin_constants={"O": [[-0.035, -0.030], [-0.030, -0.028]], "H": -0.072})
atoms.get_magnetic_moments()
```

## 二、验证结果

### 2.1 与 DFTB+ 对照

自旋常数：H、O 取常见的 PBE 值；Cu 取检验用的值。

| 算例 | 内容 | \|ΔE\| (Ha) | \|ΔF\| (Ha/Bohr) | SCC 轮数 pydftb/DFTB+ |
|---|---|---|---|---|
| `o2_triplet` | O₂ 三重态，每种原子一个 W（`ShellResolvedSpin = No`） | 1.6e-11 | 1.4e-8 | 2/2 |
| `o2_triplet_wmatrix` | 同上，W 矩阵 | 8.1e-12 | 1.4e-8 | 5/5 |
| `o2_triplet_shell` | 同上，壳层分辨 SCC | 3.8e-11 | 1.4e-8 | 6/6 |
| `oh_radical` | OH 自由基（双重态），300 K | 4.5e-11 | 1.2e-8 | 11/11 |
| `o2_relaxed_spin` | 共用一个费米能级（`RelaxTotalSpin = Yes`），300 K | 8.1e-12 | 1.4e-8 | 5/5 |
| `oh_radical_dftb3` | 自旋 + DFTB3 + H–X 阻尼 | 3.2e-11 | 1.2e-8 | 10/10 |
| `cuo_chain_spin_k422` | 周期 CuO 链，k 点，500 K | 5.2e-9 | 8.8e-9 | 23/23 |

周期链的应力差为 2.0e-10 Ha/Bohr³。

### 2.2 自洽检验

- 自旋常数全部为 0 时，结果与不加自旋的计算一致：能量差 < 1e-11，力差 < 1e-10；
- 固定自旋时，各原子磁矩之和严格等于 `UnpairedElectrons`，E_spin = ½ Σ W M²；
- 力与能量差分一致（< 1e-8，含壳层分辨和 DFTB3 组合），周期体系的应力与应变差分一致；
- 共用费米能级时，两个通道的费米能级相同。

## 三、发现的问题与学到的东西

### 1. 开自旋时，DFTB+ 第一轮不混合电荷

O₂ 三重态在 DFTB+ 中 2 轮就收敛，pydftb 最初要 4 轮。查 DFTB+ 的 SCC 输出：第 2 轮的误差是 1.9e-14，说明第 2 轮的输入就等于第 1 轮的输出。对应源码（`dftbplus/main.F90:getNextInputCharges`）：

```fortran
if ((iSCCIter + iGeoStep) == 1 .and. (nSpin > 1 .or. tMixBlockCharges) .and. .not. tReadChrg) then
  qInpRed(:) = qOutRed        ! 不混合，直接用输出
```

原因是自旋的初始猜测（每个轨道的磁矩 = 参考布居 × n/N）很粗糙，不值得混进去。pydftb 照此处理后，7 个算例的迭代轮数全部一致。

### 2. 混合向量的形式要与 DFTB+ 完全相同

DFTB+ 交给 Broyden 的是“↑/↓ 表示、按壳层求和”的布居。即使 SCC 不按壳层分辨，开自旋后也是按壳层：自旋模块的等价关系是逐壳层的，与 SCC 的等价关系取交集后就成了逐壳层。Broyden 的权重依赖向量的范数，换成“电荷/磁矩”表示虽然只差一个常数因子，迭代路径也会不同。

### 3. pydftb 的“能级已全部占满”判断有误（已修正）

共用费米能级的 O₂ 算例最初总能量错了 3.5 Ha。原因是 `occupations.fill` 判断“电子数不少于能级数，所以全部占满”时，只比较了一个 k 点（一行）的能级数。把两个自旋通道当作两行 k 点堆叠后，每行 8 个能级，而电子有 12 个，于是被误判为“全部占满”。DFTB+ 的判断是“能级数 × 自旋通道数”（`dftb/etemp.F90:Efilling`）。修正为“能级数 × 权重之和”，因为 k 点权重在每个自旋通道内归一。

### 4. 0 K 下简并能级半填充时，最后几轮迭代是噪声

OH 是线性分子，π 能级二重简并，自旋向下的通道只有 1 个电子落在其中。在 0 K 下：

- 两个程序的 SCC 误差序列直到 4e-9 都完全相同；
- 之后 DFTB+ 自己在 1e-9 附近来回波动（1e-10 → 1.3e-9 → 1e-10），收敛轮数因此成了偶然。

改用 300 K 后，两边每一轮都相同，第 10 轮同时收敛到 2.1e-12。所以这两个参考算例取 300 K。

## 四、与计划的偏差

- 原计划用“初始磁矩全为零”的极限检验，改为“自旋常数全为 0”的极限，因为 DFTB+ 的初始磁矩由未配对电子数决定，不能单独设为零（`InitialSpins` 尚未实现）。
- 尚未实现：`InitialSpins`（逐原子的初始磁矩）、非共线自旋（M6）。
- 验收：能量差 < 1e-6 Ha、力差 < 1e-5 Ha/Bohr、应力差 < 1e-7 Ha/Bohr³ 均满足，迭代轮数也全部一致。

## 五、下一步

按 M2 表格继续：色散（UFF/Lennard-Jones、Slater-Kirkwood，D3/D4 通过现成库）。
