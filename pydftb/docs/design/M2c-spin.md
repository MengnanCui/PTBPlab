# M2c 设计：共线自旋极化

## 目标与用途

自旋向上（↑）和向下（↓）的电子分别占据各自的轨道，可以描述自由基、三重态分子（如 O₂）和磁性材料。对应 DFTB+ 的输入：

```
SpinPolarisation = Colinear {
  UnpairedElectrons = 2       # N↑ − N↓
  RelaxTotalSpin = No         # No：两个费米能级，固定 N↑、N↓；Yes：共用一个费米能级，总自旋自由变化
}
SpinConstants {
  ShellResolvedSpin = Yes     # 原子分辨 SCC 时默认 No（每种原子一个 W）；壳层分辨 SCC 时必须是 Yes
  O = -0.035 -0.030 -0.030 -0.028    # 每种原子一个 W 矩阵（壳层 × 壳层）
  H = -0.072
}
```

## 物理（通俗版）

- 每个壳层 l 的自旋磁矩：m_l = q_l↑ − q_l↓。
- 自旋能量：E_spin = ½ Σ_A Σ_{l,l'} W_{A,ll'} m_{Al} m_{Al'}。W 是自旋常数，为负值，所以有未配对电子时能量降低。
- 自旋势：V^s_{Al} = Σ_{l'} W_{A,ll'} m_{Al'}，分别加到两个自旋通道上：

      H↑ = H0 + ½S(V_μ + V_ν) + ½S(V^s_μ + V^s_ν)
      H↓ = H0 + ½S(V_μ + V_ν) − ½S(V^s_μ + V^s_ν)

  其中 V 是 DFTB2/DFTB3 的电荷势，用总电荷（↑ + ↓）计算。
- **填充：**
  - 默认（`RelaxTotalSpin = No`）：N↑ = (N + n)/2，N↓ = (N − n)/2，n = `UnpairedElectrons`，两个通道各有一个费米能级；
  - `RelaxTotalSpin = Yes`：两个通道共用一个费米能级，每个能级最多容纳 1 个电子。
- **初始猜测：** 与 DFTB+ 相同，每个轨道的磁矩 m = q0 · n / N。
- **能量：** E = Σ_σ Tr(ρ^σ H0) + E_SCC(q) + E_3rd(q) + E_spin(m) + E_rep − TS，用输出电荷计算。
- **力：** 自旋能量只依赖同一原子上的磁矩，没有显式的位置依赖。力里只多出能带项中的自旋势部分：½S′(V^s_μ+V^s_ν)(ρ↑ − ρ↓)。应力同理。

## 与 DFTB+ 逐轮一致的细节

- **混合向量：** DFTB+ 把电荷换成“↑/↓”表示，再按“等价轨道”求和后交给 Broyden。开了自旋后等价单位是壳层（`spin.F90:Spin_getOrbitalEquiv` 与 SCC 的等价关系合并），所以混合向量是 [各壳层 q↑, 各壳层 q↓]。即使 SCC 不按壳层分辨也是如此。收敛判据是这个向量的最大差。
- **能量和势用输出电荷**（M2b 已核实：`RecomputeAfterDensity`）。

## DFTB+ 中的对应代码

| 内容 | DFTB+ |
|---|---|
| 自旋势、自旋能量 | `dftb/spin.F90`：`getSpinShift`、`getEnergySpin_*`；`dftb/getenergies.F90`（½ m·V^s） |
| ↑/↓ 与电荷/磁矩的换算 | `spin.F90`：`qm2ud`、`ud2qm` |
| 等价轨道（混合单位） | `spin.F90:Spin_getOrbitalEquiv`，`dftb/orbitalequiv.F90`，`dftbplus/main.F90:reduceCharges` |
| 电子数、初始磁矩 | `dftbplus/initprogram.F90`：`initElectronNumber`、`initializeCharges` |
| 填充（一个或两个费米能级） | `dftbplus/main.F90:getFillingsAndBandEnergies` |
| 输入 | `dftbplus/parser.F90`：`readSpinPolarisation`、`readSpinConstants` |

## 实现要点

1. 求解器拆出 `eig(h)`（只对角化）。填充和密度矩阵放到外面，以便两个自旋通道共用或分用费米能级。SP2 求解器不支持自旋。
2. 两个自旋通道沿 k 点方向“堆叠”：ρ、W 的形状为 (2·nk, N, N)，权重为 [w_k, w_k]。这样能带能量、Mulliken 布居和力的大部分代码不用改。力中的自旋势项需要“磁化密度” ρ↑ − ρ↓，用带符号的权重 [w_k, −w_k] 求得。
3. 新参数：
   - `spin=False | True`；
   - `unpaired_electrons=0`；
   - `relax_total_spin=False`；
   - `spin_constants={label: W}`，W 为一个数（各壳层相同）或壳层 × 壳层矩阵。

   结果中新增每个原子的磁矩，以及 ↑/↓ 两个费米能级。`PyDFTB` 提供 ASE 的 `magmoms`/`magmom`。

## 测试计划

- **与 DFTB+ 对照（新参考算例）：**
  - O₂ 三重态：原子分辨 SCC + 每种原子一个 W；原子分辨 SCC + W 矩阵；壳层分辨 SCC；
  - OH 自由基（双重态）；
  - O₂，`RelaxTotalSpin = Yes`；
  - 周期 CuO 链：k 点 + 自旋，含应力；
  - OH 自由基 + DFTB3（3ob 设置），检验组合。

  W 用 DFTB+ 手册推荐的 PBE 值（H、O）；Cu 取检验用的值。
- **自洽：**
  - 自旋势等于自旋能量对磁矩的导数；
  - 力和应力与能量差分一致；
  - 不加自旋（W = 0 或 n = 0 且初始磁矩为零）时结果与无自旋计算相同。
