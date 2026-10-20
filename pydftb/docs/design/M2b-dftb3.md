# M2b 设计：DFTB3（三阶项、Hubbard 导数、H–X 阻尼）

## 目标与用途

DFTB2 假设原子的“硬度”（Hubbard U）与它带多少电荷无关。DFTB3 让 U 随电荷线性变化：U_A(q) = U_A + Uᵈ_A·Δq_A。斜率 Uᵈ 叫 Hubbard 导数。这一修正对带电体系、质子转移和氢键很重要。3ob 参数集就是为 DFTB3 拟合的。

DFTB+ 对应的输入（都写在 `Hamiltonian = DFTB { ... }` 里）：

```
ThirdOrderFull = Yes                  # 完整三阶项；或 ThirdOrder = Yes（只有原子自身的部分）
HubbardDerivs { O = -0.1575  H = -0.1857 }
HCorrection = Damping { Exponent = 4.0 }   # H–X 阻尼（3ob 的标准设置）
```

## 物理（通俗版）

### 三阶能量

记 Δq_a 为电荷位点 a 上多出的电子数（位点是原子；按壳层分辨 SCC 时是壳层），ΔQ_A 为原子 A 上多出的电子总数：

    E₃ = (1/3) Σ_ab Γ_ab · ΔQ_A(a) · Δq_a · Δq_b

    Γ_ab = (∂γ_ab / ∂U_a) · Uᵈ_a

γ 是 DFTB2 里描述两个位点上电荷相互作用的函数（`coulomb.py`），Γ 就是“a 的 U 变了，γ 变多少”。Γ 不对称：Γ_ab 用 a 的 Uᵈ，Γ_ba 用 b 的。γ 的长程部分 1/R 与 U 无关，所以 Γ 只来自短程部分 S，按指数衰减，不需要 Ewald 求和。

| 情形 | Γ 的取法（与 DFTB+ 相同） |
|---|---|
| 同一原子上的两个位点（R = 0），U 相同 | Γ = ½ Uᵈ_a |
| 同一原子上的两个位点（R = 0），U 不同 | Γ = (∂γ₀/∂U_a) Uᵈ_a，γ₀ 是 R = 0 时的 γ（Gaus 2015 式 S7） |
| 不同原子，\|U_a − U_b\| ≥ 3.125e-6 | Γ = −3.2 ∂S₁/∂τ_a · Uᵈ_a，对 τ_a 的**偏导** |
| 不同原子，\|U_a − U_b\| < 3.125e-6 | Γ = −3.2 dS₂/dτ · Uᵈ_a，对共同 τ 的**全导数** |

最后两行有一个要注意的地方。对称性要求“偏导 = 全导数的一半”，数值检验也确认了这一点（比值 0.5007）。也就是说，DFTB+ 在两个 U 恰好相等时用的 Γ 是“U 稍有不同”时的两倍。这应该是 DFTB3 原始实现的定义，3ob 参数也是按它拟合的，所以 pydftb 照搬。对 M5“同一元素多套参数”有直接影响：两套 Li 参数的 U 只要相差超过 3.125e-6，它们之间的三阶耦合就减半。验证时要用 DFTB+ 实测这个跳变。

### 势（进入哈密顿量）

对 Δq_c 求导，得到位点 c 上的三阶势：

    V₃[c] = (1/3) [ ΔQ_C (ΓΔq)_c + (Γᵀ(ΔQ∘Δq))_c + Σ_{a∈C} Δq_a (ΓΔq)_a ]

三项分别对应 DFTB+ 的 `shift1`、`shift2`、`shift3`。V₃ 加到 DFTB2 的势上，H = H0 + ½S(V_μ + V_ν)，和 SCC 势完全一样地进入哈密顿量。

### `ThirdOrder = Yes`（只有原子自身的部分）

只保留 R = 0 的项：E₃ = Σ_A (Uᵈ_A/6) ΔQ_A³，V = (Uᵈ_A/2) ΔQ_A²。DFTB+ 只允许它和“不按壳层分辨”的 SCC 一起用，并且把这部分能量记在 “Energy SCC” 里，不记在 “Energy 3rd”。

### H–X 阻尼（`HCorrection = Damping`）

对含氢原子的原子对（DFTB+ 的判据是 SKF 里的质量 < 3.5 amu，也就是 H 及其同位素），短程函数乘以

    h = exp(−((U_a + U_b)/2)^ζ · r²)

它同时作用于 DFTB2 的 γ（S → S·h）和三阶的 Γ（对 U 求导时 h 也要求导）。ζ 在 3ob 中取 4.0。没有三阶项时也可以单独用阻尼。

### 力和应力

Γ 只依赖原子间距离，所以三阶项的力与 SCC 的短程项结构相同：每个原子对给出 dE₃/dr，然后乘以键方向；应力就是“dE/dr × 键矢量”的求和（M2a 的维里框架）。对一个有序位点对 (a, b)：

    dE₃/dr = (1/3) Δq_a Δq_b [ ΔQ_A ∂Γ_ab/∂r + ΔQ_B ∂Γ_ba/∂r ]

此外，V₃ 进入了 H，所以能带项的力（½ S′(V_μ+V_ν) ρ）自动包含三阶势，不需要另写。

## DFTB+ 中的对应代码

| 内容 | DFTB+ |
|---|---|
| Γ、∂Γ/∂r、阻尼函数 h 及其导数 | `dftb/thirdorder.F90`：`gamma3`、`gamma3pR`、`gamma2pU`、`gamma2pUpR`、`hh`、`hpU`、`hpR`、`hpUpR` |
| 三阶势 shift1/2/3、能量 | `thirdorder.F90`：`updateCharges`、`getShifts`、`getEnergyPerAtom` |
| 力、应力 | `thirdorder.F90`：`addGradientDc`、`addStressDc` |
| 三阶项的截断 | `thirdorder.F90:calcCutoffs`：每对物种用各自最小的 U，按 `expGammaCutoff`（S < 1e-10）求截断 |
| 二阶阻尼 | `dftb/shortgammafuncs.F90`：`expGammaDamped`、`expGammaDampedPrime` |
| 哪些原子被阻尼 | `dftbplus/initprogram.F90`：质量 < 3.5 amu |
| 只有原子自身的三阶项 | `dftb/chargepenalty.F90`（κ = 3，前置因子 Uᵈ/6） |
| 输入解析 | `dftbplus/parser.F90`：`ThirdOrder`、`ThirdOrderFull`、`HubbardDerivs`、`readHCorrection` |
| 用哪一步的电荷 | `dftbplus/main.F90`：每次对角化之后用输出电荷重算势（`RecomputeAfterDensity`，默认开），能量用输出电荷 |

## 实现要点

1. 新模块 `thirdorder.py`：
   - 移植 DFTB+ 的 Γ 系列函数（向量化）；
   - `ThirdOrder` 类：复用 `Coulomb` 的原子邻居表，按 DFTB+ 的物种截断筛选原子对，再展开到位点对。Γ 存成稠密矩阵（与 γ 一样），提供 `energy_shift(dq)` 和 `gradient(dq, virial)`。
2. `Coulomb` 增加阻尼：每个位点对一个“是否阻尼”标记，S 和 dS/dr 乘以 h 后再用。
3. `DFTB` 新参数：
   - `third_order=None | "full" | "onsite"`；
   - `hubbard_derivs={label: 值或每壳层的列表}`；
   - `h_damping=None | ζ`。
   能量结果新增 `e_3rd`。`PyDFTB` 透传这三个参数。
4. 截断：照搬 DFTB+ 的 `expGammaCutoff` 算法（翻倍再二分），保证与 DFTB+ 筛选出的原子对相同。

## 测试计划

- **单元测试：**
  - Γ 的 r 导数与有限差分一致；
  - V₃ 与 E₃ 对电荷的数值导数一致；
  - 收敛后的力和应力与能量的有限差分一致；
  - `"onsite"` 等价于只保留对角块。
- **与 DFTB+ 对照（新增参考算例）：**
  - 水、甲醇：完整三阶 + 阻尼；
  - 水二聚体：完整三阶、不阻尼；只有阻尼、不加三阶；
  - 水：只有原子自身的三阶项；
  - Cu₄CO：按壳层分辨，s/p/d 各自的 Uᵈ（检验同一原子上 U 不同的位点）；
  - 周期水盒子：完整三阶 + 阻尼，含应力；
  - CuO 链（k 点）：完整三阶，含应力。

  Uᵈ 用 3ob 的公开值：H −0.1857、C −0.1492、O −0.1575；Cu 没有 3ob 值，取一个检验用的数（如 −0.05）。这里只检验程序与 DFTB+ 一致，不涉及参数的物理质量。
- **实测 Γ 的跳变：** 用 DFTB+ 算一个二聚体，比较“两个原子用同一份参数”和“其中一个的 U 改动 1e-5”时的三阶能量。
- **验收：** 能量差 < 1e-6 Ha，力差 < 1e-5 Ha/Bohr（与 M0 相同），应力差 < 1e-7 Ha/Bohr³。
