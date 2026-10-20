# M2a 设计：应力张量

## 目标与用途

应力（压强）告诉我们晶胞“想要”变大还是变小，用于：

- 晶格优化：自动找到平衡晶格常数；
- 恒压分子动力学（NPT）；
- 状态方程：PTBP 的 energygeometry 模式现在只拟合能量-体积曲线，以后可以把应力也加入损失函数。

## 物理（通俗版）

把整个晶体按应变 ε 拉伸，所有原子坐标和晶胞一起变形：r → (1+ε) r。应力就是能量对应变的导数除以体积：

    σ_αβ = −(1/V) ∂E/∂ε_αβ

DFTB 的能量由几部分组成，每部分对应变的导数如下：

| 能量项 | 对应变的导数 | 说明 |
|---|---|---|
| 能带项 Tr(ρH0)，以及 SCC 势带来的 S 修正 | Σ_对 (∂E/∂R_对)_α (R_对)_β | 对每个原子对，用“力”乘以“键矢量”。求力时已经算出了每对的 ∂E/∂R |
| 排斥能 ½Σ V(r) | ½Σ_有序对 V′(r) R_α R_β / r | 同上 |
| SCC 短程 γ 和 Ewald 实空间部分 | ½Σ_有序对 Δq_i Δq_j g′(r) R_α R_β / r | 同上 |
| Ewald 倒空间部分 | ½Σ_G f(G)\|S(G)\|² [−δ_αβ + 2G_αG_β(1/(4α²) + 1/G²)] | 倒格矢随应变改变，体积也随之改变 |
| 中性化背景项 | +πQ²/(2Vα²) δ_αβ | 只有带电晶胞才有 |
| 电子熵 TS | 0 | 不显式依赖几何 |

DFTB+ 中的对应代码：

- `dftb/stress.F90:getBlockStress`：能带和 SCC 移位项；
- `dftb/coulomb.F90:invRStress`：Ewald；
- `dftb/repulsive/*:getStress`：排斥项；
- `dftbplus/main.F90:getStress`：汇总。

## 实现要点

1. 复用 `DFTB._gradient` 中已有的 `gp`（每个原子对的 ∂E/∂R），加上 `PairBlocks.vec`，得到 Σ gp ⊗ vec。
2. `Coulomb` 新增 `virial(dq)`，返回短程项、实空间项、倒空间项和背景项。
3. 排斥项的 virial 在 `_repulsive_gradient` 中顺手累加。
4. `DFTBResult.stress`（Hartree/Bohr³，3×3）；`PyDFTB` 提供 ASE 的 `stress`（Voigt 记号，单位 eV/Å³，按 ASE 的符号约定）。
5. 注意事项：
   - 用于检验的“应变有限差分”必须让原子坐标随晶胞一起缩放；
   - Ewald 的 α 和倒格矢集合在差分过程中要固定，否则能量会因截断改变而不连续；
   - 与 DFTB+ 对照只用中性晶胞：DFTB+ 的 `invRStress` 没有处理带电背景项的应变导数。

## 测试计划

- 对 Cu fcc、畸变的 Cu8 晶胞（非对角分量）、CuO 链（Ewald）、水盒子，做应变有限差分自洽检验，目标 1e-9 Ha/Bohr³。
- 与 DFTB+ 对照：在 `refgen` 中增加 `stress`，容差取绝对值 1e-7 Ha/Bohr³（约 3 MPa）。DFTB+ 的力有有限差分误差，对照时同样用 `fd_delta` 复现。
- 用 ASE 的晶胞过滤器（`FrechetCellFilter`）对 Cu fcc 做晶格优化，与能量-体积曲线的极小值对比。
