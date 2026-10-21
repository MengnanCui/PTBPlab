# M2d 设计：色散修正

## 目标与用途

DFTB 的能量项都来自短程的原子轨道重叠，缺少长程的范德华吸引（色散），所以分子晶体、层状材料、物理吸附、生物分子的结构和结合能会有系统性偏差。通常的做法是在总能量上加一个只依赖原子位置的经验修正 E_disp。它不改变电子结构，也不参与 SCC 循环。

DFTB+ 的输入写在 `Hamiltonian` 块里，例如：

```
Dispersion = DftD3 {
  Damping = BeckeJohnson { a1 = 0.5719  a2 = 3.6017 }
  s6 = 1.0
  s8 = 0.5883
}
```

pydftb 中对应的写法：`DFTB(..., dispersion={"model": "dftd3", "damping": "beckejohnson", "a1": 0.5719, "a2": 3.6017, "s6": 1.0, "s8": 0.5883})`。键名与 DFTB+ 的关键字一一对应（小写、下划线），单位与 DFTB+ 的默认单位相同（原子单位：Bohr、Hartree）。

## DFTB+ 的六种色散，pydftb 怎么做

| DFTB+ 关键字 | DFTB+ 的实现 | pydftb |
|---|---|---|
| `LennardJones`（UFF） | 原生（`dftb/dispuff.F90`） | 原生 NumPy，按同样的公式和截断规则 |
| `SlaterKirkwood` | 原生（`dftb/dispslaterkirkw.F90`） | 原生 NumPy |
| `DftD3` | 调用 s-dftd3 库（`extlibs/sdftd3.F90`） | 调用同一个库的 Python 绑定 `dftd3`（1.0.0，与 DFTB+ 链接的版本相同） |
| `SimpleDftD3` | 原生 D3(BJ) 二体项（`dftb/simpledftd3.F90`） | 映射到同一个 `dftd3` 库：BJ 阻尼、无三体项、截断照 DFTB+ |
| `DftD4` | **原生**（`dftb/dispdftd4.F90`，不调用 dftd4 库） | 调用参考实现 dftd4 库的 Python 绑定 `dftd4`（3.6.0） |
| `Ts`、`Mbd` | libmbd | 暂缓 |

为什么 D3/D4 用库：

- D3 和 D4 的参考数据（C6 表、参考体系的极化率）量很大，而且属于 LGPL-3 的代码库，不能复制进 MIT 许可的 pydftb。
- 两个库都是作者组维护的参考实现。`dftd3`/`dftd4` 是可选依赖，只在运行时导入，没装时给出安装提示。

代价：这两项暂时不能放到 GPU 上，也不能对色散参数自动求导（M4 再考虑）。

## 先做的探测实验

动手实现之前，先在 scratchpad 里直接比较了 DFTB+ 和两个库。DFTB+ 的色散能量取“加色散”与“不加色散”两次计算的差，力和应力同理（电子部分完全相同，可以逐位相减）。

| 模型 | 体系 | 能量差 (Ha) | 力差 (Ha/Bohr) | 结论 |
|---|---|---|---|---|
| `DftD3`（BJ / 零阻尼 / 三体项） | 水二聚体、甲醇、水盒子、CuO 链 | ≤ 6e-14 | ≤ 1e-14 | 同一个库，逐位一致 |
| `SimpleDftD3` → `dftd3` 库 | 同上 | ≤ 6e-11 | ≤ 7e-11 | 一致（参考数据舍入） |
| `DftD4` | 只含 H、C、O 的分子 | ≤ 1e-11 | ≤ 6e-11 | 一致 |
| `DftD4` | Cu₄CO 分子 | 2.1e-3 | 8e-5 | **不一致**，见下文 1 |
| `DftD4` | 周期体系（默认参数） | 4e-4 – 1e-3 | 1e-5 – 3e-4 | **不一致**，见下文 2、3 |
| `DftD4` | 周期体系，`ChargeScale = 0`、`s9 = 0`、截断对齐 | 1e-10（水盒子） | 1e-10 | 一致 |

### 1. Cu、Zn、Ag、Cd、Au、Hg：DFTB+ 多用一个参考体系

逐项比较了 DFTB+ 的 `external/dftd4refs/dftd4_references.fh` 和 dftd4 3.6.0 的 `src/dftd4/reference.inc`：3366 项共同数据全部相同，只有 6 项不同，即上述 6 种元素的参考体系数 `refn` 不同（DFTB+ 为 3，库为 2）。

- 第 3 个参考体系（Cu：CN = 2.9，电荷与缩放取整数值）在库的数据里仍然存在，但没有启用。
- 所以 Cu 的配位数在 1.5 以上时两者差别很大（Cu₃：1e-7 Ha；Cu₄CO：2e-3 Ha，约为色散能本身）；双原子分子（CN ≤ 1）则完全一致。
- pydftb 跟随参考实现（库）。与 DFTB+ 对照的 D4 算例不含这 6 种元素。

### 2. DFTB+ 周期体系的 EEQ 电荷有错

D4 的 C6 依赖原子电荷，电荷由 EEQ 模型求得；周期体系中 EEQ 的库仑矩阵要用 Ewald 求和。

- 把水二聚体放进越来越大的盒子：
  - 库的结果收敛到分子值（L = 40 Å 时差 4e-9 Ha）；
  - DFTB+ 不收敛，差值约按 1/L 衰减（12、20、40 Å 时分别为 4.4e-5、2.4e-5、1.1e-5 Ha）。
- DFTB+ 的结果随 EEQ 的 Ewald 劈分参数变化（`EwaldParameter` 取自动、0.15、0.3 时分别为 −7.83e-4、−8.34e-4、−9.09e-4 Ha）。正确的 Ewald 和不可能依赖这个参数。
- `ChargeScale = 0`（关掉电荷依赖）后两者一致到 4e-12。
- 源码中可疑的地方：`dftb/encharges.F90:addEwaldContribs` 的内层循环从 `iAt2f = iAt1` 开始，并且对 `aMat(iAt2f, iAt1)` 和 `aMat(iAt1, iAt2f)` 各加一次。对角元因此加了两次倒空间项，自能修正也只有 −α/√π，而不是 −2α/√π。

### 3. DFTB+ 周期体系的 D4 三体项计数有错

- **超胞检验：** 同一个 CuO 链晶体，用 1×1×1、2×1×1、1×2×1 晶胞计算，每 4 个原子的三体能量：
  - 库：都是 3.333770e-4 Ha；
  - DFTB+：2.760e-4、2.787e-4、3.054e-4 Ha。
  DFTB+ 的结果随所选晶胞变化，随超胞增大逐渐接近库的值。
- **原因（源码）：**
  - DFTB+ 的近邻表只存“像原子的中心胞编号 ≥ 本原子”的近邻（`dftb/periodic.F90:updateNeighbourList`）。
  - 三体循环（`dispdftd4.F90:threeBodyDispersionGradient`）从编号最小的原子出发枚举三元组，再按 `tripleScale` 打折扣。
  - 对“两个原子是同一原子的不同像”的三元组，这个折扣与实际枚举次数不符：
    - (i, j′, j″) 只被枚举一次，却乘了 ½；
    - (i, i′, i″) 枚举 3 次、每次乘 1/6，合计 ½。
  - 梯度和应力不乘这个折扣，所以力与能量也不自洽。
  - 另外，截断只判断从编号最小原子出发的两条边，结果依赖原子编号顺序。
- **库的做法：** 三条边都在截断内才计入，按枚举次数配权，超胞检验通过。

### 4. 截断

dftd4 3.6 的实空间截断写死在代码里：

| | 双体 | 三体 | 配位数 |
|---|---|---|---|
| dftd4 3.6（C/Python 接口都不能改） | 60 Bohr | 40 Bohr | 30 Bohr |
| DFTB+ 默认 | 64 Bohr | 40 Bohr | 40 Bohr |

对分子没有影响。对稠密周期体系，双体截断的差别约为 1e-7 Ha/原子（水盒子 24 个原子：3e-6 Ha）。参考算例中 DFTB+ 一律写成库的值（`CutoffInter = 60`、`CoordinationNumber = Cov { Cutoff = 30 }`），pydftb 给出与库不同的截断时报错。

### 小结：D4 与 DFTB+ 如何对照

- 分子（不含第 11、12 族元素）：完整模型对照。
- 周期体系：用 `ChargeScale = 0`、`s9 = 0` 对照二体部分（配位数、参考体系权重、C6、BJ 阻尼、周期求和、应力）。
- 完整模型的周期行为改由 pydftb 的单元测试检查物理一致性：大盒子收敛到分子极限、超胞不变性。

## 各模型的公式与 DFTB+ 细节

### Lennard-Jones / UFF（`dftb/dispuff.F90`、`dispuffdata.F90`）

- **参数：** 每种原子一个距离 x_i 和阱深 D_i。UFF 表来自 Rappé 等，JACS 114, 10024 (1992) 表 1（103 种元素，Å 与 kcal/mol）。换算用 DFTB+ 的常数：1 Bohr = 0.529177249 Å，1 kcal/mol = 0.0015946683859874898 Ha。表中 103 号元素在 DFTB+ 里写作 “Lw”，pydftb 用 “Lr”。
- **组合规则：** d = √(dᵢdⱼ)，r = √(rᵢrⱼ)；c6 = 2 d r⁶，c12 = d r¹²。r > r0 = 2^(−1/6) r 时 E = c12/R¹² − c6/R⁶。
- **短程“墙”：** R ≤ r0 时改用 E = U0 − U5 R⁵ − U10 R¹⁰，其中 U0 = 396/25·d，U5 = 2^(5/6)·672/25·d/r⁵，U10 = −2^(2/3)·552/25·d/r¹⁰。它在 r0 处与 LJ 的值和斜率都连续（两者都为 0 和 −24d/r0），避免 r⁻¹² 在极短距离发散。
- **分子：** 截断取 (max c6 / 1e-9)^(1/6) 以内的所有原子对。
- **周期：**
  - 实空间只算 c12/R¹² 和短程墙；墙的区域要把 Ewald 已经减掉的 −c6/R⁶ 加回来。截断取 (max c12 / 1e-11)^(1/12)。
  - 全部 −c6/R⁶ 项用 r⁻⁶ 的 Ewald 和（见下文）。
- **类型标签：**
  - DFTB+ 按物种名的前两个字母查 UFF 表（`O2` 查不到，`Li1` 会查成 Li）；
  - pydftb 用 `element_of_label` 映射到元素，也可以逐标签给数值；
  - 参考算例中，类型标签一律写显式数值。

### Slater-Kirkwood（`dftb/dispslaterkirkw.F90`）

- **参数：**
  - 每个原子给极化率 α、范德华半径 R、有效电子数 N（`PolarRadiusCharge`，逐原子）；
  - 或者每种原子给共价半径和 13 个值（`HybridDependentPol`：6 个极化率、6 个半径、1 个电子数）。13 个值按邻居数 n 选取：n ≤ 4 取第 n+1 个，n ≥ 5 取第 6 个。距离不超过两原子共价半径之和的原子算邻居，包括周期像。
  - DFTB+ 只在读入时按输入结构定一次，pydftb 每次计算按当前结构定。单点结果相同；只有在结构优化中邻居数发生变化时才不同。
- **公式：**
  - C6 = 1.5 αᵢαⱼ / (√(αᵢ/Nᵢ) + √(αⱼ/Nⱼ))，R0 = (Rᵢ³ + Rⱼ³)/(Rᵢ² + Rⱼ²)；
  - E = −C6 [1 − exp(−3 (R/R0)⁷)]⁴ / R⁶；
  - α 或 R 为 0 的原子不参与。
- **分子：** 截断取 (max C6 / 1e-9)^(1/6)。
- **周期：**
  - 先对全部原子对做纯 −C6/R⁶ 的 Ewald 和；
  - 再在“阻尼截断”内加上修正 +C6 (1 − f⁴)/R⁶。阻尼截断由 1 − f⁴ < 1e-10 求出，R0 取所有原子对中最大的。

### r⁻⁶ 的 Ewald 求和（`dftb/dispcommon.F90`）

E = −½ Σᵢⱼ Σ_L′ C6ᵢⱼ / |rᵢⱼ + L|⁶ 收敛很慢，用 Ewald 方法拆成三部分（x = R/η）：

- **实空间：** −½ Σ C6 e^(−x²)(1 + x² + x⁴/2)/R⁶；
- **倒空间：** −π^(3/2)/(24V) Σ_{G≠0} |G|³ h(|G|η/2) Σᵢⱼ C6ᵢⱼ cos(G·rᵢⱼ)，其中 h(b) = √π erfc(b) + (1/(2b²) − 1) e^(−b²)/b；
- **自能和 G = 0 项：** + Σᵢ C6ᵢᵢ/(12η⁶) − π^(3/2) Σᵢⱼ C6ᵢⱼ/(6Vη³)。

参数选取照 DFTB+：

- η = η_opt/√2，η_opt = √(a·V/(π |a×b|))，a、b 是最短的两个晶格矢量；
- 实空间、倒空间截断用二分法（最多 30 轮），使误差估计低于 1e-9 Ha。

这样两边截取的是同一组 L 和 G，结果一致到舍入误差。

应变导数（应力）：

- 实空间项与普通对势相同；
- 倒空间项 = −E_rec δ_ab + π^(3/2)/(24V) Σ_G S(G)·3|G|·(√π erfc(b) − e^(−b²)/b)·G_a G_b；
- G = 0 项 ∝ 1/V。

### DftD3（库）与 HHRepulsion（原生）

- **参数逐项照 DFTB+（`extlibs/sdftd3.F90`）：**
  - BJ 阻尼：a1、a2，三体项 alp = 14；
  - 零阻尼：rs6 = sr6、rs8 = 1、alp = alpha6；
  - 修正零阻尼另有 beta。
  - `Threebody = Yes/No` 对应 s9 = 1/0（DFTB+ 默认关）。
  - 截断：`set_realspace_cutoff(disp2 = Cutoff, disp3 = CutoffCN, cn = CutoffCN)`，默认 Cutoff = √9000 Bohr、CutoffCN = 40 Bohr。
- **HHRepulsion（D3H5 的 H–H 排斥）：**
  - 10 Bohr 以内的每对氢原子加 E = k [1 − 1/(1 + exp(−e (R/r0 − 1)))]，k = 0.30 kcal/mol，e = 14.31，r0 = 2.35 Å；
  - pydftb 原生实现。
  - DFTB+ 对“氢原子与自己的周期像”这类原子对，能量按两次计（近邻表同时含 +L 和 −L），应力却按一半计。只有晶胞小于 10 Bohr 时才会出现；pydftb 按物理正确的方式计数，参考算例避开这种情况。
- 类型标签用 `atomic_numbers={label: Z}`，对应 DFTB+ 的 `AtomicNumbers`。

### SimpleDftD3 → 同一个 D3 库

- DFTB+ 的 SimpleDftD3 就是 D3(BJ) 的二体项：s6 = 1，s8、a1、a2 必填，s10 = 0，`CutoffInter = 64`，`WeightingFactor = 4`，`CoordinationNumber = Exp { Cutoff = 40 }`。
- 用库计算：BJ 阻尼、s9 = 0、disp2 = CutoffInter、cn = CN 截断。
- 库不支持的设置直接报错：s10 ≠ 0、WeightingFactor ≠ 4、CN 不是 Exp、CutCN ≠ 0。

### DftD4（库）

- **参数：**
  - s6（默认 1），s8、s9、a1、a2 必填，alpha（默认 16）；
  - WeightingFactor、ChargeSteepness、ChargeScale 对应库的 wf、gc、ga（默认 6、2、3），不等于默认值时用库的 `custom_d4_model`；
  - s10 必须为 0。
- **EEQ 电荷：** 需要体系的总电荷，pydftb 把 `DFTB(charge=...)` 传进去。DFTB+ 的 `ChargeModel = SelfConsistent`（用 DFTB 布居代替 EEQ 电荷并与 SCC 耦合）暂缓。
- **截断：** 见上文“截断”一节。
- **常用参数：** 库的参数表中，3ob 的 D4（含三体项）为 s8 = 0.6635015、a1 = 0.5523240、a2 = 4.3537076、s9 = 1；mio 为 s8 = 1.2916225、a1 = 0.5965326、a2 = 4.8778602。

## 接口

新模块 `pydftb/dispersion.py`：

- 每个模型提供 `evaluate(labels, pos, cell, pbc, grad=True, charge=0.0)`，返回 `DispersionTerms(energy, gradient, virial)`：
  - gradient = dE/dR；
  - virial = dE/dε，与 `calculator.py:_repulsive_gradient` 的约定相同。库返回的 sigma 也是 dE/dε。
- `make_dispersion(spec)`：spec 是 dict（`"model"` 取 DFTB+ 关键字的小写形式）、模型对象或 None。
- 原生模型只支持非周期或三维周期；部分周期的体系只能用库模型。

接入 `calculator.py`：

- 色散在 `_setup` 中按几何结构算一次（它与 SCC 无关），存进 `_Setup.disp`；
- `_energies` 把 `e_disp` 加进总能量（与 DFTB+ 相同，Total、Mermin、Extrapolated 都含色散），`terms["dispersion"]`；
- `_gradient` 加上梯度和维里。

`PyDFTB(dispersion=...)` 和 `pydftb.SCC(..., dispersion=...)` 透传。

## 验证计划

| 算例 | 内容 |
|---|---|
| `water_dimer_uff`、`water_box_uff`、`cuo_chain_uff_k422` | UFF：分子、周期（应力）、k 点 |
| `water_dimer_sk`、`ch3oh_sk_hybrid`、`water_box_sk_hybrid` | SK：逐原子参数；按邻居数选参数（含 α = 0 的原子） |
| `ch3oh_d3bj`、`water_box_d3bj_3body`、`cuo_chain_d3bj_k422` | D3(BJ)，含三体项和周期 |
| `water_dimer_d3zero_3body`、`water_dimer_d3_hhrep`、`water_dimer_typed_d3` | 零阻尼 + 三体项；HHRepulsion；类型标签 + AtomicNumbers |
| `ch3oh_simple_d3`、`water_box_simple_d3` | SimpleDftD3 |
| `water_dimer_d4`、`ch3oh_d4`、`water_box_d4_2body` | D4：分子完整模型；周期二体（ChargeScale = 0、s9 = 0） |
| `h2o_dftb3_d3bj`、`o2_triplet_d3bj` | 与 DFTB3、自旋组合 |

单元测试（`tests/test_dispersion.py`）：

- 各模型的力、应力与能量差分一致；
- Ewald 结果与 η 无关；
- 超胞不变性；
- D4 大盒子收敛到分子极限；
- UFF 参数抽查、标签映射、选项校验。

没装库或库版本与参考 JSON 记录不同时，D3/D4 算例跳过，并写明原因。

## 局限

- 不支持 TS/MBD；D4 的自洽电荷变体（`ChargeModel = SelfConsistent`）暂缓。
- D3/D4 依赖库，不能放到 GPU 上，也不能对参数自动求导。
- D4 截断固定为库的值。
