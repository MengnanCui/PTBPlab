# DFTB+ 源码对照表：pydftb 每个模块对应 DFTB+ 的哪里

这张表是“读懂 DFTB+ 是怎么搭起来的”学习地图。**建议的读法是：先读 pydftb 的 Python 实现，看懂公式，再去 DFTB+ 里找同一段物理的 Fortran 写法。** Python 版没有 MPI、BLACS、螺旋几何等工程分支，主线更清楚。

- **DFTB+ 源码版本**：`dftbplus/dftbplus` 的 `main` 分支，commit `041e723`（2026-10-12）。下文路径都相对于 `src/dftbp/`。
- **对照用的 DFTB+ 程序**：conda-forge 的开发版（commit `a23bfb2`，输入解析器版本 14）。
- **规模**：`src/dftbp` 共约 21.8 万行 Fortran。其中“DFTB2 核心路径”（SK 表、旋转、邻居表、SCC、混合、展宽、密度矩阵、力/应力）约 2.3 万行，其余是扩展功能和并行变体。
- **说明**：pydftb 是按公式独立实现的，**没有复制 DFTB+ 代码**（DFTB+ 是 LGPL-3 许可）。这里列出的文件只用于学习和核对。

## 1. 已实现的部分

| 物理步骤（通俗说明） | pydftb | DFTB+ 源文件 | 关键子程序 / 类型 | 备注 |
|---|---|---|---|---|
| 读 SKF 参数表 | `skf.py: SKFile.read` | `type/oldskdata.F90` | `TOldSKData`, `readFromFile`, `readSplineRep` | 首字符 `@` 表示扩展格式（f 轨道）；pydftb 暂不支持。表头写 n 行时 DFTB+ 只用前 n−1 行（`oldskdata.F90:145`），pydftb 自 M2a 起相同 |
| 按物种对读取所有 SKF | `skf.py: SKSet`, `params.py: ParamSet` | `dftbplus/parser.F90` | `readSKFiles`, `getFullTable` | DFTB+ 按“物种名”找文件（`Type2FileNames`），所以 `O2-H.skf` 这类自定义类型可以直接用 |
| 表格插值（8 点多项式 + 尾部 5 阶衰减） | `skf.py: SKFile.evaluate` | `dftb/slakoeqgrid.F90`, `math/interpolation.F90` | `TSlakoEqGrid`, `getSKIntegrals`, `polyInterUniform`, `poly5ToZero` | pydftb 预先算好每个插值窗口的多项式系数，所以导数是解析的 |
| 排斥势（样条 / 多项式） | `skf.py: Repulsive` | `dftb/repulsive/*.F90` | `TSplineRep`, `TPolyRep`, `TTwoBodyRep` | `PolynomialRepulsive = Yes` 对应 pydftb 的 `repulsive="polynomial"` |
| 最高角动量规则 | `skf.py: max_l_from_skf` | （ASE `read_max_angular_momentum`） | — | DFTB+ 由输入指定；ASE 和 PTBP 取“占据数不为 0 的最高 l” |
| Slater-Koster 旋转（键方向 → 实验室坐标） | `slako.py` | `dftb/sk.F90` | `rotateH0`, `ss/sp/sd/pp/pd/dd` | DFTB+ 手写了每个 (l1,l2) 的公式；pydftb 用实球谐旋转矩阵统一处理，结果相同 |
| 邻居表（含周期镜像） | `neighbors.py` | `dftb/periodic.F90`, `type/dynneighlist.F90` | `TNeighbourList`, `updateNeighbourListAndSpecies`, `getCellTranslations` | pydftb 用 k-d 树 |
| H0 和 S 的构建及其导数 | `hamiltonian.py: build_pairs` | `dftb/nonscc.F90` | `buildH0`, `buildS`, `getFirstDerivFiniteDiff` | **DFTB+ 用数值差分（中心差分，默认步长 1.22e-4，老版本输入为 0.01）**，pydftb 默认用复步长法（解析精度）；设 `fd_delta` 后可以逐位复现 DFTB+ 的力（差异 1e-12） |
| 稀疏存储 → 稠密 / k 点矩阵 | `hamiltonian.py: fold, fold_all` | `dftb/sparse2dense.F90` | `getSparseDescriptor`, `unpackHS`, `unpackHS_cmplx_kpts` | 相位因子是 exp(2πi k·T) |
| 短程 γ 函数 | `coulomb.py: _short, _gamma0` | `dftb/shortgammafuncs.F90`, `dftb/shortgamma.F90` | `expGamma`, `expGammaPrime`, `TShortGamma` | “两个 U 相等”的判据 \|ΔU\| < 3.125e-6 已与 DFTB+ 一致 |
| 长程 1/R（Ewald 求和） | `coulomb.py: Coulomb` | `dftb/coulomb.F90` | `TCoulomb`, `ewaldReal`, `ewaldReciprocal`, `getOptimalAlphaEwald` | — |
| SCC 势加入哈密顿量 | `calculator.py: DFTB._scc` | `dftb/hamiltonian.F90`, `dftb/shift.F90`, `dftb/scc.F90` | `getSccHamiltonian`, `addShift`, `TScc` | H = H0 + ½S(V_μ+V_ν) |
| Mulliken 布居 | `calculator.py` | `dftb/populations.F90`, `dftb/charges.F90` | `mulliken` | — |
| 轨道分辨 SCC | `calculator.py: _sites` | `dftb/uniquehubbard.F90`, `dftb/orbitalequiv.F90` | `TUniqueHubbard` | — |
| 电荷混合 | `mixer.py: BroydenMixer`（默认），`AndersonMixer`（Pulay 型） | `mixer/broydenmixer.F90` | `modifiedBroydenMixing` | 照搬 DFTB+ 的改进 Broyden（M1）：19 个对照算例的 SCC 轮数与 DFTB+ 完全相同 |
| 占据数与费米能级 | `occupations.py` | `dftb/etemp.F90`，`dftbplus/main.F90:getFillingsAndBandEnergies` | `Efilling`, `electronCount`, `electronFill`, `middleGap` | M1 已照搬：先试带隙中点，再按 DFTB+ 的方式二分；T 不低于 1e-8 Ha；Gaussian 即 MP 0 阶；另提供变分熵 |
| 本征求解与密度矩阵 | `solvers.py` | `math/eigensolver.F90`, `dftb/densitymatrix.F90` | `hegvd`, `getDensityMatrix`, `getEDensityMatrix` | pydftb 另有 SP2 纯化和批量求解器 |
| 力 | `calculator.py: DFTB._gradient` | `dftb/forces.F90`, `dftbplus/main.F90` | `derivative_shift`, `getGradients` | — |
| DFTB3 三阶项（M2b） | `thirdorder.py: ThirdOrder`, `dgamma_du`, `dgamma_du_onsite` | `dftb/thirdorder.F90`，`dftb/chargepenalty.F90`（`ThirdOrder`） | `gamma3`, `gamma2pU`, `gamma2pUpR`, `dGdUr0`, `updateCharges`, `getShifts`, `addGradientDc`, `addStressDc` | U 几乎相等时 pydftb 改用稳定的积分形式（见第 3 节第 6 条） |
| H–X 阻尼（M2b） | `coulomb.py: Coulomb._sp_short`，`thirdorder.py: dgamma_du` | `dftb/shortgammafuncs.F90`，`dftbplus/initprogram.F90` | `expGammaDamped`, `expGammaDampedPrime`, `hh`, `hpU`；被阻尼的是质量 < 3.5 amu 的原子 | — |
| 共线自旋（M2c） | `calculator.py`：`_Spin`、`_spin_matrix`、`_potentials`、`_solve`、`_populations` | `dftb/spin.F90`，`dftbplus/main.F90`（`getFillingsAndBandEnergies`、`getNextInputCharges`、`reduceCharges`），`dftbplus/initprogram.F90`（`initElectronNumber`、`initializeCharges`） | `getSpinShift`, `getEnergySpin_*`, `Spin_getOrbitalEquiv`, `qm2ud`, `Efilling` | 两个自旋通道沿 k 点方向堆叠；混合向量与 DFTB+ 相同（↑/↓、按壳层） |
| 应力（M2a） | `calculator.py: DFTB._gradient(virial=True)`，`coulomb.py: Coulomb.gradient(virial=True)` | `dftb/stress.F90`，`dftb/coulomb.F90`，`dftbplus/main.F90` | `getBlockStress`, `invRStress`, `getStress` | 每个原子对的“力 × 键矢量”求和，加 Ewald 倒空间项和背景项；与力在同一遍里算出 |
| SCC 主循环 | `calculator.py: DFTB.calculate` | `dftbplus/main.F90` | `processGeometry`, SCC 循环 `lpSCC` | pydftb 分成 `_setup/_scc/_energies/_gradient` 四段 |
| 能量记账（Total / Mermin / Extrapolated） | `calculator.py: DFTBResult` | `dftb/getenergies.F90`, `dftbplus/mainio.F90` | `writeDetailedOut*` | — |

## 2. 待实现部分对应的 DFTB+ 文件（路线图见 `ROADMAP.md`）

| 功能 | DFTB+ 源文件 |
|---|---|
| 非共线自旋、自旋轨道耦合 | `dftb/spinorbit.F90`，`dftb/sparse2dense.F90`（`unpackHPauli`） |
| DFTB+U、onsite 修正 | `dftb/dftbplusu.F90`，`dftb/onscorrection.F90` |
| 长程校正 LC/CAM | `dftb/hybridxc.F90`（7,459 行），`dftb/rshgamma.F90` |
| 色散 | `dftb/dispuff.F90`，`dispslaterkirkw.F90`，`simpledftd3.F90`，`dispdftd4.F90`，`dispmbd.F90` |
| 氢键 / 卤素修正 | `dftb/h5correction.F90`，`dftb/halogenx.F90` |
| 溶剂 | `solvation/born.F90`，`cosmo.F90`，`sasa.F90`，`cm5.F90` |
| 外场、外部点电荷 | `dftb/extfields.F90`，`dftb/extcharges.F90` |
| 结构优化 | `geoopt/`（`lbfgs.F90`，`fire.F90`，`gdiis.F90`，`conjgrad.F90`） |
| 分子动力学 / XL-BOMD | `md/velocityverlet.F90`，`md/thermostats.F90`，`md/xlbomd.F90` |
| 激发态（Casida / 实时） | `timedep/linresp.F90`，`timedep/linrespgrad.F90`，`timedep/timeprop.F90` |
| REKS、pp-RPA | `reks/`，`timedep/pprpa.F90` |
| 输运（NEGF） | `transport/negfint.F90`（底层是 libNEGF） |
| 并行与求解器框架 | `common/mpienv.F90`，`common/blacsenv.F90`，`type/parallelks.F90`（k 点/自旋分组），`elecsolvers/`，`extlibs/elsiiface.F90` |
| 输入解析与初始化 | `io/hsdparser.F90`，`dftbplus/parser.F90`（8,685 行），`dftbplus/initprogram.F90`（7,606 行） |

## 3. 对照时踩过的坑

1. **ParserVersion。** 解析器版本低于 14 时，DFTB+ 会把 `MethfesselPaxton { Order = n }` 悄悄改成 n−1（`dftbplus/oldcompat.F90`），而 ASE 写的是版本 1。pydftb 的对照工具固定写 14。
2. **数值求导方式。** DFTB+ 的 `Differentiation = Richardson {}` 在这些 SKF 上反而比默认差分更不准：水分子的力误差从 1.2e-6 升到 1.2e-5 Ha/Bohr。原因可能是它的步长较大，跨过了插值多项式更换窗口的格点。因此参考数据使用默认差分。
3. **SKF 表格少用一行。** 表头写 `0.02, 719`、文件里也有 719 行时，DFTB+ 只读前 718 行（`type/oldskdata.F90:145`，`nGrid = nGrid - 1`）。表格的终点因此是 14.36 Bohr，而不是 14.38 Bohr，最后几个插值窗口和其后 1 Bohr 的衰减尾巴都随之改变。Cu、H、O、C 的积分在这么远处已经极小，看不出差别；Li 的 2s 轨道很弥散，Li₂ 在 14.5–14.9 Bohr 处的能量差达到 2.5e-7 Ha，力甚至反号。M2a 起 pydftb 与 DFTB+ 一致（差异约 1e-11 Ha）。
4. **数值求导跨过网格节点。** DFTB+ 用步长 1.22e-4 Bohr 的中心差分求 H0、S 的导数。如果某个原子间距离到最近网格节点的距离比这个步长还小，差分的两端会落在两个不同的插值多项式上，导数就有误差。在晶格常数 3.45 Å 的 Li bcc 中，a√2 那一层邻居离第 461 个节点只有 4.3e-5 Bohr，DFTB+ 的应力因此偏差 1.2e-8 Ha/Bohr³。pydftb 默认用复步长精确求导，没有这个问题；设 `fd_delta` 后能逐位复现 DFTB+ 的结果（差异 6e-13）。第 2 条中 Richardson 求导变差，很可能也是这个原因。
5. **Γ 在 U 恰好相等时大一倍。** DFTB+ 的 `gamma2pU` 在 |U_a − U_b| < 3.125e-6 时对共同的 τ 求全导数，否则求偏导；按对称性，偏导恰好是全导数的一半。3ob 等参数按这个约定拟合，pydftb 照搬。
6. **U 几乎相等时，DFTB+ 的 γ 闭式公式失效。** 公式里 1/(τ_a² − τ_b²)ⁿ 各项彼此几乎完全抵消。|ΔU| = 1e-5 时，S 的相对误差是 6e-4，∂γ/∂U 的相对误差是 24。DFTB3 下实测总能量错 0.5 eV（ΔU = 4e-6）。pydftb 在 U 相差不到 5% 时改用 Hermite–Genocchi 积分形式，误差 ≤ 4e-12（见 `reports/M2b-dftb3.md`）。
7. **费米能级捷径影响力的自洽。** “带隙中点”只要让电子数误差 ≤ 1e-7 就被接受；电子数不严格守恒时，力与能量导数会相差约 5e-7 Ha/Bohr（实例：按壳层分辨的 Cu₄CO，300 K）。
8. **开自旋时第一轮不混合。** `main.F90:getNextInputCharges` 在第一次 SCC 迭代、且开了自旋（或 DFTB+U 等）时，直接用输出电荷作为下一轮输入，不调用混合器。不照做，SCC 轮数就对不上（O₂：2 轮对 4 轮）。
9. **“能级全部占满”按“能级数 × 自旋通道数”判断**（`etemp.F90:Efilling`）。把自旋通道当作额外的 k 点处理时，要乘上通道数，否则会误判。
