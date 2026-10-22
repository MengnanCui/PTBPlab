# DFTB+ 源码对照表：pydftb 每个模块对应 DFTB+ 的哪里

这张表是“读懂 DFTB+ 是怎么搭起来的”学习地图。**建议的读法是：先读 pydftb 的 Python 实现，看懂公式，再去 DFTB+ 里找同一段物理的 Fortran 写法。** Python 版没有 MPI、BLACS、螺旋几何等工程分支，主线更清楚。

- **DFTB+ 源码版本**：`dftbplus/dftbplus` 的 `main` 分支，commit `041e723`（2026-10-12）。下文路径都相对于 `src/dftbp/`。
- **对照用的 DFTB+ 程序**：conda-forge 的开发版（commit `a23bfb2`，输入解析器版本 14）。
- **规模**：`src/dftbp` 共约 21.8 万行 Fortran。其中“DFTB2 核心路径”（SK 表、旋转、邻居表、SCC、混合、展宽、密度矩阵、力/应力）约 2.3 万行，其余是扩展功能和并行变体。
- **说明**：pydftb 是按公式独立实现的，**没有复制 DFTB+ 代码**（DFTB+ 是 LGPL-3 许可）。这里列出的文件只用于学习和核对。

## 1. 已实现的部分

| 物理步骤（通俗说明） | pydftb | DFTB+ 源文件 | 关键子程序 / 类型 | 备注 |
|---|---|---|---|---|
| 读 SKF 参数表 | `skf.py: SKFile.read` | `type/oldskdata.F90` | `TOldSKData`, `readFromFile`, `readSplineRep` | 首字符 `@` 表示扩展格式（f 轨道，M2g 起支持，见坑 29）。表头写 n 行时 DFTB+ 只用前 n−1 行（`oldskdata.F90:145`），pydftb 自 M2a 起相同 |
| 按物种对读取所有 SKF | `skf.py: SKSet`, `params.py: ParamSet` | `dftbplus/parser.F90` | `readSKFiles`, `getFullTable`（`skMap`） | DFTB+ 按“物种名”找文件（`Type2FileNames`），所以 `O2-H.skf` 这类自定义类型可以直接用。20 个积分的编号与 `skMap` 相同（`skf.py: SK_COLUMN`）；l₁ ≤ l₂ 取 A-B 文件，否则取 B-A 文件 |
| 表格插值（8 点多项式 + 尾部 5 阶衰减） | `skf.py: SKFile.evaluate` | `dftb/slakoeqgrid.F90`, `math/interpolation.F90` | `TSlakoEqGrid`, `getSKIntegrals`, `polyInterUniform`, `poly5ToZero` | pydftb 预先算好每个插值窗口的多项式系数，所以导数是解析的 |
| 排斥势（样条 / 多项式） | `skf.py: Repulsive` | `dftb/repulsive/*.F90` | `TSplineRep`, `TPolyRep`, `TTwoBodyRep` | `PolynomialRepulsive = Yes` 对应 pydftb 的 `repulsive="polynomial"` |
| 最高角动量规则 | `skf.py: max_l_from_skf` | （ASE `read_max_angular_momentum`） | — | DFTB+ 由输入指定；ASE 和 PTBP 取“占据数不为 0 的最高 l” |
| Slater-Koster 旋转（键方向 → 实验室坐标） | `slako.py` | `dftb/sk.F90` | `rotateH0`, `ss/sp/sd/sf/pp/pd/pf/dd/df/ff` | DFTB+ 手写了每个 (l1,l2) 的公式（ff 一项就有 49 个表达式）；pydftb 用无迹对称张量构造实球谐旋转矩阵（d 二阶、f 三阶，M2g），统一处理，与 DFTB+ 的公式差 ≤ 2e-15 |
| 邻居表（含周期镜像） | `neighbors.py` | `dftb/periodic.F90`, `type/dynneighlist.F90` | `TNeighbourList`, `updateNeighbourListAndSpecies`, `getCellTranslations` | pydftb 用 k-d 树 |
| H0 和 S 的构建及其导数 | `hamiltonian.py: build_pairs` | `dftb/nonscc.F90` | `buildH0`, `buildS`, `getFirstDerivFiniteDiff` | **DFTB+ 用数值差分（中心差分，默认步长 1.22e-4，老版本输入为 0.01）**，pydftb 默认用复步长法（解析精度）；设 `fd_delta` 后可以逐位复现 DFTB+ 的力（差异 1e-12） |
| 稀疏存储 → 稠密 / k 点矩阵 | `hamiltonian.py: fold, fold_all` | `dftb/sparse2dense.F90` | `getSparseDescriptor`, `unpackHS`, `unpackHS_cmplx_kpts` | 相位因子是 exp(2πi k·T) |
| 短程 γ 函数 | `coulomb.py: _short, _gamma0, short_cutoff` | `dftb/shortgammafuncs.F90`, `dftb/shortgamma.F90` | `expGamma`, `expGammaPrime`, `expGammaCutoff`, `TShortGamma` | “两个 U 相等”的判据 \|ΔU\| < 3.125e-6 已与 DFTB+ 一致；每对 U 值在 S < 1e-10 处截断（`expGammaCutoff`，M2e 起第二阶和第三阶共用，见坑 19） |
| 长程 1/R（Ewald 求和） | `coulomb.py: Coulomb` | `dftb/coulomb.F90` | `TCoulomb`, `ewaldReal`, `ewaldReciprocal`, `getOptimalAlphaEwald` | — |
| SCC 势加入哈密顿量 | `calculator.py: DFTB._scc` | `dftb/hamiltonian.F90`, `dftb/shift.F90`, `dftb/scc.F90` | `getSccHamiltonian`, `addShift`, `TScc` | H = H0 + ½S(V_μ+V_ν) |
| Mulliken 布居 | `calculator.py` | `dftb/populations.F90`, `dftb/charges.F90` | `mulliken` | — |
| 轨道分辨 SCC | `calculator.py: _sites` | `dftb/uniquehubbard.F90`, `dftb/orbitalequiv.F90` | `TUniqueHubbard` | — |
| 电荷混合 | `mixer.py: BroydenMixer`（默认），`AndersonMixer`（Pulay 型） | `mixer/broydenmixer.F90` | `modifiedBroydenMixing` | 照搬 DFTB+ 的改进 Broyden（M1）：19 个对照算例的 SCC 轮数与 DFTB+ 完全相同 |
| 占据数与费米能级 | `occupations.py` | `dftb/etemp.F90`，`dftbplus/main.F90:getFillingsAndBandEnergies` | `Efilling`, `electronCount`, `electronFill`, `middleGap` | M1 已照搬：先试带隙中点，再按 DFTB+ 的方式二分；T 不低于 1e-8 Ha；Gaussian 即 MP 0 阶；另提供变分熵。玻尔兹曼常数与 DFTB+ 25.1 相同（M2e，坑 17）；带隙中点的搜索允许权重累加的舍入（M2e，坑 18） |
| 本征求解与密度矩阵 | `solvers.py` | `math/eigensolver.F90`, `dftb/densitymatrix.F90` | `hegvd`, `getDensityMatrix`, `getEDensityMatrix` | pydftb 另有 SP2 纯化和批量求解器 |
| 同一套计算在 PyTorch 上（M4a） | `backend/torch_solver.py`、`backend/spec.py` | （无对应：DFTB+ 的 GPU 版用 MAGMA / ELPA-GPU） | — | Cholesky + 三角求解化为标准问题，与 `hegvd` 同一算法；74 个单点、11 个优化/MD 参考在 torch 后端上用同样的容差通过，SCC 轮数与 DFTB+ 相同 |
| 力 | `calculator.py: DFTB._gradient` | `dftb/forces.F90`, `dftbplus/main.F90` | `derivative_shift`, `getGradients` | 电荷项和势用最后一轮的输出电荷（M2f 起，见坑 22） |
| DFTB3 三阶项（M2b） | `thirdorder.py: ThirdOrder`, `dgamma_du`, `dgamma_du_onsite` | `dftb/thirdorder.F90`，`dftb/chargepenalty.F90`（`ThirdOrder`） | `gamma3`, `gamma2pU`, `gamma2pUpR`, `dGdUr0`, `updateCharges`, `getShifts`, `addGradientDc`, `addStressDc` | U 几乎相等时 pydftb 改用稳定的积分形式（见第 3 节第 6 条） |
| H–X 阻尼（M2b） | `coulomb.py: Coulomb._sp_short`，`thirdorder.py: dgamma_du` | `dftb/shortgammafuncs.F90`，`dftbplus/initprogram.F90` | `expGammaDamped`, `expGammaDampedPrime`, `hh`, `hpU`；被阻尼的是质量 < 3.5 amu 的原子 | — |
| 共线自旋（M2c） | `calculator.py`：`_Spin`、`_spin_matrix`、`_potentials`、`_solve`、`_populations` | `dftb/spin.F90`，`dftbplus/main.F90`（`getFillingsAndBandEnergies`、`getNextInputCharges`、`reduceCharges`），`dftbplus/initprogram.F90`（`initElectronNumber`、`initializeCharges`） | `getSpinShift`, `getEnergySpin_*`, `Spin_getOrbitalEquiv`, `qm2ud`, `Efilling` | 两个自旋通道沿 k 点方向堆叠；混合向量与 DFTB+ 相同（↑/↓、按壳层） |
| 应力（M2a） | `calculator.py: DFTB._gradient(virial=True)`，`coulomb.py: Coulomb.gradient(virial=True)` | `dftb/stress.F90`，`dftb/coulomb.F90`，`dftbplus/main.F90` | `getBlockStress`, `invRStress`, `getStress` | 每个原子对的“力 × 键矢量”求和，加 Ewald 倒空间项和背景项；与力在同一遍里算出 |
| SCC 主循环 | `calculator.py: DFTB.calculate` | `dftbplus/main.F90` | `processGeometry`, SCC 循环 `lpSCC` | pydftb 分成 `_setup/_scc/_energies/_gradient` 四段 |
| 几何步之间的电荷（M2f） | `ase_calc.py: PyDFTB(reuse_charges=True)` | `dftbplus/main.F90`（每个几何步重置混合器、保留 `qInput`），`dftbplus/initprogram.F90:getMinSccIters` | `chrgMixerReal%reset` | 默认关闭：ASE + DFTB+ 每次计算是一个新进程，从中性原子开始 |
| 结构优化（M2f，只对照，用 ASE 的优化器） | `validation/dynamics.py` | `dftbplus/input/geoopt.F90`，`geoopt/`（`lbfgs.F90`、`rationalfunc.F90`、`fire.F90`、`filter.F90`） | `readGeoOptInput`, `TFilter%transformDerivative` | 收敛：最大梯度分量 < GradElem 且能量没有升高（坑 24） |
| 速度 Verlet MD（M2f，只对照，用 ASE 的积分器） | `validation/dynamics.py`（`verlet_dftbplus_units` 是同一算法、DFTB+ 的常数） | `md/velocityverlet.F90`，`dftbplus/parser.F90`（VelocityVerlet 输入） | `VelocityVerlet_next` | 更新顺序与 ASE 相同；`KeepStationary` 默认 Yes（坑 23） |
| 冻结电荷求本征态，即 DFTB+ 的能带运行（M2e） | `bands.py: eigenstates`（复用 `_setup` 和 `_potentials`） | `dftbplus/parser.F90`（`getEuclideanKSampling` 的 Klines 分支、`maxSelfConsIterations`），`dftbplus/main.F90`（`getNextInputCharges`、`writeCharges`） | `readKPoints`, `writeCharges` | `charges.bin` 存的是最后一轮的输入电荷 `qInput`（收敛后不再混合），对应 `DFTBResult.q_in`；Klines 的规则见坑 16 |
| 投影态 / 区域（M2e） | `bands.py: States.orbitals, weights` | `dftbplus/initprogram.F90`（区域与文件名），`dftbplus/mainio.F90` | `writeProjCplxEigvecsSerial`, `writeProjEigvecHeader`, `writeProjEigvecData` | 权重 Re[c*·(S c)]，不乘占据数和 k 权重；文件名 `标签.out`、`标签.壳层.out`、`标签.壳层.m.out` |
| DOS/PDOS 曲线（M2e） | `bands.py: broaden, States.dos, States.pdos` | dptools `scripts/dp_dos.py`（Python 工具，不在二进制里） | — | Gaussian、Fermi、Methfessel-Paxton；网格起点向下取整到网格间隔的整数倍 |
| 能量记账（Total / Mermin / Extrapolated） | `calculator.py: DFTBResult` | `dftb/getenergies.F90`, `dftbplus/mainio.F90` | `writeDetailedOut*` | — |
| 色散：Lennard-Jones/UFF（M2d） | `dispersion.py: LennardJones` | `dftb/dispuff.F90`，`dftb/dispuffdata.F90` | `DispUff_init`, `getDispEnergyAndGrad_cluster`, `getUffValues` | R ≤ 2^(−1/6)r 时用多项式“墙” |
| 色散：Slater-Kirkwood（M2d） | `dispersion.py: SlaterKirkwood` | `dftb/dispslaterkirkw.F90`，`dftbplus/parser.F90:readDispSlaKirk` | `addDispEnergyAndGrad_cluster`, `getDampCutoff_`；`HybridDependentPol` 的邻居计数在 parser 中 | — |
| 色散：r⁻⁶ 的 Ewald 求和（M2d） | `dispersion.py: ewald_r6` | `dftb/dispcommon.F90` | `addDispEGr_per_atom/_per_species`, `getOptimalEta`, `getMaxRDispersion`, `getMaxGDispersion` | η 和截断照 DFTB+ 选取 |
| 色散：D3、SimpleDftD3、D4（M2d） | `dispersion.py: DFTD3, SimpleDFTD3, DFTD4`（调用 `dftd3`/`dftd4` 库） | `extlibs/sdftd3.F90`（库），`dftb/simpledftd3.F90`（原生），`dftb/dispdftd4.F90`、`dftd4param.F90`、`encharges.F90`、`coordnumber.F90`（原生） | `TSDFTD3_init`, `addHHRepulsion`, `threeBodyDispersionGradient`, `weightReferences`, `addEwaldContribs` | DFTB+ 的 D4 是原生实现，与参考库有三处不同（第 3 节第 10–12 条） |

## 2. 待实现部分对应的 DFTB+ 文件（路线图见 `ROADMAP.md`）

| 功能 | DFTB+ 源文件 |
|---|---|
| 非共线自旋、自旋轨道耦合 | `dftb/spinorbit.F90`，`dftb/sparse2dense.F90`（`unpackHPauli`） |
| DFTB+U、onsite 修正 | `dftb/dftbplusu.F90`，`dftb/onscorrection.F90` |
| 长程校正 LC/CAM | `dftb/hybridxc.F90`（7,459 行），`dftb/rshgamma.F90` |
| 色散 TS/MBD、D4 自洽电荷 | `dftb/dispmbd.F90`（libmbd），`dftb/dispdftd4.F90`（`selfConsistent` 分支） |
| 氢键 / 卤素修正 | `dftb/h5correction.F90`，`dftb/halogenx.F90` |
| 溶剂 | `solvation/born.F90`，`cosmo.F90`，`sasa.F90`，`cm5.F90` |
| 外场、外部点电荷 | `dftb/extfields.F90`，`dftb/extcharges.F90` |
| 热浴、压浴、XL-BOMD | `md/thermostats.F90`，`md/nhctherm.F90`，`md/xlbomd.F90`（XL-BOMD 在 M6） |
| 激发态（Casida / 实时） | `timedep/linresp.F90`，`timedep/linrespgrad.F90`，`timedep/timeprop.F90` |
| REKS、pp-RPA | `reks/`，`timedep/pprpa.F90` |
| 输运（NEGF） | `transport/negfint.F90`（底层是 libNEGF） |
| 并行与求解器框架 | `common/mpienv.F90`，`common/blacsenv.F90`，`type/parallelks.F90`（k 点/自旋分组），`elecsolvers/`，`extlibs/elsiiface.F90`。pydftb/PTBP 的对应部分（M3）只做单机：PTBP 把结构分给 fork 出来的进程（`utils/parallel/`，相当于 DFTB+ 外面再套一层任务分发），pydftb 把一个结构的 k 点分给线程（`pydftb/parallel.py`，相当于 `parallelks` 的 k 点分组，但在一个进程内）；进程数和线程数由成本模型决定（`pydftb/perfmodel.py`、`utils/recommender.py`，M3c）；没有 MPI、BLACS |
| 输入解析与初始化 | `io/hsdparser.F90`，`dftbplus/parser.F90`（8,685 行），`dftbplus/initprogram.F90`（7,606 行） |

## 3. 对照时踩过的坑

标有【顾问 Xnn】的条目已做成 DFTB+ 配置顾问的规则（`python -m pydftb.advisor`，M3d）。

1. **ParserVersion。**【顾问 P01、P02】 解析器版本低于 14 时，DFTB+ 会把 `MethfesselPaxton { Order = n }` 悄悄改成 n−1（`dftbplus/oldcompat.F90`），而 ASE 写的是版本 1。pydftb 的对照工具固定写 14。
2. **数值求导方式。** DFTB+ 的 `Differentiation = Richardson {}` 在这些 SKF 上反而比默认差分更不准：水分子的力误差从 1.2e-6 升到 1.2e-5 Ha/Bohr。原因可能是它的步长较大，跨过了插值多项式更换窗口的格点。因此参考数据使用默认差分。
3. **SKF 表格少用一行。** 表头写 `0.02, 719`、文件里也有 719 行时，DFTB+ 只读前 718 行（`type/oldskdata.F90:145`，`nGrid = nGrid - 1`）。表格的终点因此是 14.36 Bohr，而不是 14.38 Bohr，最后几个插值窗口和其后 1 Bohr 的衰减尾巴都随之改变。Cu、H、O、C 的积分在这么远处已经极小，看不出差别；Li 的 2s 轨道很弥散，Li₂ 在 14.5–14.9 Bohr 处的能量差达到 2.5e-7 Ha，力甚至反号。M2a 起 pydftb 与 DFTB+ 一致（差异约 1e-11 Ha）。
4. **数值求导跨过网格节点。** DFTB+ 用步长 1.22e-4 Bohr 的中心差分求 H0、S 的导数。如果某个原子间距离到最近网格节点的距离比这个步长还小，差分的两端会落在两个不同的插值多项式上，导数就有误差。在晶格常数 3.45 Å 的 Li bcc 中，a√2 那一层邻居离第 461 个节点只有 4.3e-5 Bohr，DFTB+ 的应力因此偏差 1.2e-8 Ha/Bohr³。pydftb 默认用复步长精确求导，没有这个问题；设 `fd_delta` 后能逐位复现 DFTB+ 的结果（差异 6e-13）。第 2 条中 Richardson 求导变差，很可能也是这个原因。
5. **Γ 在 U 恰好相等时大一倍。** DFTB+ 的 `gamma2pU` 在 |U_a − U_b| < 3.125e-6 时对共同的 τ 求全导数，否则求偏导；按对称性，偏导恰好是全导数的一半。3ob 等参数按这个约定拟合，pydftb 照搬。
6. **U 几乎相等时，DFTB+ 的 γ 闭式公式失效。**【顾问 H01】 公式里 1/(τ_a² − τ_b²)ⁿ 各项彼此几乎完全抵消。|ΔU| = 1e-5 时，S 的相对误差是 6e-4，∂γ/∂U 的相对误差是 24。DFTB3 下实测总能量错 0.5 eV（ΔU = 4e-6）。pydftb 在 U 相差不到 5% 时改用 Hermite–Genocchi 积分形式，误差 ≤ 4e-12（见 `reports/M2b-dftb3.html`）。
7. **费米能级捷径影响力的自洽。** “带隙中点”只要让电子数误差 ≤ 1e-7 就被接受；电子数不严格守恒时，力与能量导数会相差约 5e-7 Ha/Bohr（实例：按壳层分辨的 Cu₄CO，300 K）。
8. **开自旋时第一轮不混合。** `main.F90:getNextInputCharges` 在第一次 SCC 迭代、且开了自旋（或 DFTB+U 等）时，直接用输出电荷作为下一轮输入，不调用混合器。不照做，SCC 轮数就对不上（O₂：2 轮对 4 轮）。
9. **“能级全部占满”按“能级数 × 自旋通道数”判断**（`etemp.F90:Efilling`）。把自旋通道当作额外的 k 点处理时，要乘上通道数，否则会误判。
10. **DFTB+ 的 D4 对 Cu、Zn、Ag、Cd、Au、Hg 多用一个参考体系。** `external/dftd4refs/dftd4_references.fh` 中这 6 种元素的 `refn` 为 3，dftd4 3.6 为 2（第 3 个参考体系仍在数据里，但未启用）；其余 3366 项数据完全相同。Cu₄CO 的色散能因此差 2.1e-3 Ha。
11. **DFTB+ 周期体系的 EEQ 电荷有错。**【顾问 D02】 结果随 `EwaldParameter` 变化，大盒子也不收敛到分子极限（按 1/L 衰减）。可疑处：`dftb/encharges.F90:addEwaldContribs` 中对角元加了两次倒空间项。对照时设 `ChargeScale = 0`。
12. **DFTB+ 周期体系的 D4 三体项计数有错。**【顾问 D03】 近邻表只存“像原子编号 ≥ 本原子”的近邻，`tripleScale` 的折扣与实际枚举次数不符；梯度不乘这个折扣。同一晶体用原胞和超胞算出的每胞能量不同（差 10%）。对照时设 `s9 = 0`。
13. **dftd4 库的截断写死为 60/40/30 Bohr**（二体/三体/配位数），DFTB+ 默认 64/40/40。对照时 DFTB+ 写成库的值；稠密周期体系中二体截断的差别约 1e-7 Ha/原子。
14. **SlaterKirkwood：被跳过的原子不能同时 N = 0。** DFTB+ 只检查原子对中的第一个原子，第二个原子的 √(α/N) 会算成 0/0，总能量变成 NaN。
15. **UFF 参数按物种名的前两个字母查找**（`dispuffdata.F90:getUffValues`）：`Li1` 会查成 Li，`O2` 查不到。对照工具对类型标签写显式数值。
16. **Klines 的规则与能带运行。**【顾问 B01】 第一行之前隐含一个 Γ 点；每行“N k”取从上一个点（不含）到 k（含）的 N 个等距点；N = 0 表示跳到 k 而不取点。有 Klines 时 `MaxSCCIterations` 默认为 1（`parser.F90:maxSelfConsIterations`）。只算一轮时多原子晶胞会被判定“SCC 未收敛”并报错退出，要写 `ConvergentSCCOnly = No`；单原子晶胞的原子布居恒等于电子数，SCC 误差恒为 0，不写也能跑。
17. **玻尔兹曼常数。** DFTB+ 25.1 是 3.16681534524639e-6 Ha/K（`common/constants.F90`），pydftb 在 M2e 之前用 3.166811429e-6，相对差 1.2e-6，在 500–1000 K 时造成约 3e-9 Ha、3e-9 e 的差别。核实方法：同一温度分别以 K 和以 Hartree 输入，结果必须逐位相同。
18. **middleGap 在舍入时跳过能隙。** `etemp.F90:middleGap` 逐个累加 k 点权重直到不小于填充数；权重为 1/13 之类时累加和比填充数少一个舍入单位，于是多加一个导带能级，“中点”落在导带内；之后的二分法把绝缘体的费米能级推到导带底下方约 28 kT。占据数不变。pydftb 累加时允许 1e-9 的舍入，取真正的中点。
19. **短程 γ 的截断。** DFTB+ 对每一对 U 值在 S 降到 1e-10 处截断（`shortgammafuncs.F90:expGammaCutoff`：先倍增再二分，二分精度也是 1e-10）。pydftb 的第二阶项在 M2e 之前一直加到 1e-14，周期体系中每个原子的 SCC 势因此差约 1e-9 Ha（能量差 1.6e-9 Ha，本征值差 ~1e-9 Ha）。
20. **`WriteRealHS` 可以用来对照势。** 它在第一轮 SCC 建好 H 后写出实空间 H、S 并退出（返回码非 0）。配合 `ReadInitialCharges` 读回收敛电荷时，on-site 块的对角元减去 on-site 能就是每个原子的 SCC 势。
21. **输出文件的精度。** `band.out` 的本征值只有 4 位小数（eV），区域文件 6 位小数、k 权重 8 位小数，`results.tag` 是全精度。对照本征值一律读 `results.tag`；需要 `WriteChargesAsText = Yes`（在 `Options` 中）才能读到全精度的 `charges.dat`。
22. **力用输出电荷。** `RecomputeAfterDensity` 默认开启：每次对角化之后，DFTB+ 用这一轮的输出电荷重算 SCC 势（`main.F90:processPotentials`），所以力中的库仑项（`addForceDc`）和势项（`derivative_shift` 用的 `potential%intBlock`）都来自输出电荷。pydftb 在 M2f 之前用输入电荷：SCC 收敛到 1e-10 时看不出区别，收敛到 1e-5 时（PTBP 的设置）力差可达 8e-7 Ha/Bohr。
23. **MD 默认去掉质心运动。** `VelocityVerlet` 的 `KeepStationary` 默认 Yes，每一步去掉质心速度；ASE 的积分器不这样做。给定的初速度原样使用（不重新缩放）。
24. **新的 `GeometryOptimisation` 驱动。**【顾问 G02】 收敛需要最大梯度分量 < `GradElem`（默认 1e-4 Ha/Bohr）**并且**这一步能量没有升高；梯度范数、位移判据默认关闭。LBFGS 没有步长上限：从远离极小点的结构开始时，在软的势能面上步长会越来越大，直到某个结构上 SCC 发散、程序退出。Rational 稳定但收敛慢（甲醇到 1e-7：1918 步，LBFGS 45 步）。
25. **`LatticeOpt` 的 `Isotropic = Yes` 只适合正交晶胞。**【顾问 G01】 它把应力换成压强，但晶格的梯度是 p·L⁻ᵀ，对非正交晶胞（如 bcc 原胞）不平行于 L，LBFGS 的步长会剪切晶胞，而收敛判据只看压强，于是停在剪切过的晶胞上。
26. **MD 输出的精度。** `geo_end.xyz` 的坐标和速度是 `F16.8`（1e-8 Å、1e-8 Å/ps），`md.out` 的能量是 Hartree 的 10 位小数。
27. **单位常数。** DFTB+ 用自己的一组常数（Bohr 0.529177249 Å、Hartree 27.2113845 eV、au__fs 0.02418884326505、amu__au = 1/5.4857990945e-4），ASE 用 CODATA 2014，相对差约 1e-7。经 ASE 跑的 40 步 MD 因此与 DFTB+ 相差约 4e-8 Å；用 DFTB+ 的常数则相差 ≤ 1.5e-8 Å（xyz 的精度）。
28. **插值在网格节点处有折点。** `slakoeqgrid.F90` 的 8 点插值每过一个网格节点就换一组插值点，插值函数的导数在节点处跳变。表格光滑时跳变可忽略；PTBP 用 hotcent 生成的表格有约 1e-7 Ha 的数值噪声，跳变中位数 5e-6、最大 1e-3 Ha/Bohr。DFTB+ 默认用步长 1.22e-4 Bohr 的中心差分求 H0、S 的导数（`nonscc.F90`），原子对距离落在节点一个步长之内时差分跨过跳变，力的误差可达 1.3e-4 Ha/Bohr（水二聚体 MD 中 2% 的步）。pydftb 的复步长导数是插值函数在该点所在分段上的精确导数；要与 DFTB+ 逐位比较力，pydftb 用同样的差分（`fd_delta`）。
29. **扩展 SKF 格式（f 轨道）。** 第 1 行以 `@` 开头（DFTB+ 只看第一个字符），其余内容整体下移一行；同核文件的在位行是 13 个数（E、U、占据数各 4 个，按 f d p s，中间是 SPE）；积分表每行先 20 个 H 再 20 个 S，顺序 ff0–3、df0–2、dd0–2、pf0–1、pd0–1、pp0–1、sf0、sd0、sp0、ss0。简单格式的 10 个积分放进第 8、9、10、13、14、15、16、18、19、20 个位置（`iSKInterOld`），所以两种格式可以混用。把同核文件改成异核文件（类型副本）时要删的在位行，在扩展格式里是第 3 行。
30. **没有 Spline 段的 SKF。**【顾问 K01】 `readSplineRep` 找不到 `Spline` 就报错退出，只有多项式（哪怕全为 0）的文件要配 `PolynomialRepulsive = SetForAll { Yes }`。hotcent 直接写出的文件就是这样（M2g 的 Gd/O 测试参数）。
31. **两处输入写法。** `HubbardDerivs` 的每个元素要单独一行，写在同一行会报 “Superfluous data”；壳层分辨 SCC 时不能写 `SpinConstants` 中的 `ShellResolvedSpin`（被忽略的节点会让 DFTB+ 报错退出）。
32. **OpenMP 线程在小矩阵上更慢。**【顾问 T03】 不设 `OMP_NUM_THREADS` 时 DFTB+ 用满全部核心。M3c 实测（Cu 晶胞，1000 K）：36 个轨道（864 个 k 点）用 4 个线程慢 3.3 倍，288 个轨道慢 1.2–1.3 倍，576 个轨道的 Γ 点不变快，576 个轨道 16 个 k 点快 1.4 倍、972 个轨道 4 个 k 点快 1.74 倍；PTBP 的 Cu 数据集（9–18 个轨道）整体慢 11%（M3a）。线程数还会改变应力的最后一位。PTBP 只在每个结构都有 ≥ 500 个轨道时给 DFTB+ 线程，其余情况每个进程 1 个线程。
33. **库仑求和的内存。** 壳层分辨 SCC 的短程 γ 算到 1e-10，截断由最小的 Hubbard U 决定（Cu 的 4p 壳层 U = 0.08 Ha 时 96 Bohr）。64 个 Cu 原子的晶胞有约 300 万个原子对：DFTB+ 峰值内存 0.3 GB，pydftb 把所有原子对和壳层对展开成数组，要 2.2 GB（M3c）。
