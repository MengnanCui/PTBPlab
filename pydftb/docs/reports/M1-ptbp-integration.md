# M1 完成报告：pydftb 接入 PTBP

## 一句话结论

`ptbp optimize ... --engine pydftb` 已经可以用。默认设置下，它与 PTBP 实际调用的 DFTB+ 算得**一样**：PTBP 损失函数的相对差在 1e-7 以内，66 个结构的能量差不超过 6e-7 eV。DFTB 阶段还**快 1.8 倍**（104 s 对 187 s）。顺带查清了 PTBP 调用 DFTB+ 时几处“写的和实际算的不一样”的地方。

## 一、最重要的发现：PTBP 实际让 DFTB+ 算了什么

PTBP 通过 ASE 调用 DFTB+。ASE 写的输入带有 `ParserVersion = 1`，DFTB+ 因此按“老版本兼容规则”悄悄改了几处设置。这几处从 DFTB+ 处理后的输入 `dftb_pin.hsd` 和它的源码中都可以核实：

| 设置 | PTBP 代码里写的 | DFTB+ 实际用的 |
|---|---|---|
| 电子温度 | 2000 K、MP 2 阶 | **T = 0**（Fermi）。关键字放在 `Filling` 之外，被静默忽略 |
| SCC | 未写 | **按壳层分辨**：s、p、d 各用自己的 Hubbard U（`oldcompat.F90:182`） |
| 力的求导 | 未写 | **有限差分，步长 0.01 Bohr**，是现在默认步长的 80 倍（`oldcompat.F90:368`） |
| 排斥能 | `PolynomialRepulsive = Yes` | 多项式系数全为 0，所以**排斥能为 0**，之后由 physrep 在 Python 里另算 |

**影响：** 已发表的 PTBP 参数是在“按壳层分辨 SCC、T = 0、力用粗差分”的条件下拟合的。其中第二条最容易被忽略，所有通过 ASE 用 DFTB+ 的人都会碰到。

**处理：**
- 默认完全保持现状，保证可复现。
- 新增 `--smearing` 开关，让展宽**真正生效**。DFTB+ 一侧改为正确写入 `Filling`；因为 `ParserVersion = 1` 时 DFTB+ 会把 MP 阶数减 1，写入的是阶数 +1。每次计算后还会读取 DFTB+ 实际使用的展宽，不符就报错。已在 MP、Fermi、Gaussian 三种方式上验证：两个引擎的能量差约 1e-9 eV，力差约 3e-6 eV/Å。

## 二、完成的内容

### pydftb 侧（第一部分）
- 展宽：Fermi / Gaussian / Methfessel-Paxton，照搬 DFTB+ 的费米能级搜索和熵（TS）约定。另提供“变分”熵：只有用它，MP 展宽下的 Mermin 能量才与力严格自洽。实测按 DFTB+ 的约定，力与能量导数相差 1.6e-4 Ha/Bohr；改用变分约定后只差 2e-9。
- 改进 Broyden 混合（DFTB+ 默认）：19 个对照算例中，**SCC 迭代轮数与 DFTB+ 完全相同**。
- 排斥势开关 `repulsive="polynomial"`，对应 DFTB+ 的 `PolynomialRepulsive`，用一套多项式系数不为零的 SKF 验证过。
- `fd_delta`：按 DFTB+ 的方式用有限差分求力。用上之后，与 DFTB+ 的力差从 1.2e-6 降到 1.2e-12 Ha/Bohr，**证明此前残余的差异全部来自 DFTB+ 的数值求导**。
- γ 函数改为按原子搜邻居、每对 U 各用自己的截断：PTBP 最大 k 网格的结构从 7.9 秒降到 2.0 秒（DFTB+ 为 3.9 秒）。
- `PyDFTB`（ASE 计算器）：PTBP 的 k 网格、显式 k 点、自动确定最高角动量、线程限制、类型标签，以及能带工具需要的访问器。

### PTBP 侧（第二部分）
- **`utils/engines.py`**：唯一的引擎分发点。`dftbplus_calculator` 保留原名，内部转交给它，DFTB+ 分支原有代码不变。
- **新增选项**：`--engine`、`--smearing`、`--smearing-temperature`、`--mp-order`、`--pydftb-forces`。命令行、YAML（`ptbp_run.yaml`）和老式命令行都已接通；TUI 向导第 6 步可以选择引擎。
- **测试**：`tests/test_engines.py` 共 9 个，覆盖引擎分发、两种 DFTB+ 写法、`Filling` 校验、命令行和 YAML 传递、TUI，以及在 `dft/dft.xyz` 上实际运行两个引擎的等价性测试（没有 dftb+ 时自动跳过）。
- **`tools/engine_parity.py`**：在固定参数下计算 PTBP 的真实损失函数，对比两个引擎（结果见第三节）。
- **文档**：README 新增“4b. DFTB engine”一节；`RESEARCH.md` 文首加了更正说明。

## 三、验证结果

### 3.1 与 DFTB+ 的 19 个对照算例（`refgen --check`）

全部通过。SCC 轮数全部一致；能量差 ≤ 1.3e-7 Ha（这一项来自 DFTB+ 默认的 Ewald 精度 1e-9，把它调到 1e-12 后差异降到 1.6e-9）。

### 3.2 PTBP 实际路径上逐个结构对比（`dft/dft.xyz`，默认设置）

| 结构 | k 网格 | 能量差 (eV) | 力差 (eV/Å) | 最大受力 (eV/Å) |
|---|---|---|---|---|
| 0（强压缩） | 3×4×4 | −2.1e-09 | 2.4e-06 | 7.75 |
| 7 | 3×3×3 | −3.3e-10 | 9.2e-09 | 0.53 |
| 20 | 8×8×5 | −1.2e-09 | 1.2e-07 | 0.01 |
| 65 | 14×14×14 | +2.0e-07 | 4.5e-16 | 0.00 |

如果 pydftb 用精确导数（`--pydftb-forces exact`），结构 0 的力差为 1.7e-2 eV/Å。这是 DFTB+ 0.01 Bohr 粗差分本身的误差，而不是 pydftb 的误差。

### 3.3 PTBP 损失函数（`tools/engine_parity.py`）

在三组参数和两种模式下，用两个引擎分别计算 PTBP 的真实损失函数。每次计算都走完整流程：hotcent 生成 SKF → 对全部 66 个结构做 DFTB → physrep 拟合排斥势 → 计算损失。

| 模式 | (r0_w, r0_d) | DFTB+ | pydftb | 相对差 | 单次耗时 DFTB+ / pydftb |
|---|---|---|---|---|---|
| dataset | (3.000, 6.000) | 53211.18153 | 53211.18205 | 9.8e-09 | 168 s / 125 s |
| dataset | (2.749, 12.036)（已发表的 Cu 参数） | 901629.832 | 901629.821 | 1.2e-08 | 163 s / 124 s |
| energygeometry | (2.749, 12.036) | 3837.92402 | 3837.92365 | 9.6e-08 | 178 s / 141 s |

**损失函数的相对差在 1e-7 以内，比验收目标 1e-4 好三个数量级以上。** 单次耗时中 hotcent 生成 SKF 占了很大一部分，这部分两边相同。

> 附带观察：在已发表的 Cu 参数下，dataset 模式的损失值高达 9e5，EOS 模式为 3.8e3，而论文中 Cu 的损失为 0.288。两个引擎给出的数值一致，说明这是 PTBP 自身流程的问题（可能与 E0 或惩罚项的设置有关）。这不在本阶段范围内，建议另行检查。

### 3.4 PTBP 的 DFTB 阶段耗时（`pydftb/bench/throughput.py`，真实设置）

把 `dft/dft.xyz` 的 66 个结构逐个交给 `utils.calculator.dftbplus_calculator`。设置与 PTBP 实际运行完全相同：PTBP 的 k 网格、`PolynomialRepulsive = Yes`、SCC 容差 1e-5、T = 0、按壳层分辨 SCC。

| 引擎 | 66 个结构总耗时 | 与 DFTB+ 的最大差异 |
|---|---|---|
| DFTB+（经 ASE） | 186.8 s | — |
| pydftb | **104.2 s（快 1.8 倍）** | 能量 5.6e-7 eV，力 1.1e-5 eV/Å |

更正：原型阶段报告里“两者打平（21 s 对 21.5 s）”的测试，用的不是 PTBP 的真实设置，既没有按壳层分辨 SCC，k 网格的取整方式也不同。在真实设置下，按壳层分辨 SCC 让两边都更慢，因为 Cu 的 p 壳层 U 很小，短程 γ 的截断很长。pydftb 已针对这一点做了优化（见第二节），所以领先。

### 3.5 端到端

```bash
ptbp optimize dft/dft.xyz --mode dataset --engine pydftb --optimizer pso --n-particles 2 --n-calls 4
```

完整跑通：4 次评估后得到 `result.pkl`、`results.tgz`、`convergence.png`。生成的 `ptbp_run.yaml` 中记录了 `engine: pydftb` 和展宽设置，可以用 `--config` 原样复现。

## 四、学到的东西

1. **“写了”不等于“生效了”。** ASE 默认的 `IgnoreUnprocessedNodes = Yes` 会吞掉写错位置的关键字。现在对展宽做了“运行后核对”。
2. **老版本兼容规则会悄悄改变物理设置。** 按壳层分辨的 SCC 和力的步长都是这样被改的。核对时一定要看 DFTB+ 处理后的输入 `dftb_pin.hsd`，而不是自己写进去的 `dftb_in.hsd`。
3. **对照到 1e-12 才算真正搞懂。** 把剩下的差异逐一归因（Ewald 精度、有限差分、SCC 收敛容差），等于把 DFTB+ 的这几块重新学了一遍。

## 五、下一步

- 按 M2 表格推进：应力、DFTB3、共线自旋、色散、能带/DOS、结构优化/MD、f 轨道。
- PTBP 吞吐的进一步提速：同一个 BO 步只解析一次 SKF、进程内批量处理（M4，需要 PyTorch 后端）。
