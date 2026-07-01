# PTBP GUI 功能与使用指南

一个基于 **customtkinter** 的桌面图形界面（VSCode 深色风格），把 PTBP DFTB
参数化工具箱的全部功能集中到一个窗口里，并补充了工具箱原本缺失的**结构/EOS
训练集构建**能力。

## 启动

```bash
pip install -e '.[gui]'      # 安装 GUI 依赖 (customtkinter)
ptbp gui                     # 启动（或：python -m gui）
ptbp gui --project /path/dir # 指定工作目录
```

> Linux 上如果缺少系统 Tk，需要 `apt install python3-tk`。若未安装
> customtkinter，`ptbp gui` 会打印友好提示并退出，不会报栈。

## 架构（两层）

- **`gui/core/`** — 纯 Python、不依赖 Tk、可无显示环境单测：
  - `structure_tools.py` — 体积缩放、EOS 序列生成、位置微扰、数据集摘要（仅需 ASE）
  - `gui_state.py` — 复用 `cli.tui.state.SetupState` 的图形桥接层，生成 YAML/命令
  - `runner.py` — `JobRunner`：子进程运行 `ptbp`、流式输出、写单一合并日志
  - `env_doctor.py` — 依赖体检（ase/skopt/pyswarms/hotcent/pylibxc/DFTB+）
  - `logging_setup.py` — 统一日志 + 去 ANSI 色码 + 屏蔽 customtkinter 良性重绘噪音
- **Tk 层** — `theme.py` / `widgets/` / `app.py` / `views/`：主题、控件、主窗口、7 个视图

## 核心使用逻辑：预览 → 运行

优化和 SKF 生成默认**只预览**（生成 YAML + 等价命令），**运行**是显式按钮。
真正的重计算（DFTB+ / hotcent / pylibxc）只在装了这些 conda-only 依赖的机器上
才会实际计算；缺依赖时“计算器”视图会说明原因。

## 七个视图

### 1. Home（主页）
选择工作目录、快捷导航、列出最近的 `run_*` 运行文件夹（点击直接进监控）。

### 2. Structures（结构与 EOS 构建）
- **加载**数据库或单结构（xyz / traj / extxyz / json）
- **检查**：结构数、能量/力是否齐全、化学式、自动检测的模式、警告
- **2D 预览**：逐结构的 matplotlib 投影
- **体积调整 + EOS 训练集生成**：给定最小/最大体积百分比与点数，生成一系列体积
  缩放结构（按相分块连续写出，兼容 `run.py` 的 `eos_points` 取中心逻辑），保存为 xyz
- **位置微扰**：给定 sigma(Å)、种子、份数，生成 rattle 微扰结构并保存

### 3. Generate（SKF 生成）
- 元素符号、参数集（PTBP / QNplusRep / QUASINANO2013 / Prior / Personal）
- 部分（full / band / rep）、Personal 的 r0 与 rep 参数、XC 泛函、输出目录
- **预览命令** / **运行**（缺 hotcent/pylibxc 会提示）

### 4. Optimize（参数优化）
- **数据集**选择 + 自动检测目标
- **目标（损失）**：能量+几何(EOS) / 能量+力(dataset) / 反应 / 带隙(bandstructure)
- **优化器**：Bayesian(bo) / 并行 BO(parallel_bo) / 粒子群(pso) + n_calls / n_particles
- **参数**：r0_w / r0_d / sigma_rep 开关 + 叠加方式 density / potential
- **E0s** 策略：auto / cache / manual(JSON)
- **高级**：xc、k 点密度、eos_points、multi_element、seed、输出目录
- **预览 YAML+命令** / **运行 + 取消**；运行时把运行文件夹交给监控视图

### 5. Monitor（监控）
- 选择运行文件夹（默认接住刚启动的运行）
- **最优结果**：从 `summary.json` / `result.pkl` 读最优参数、损失、评估次数、耗时
- **收敛曲线**：优先从 `result.pkl` 的 `func_vals` 重绘 best-so-far，否则嵌入 `convergence.png`
- **产物浏览**：列出运行文件夹中的产物文件

### 6. Calculator（计算器/依赖）
- **ASE_DFTB_COMMAND** 设置（本会话生效，子进程继承）
- **依赖体检**：逐项 ✓/✗ 显示 ase/scipy/skopt/pyswarms/hotcent/pylibxc/DFTB+ 等
- 就绪度总览：结构 / SKF / 优化(bo) / 优化(pso)
- 默认计算参数说明（xc、k 点、叠加方式）

### 7. Logs（日志）
浏览 `gui_logs/gui_run_*.log`（每次运行一个**合并日志文件**：整洁的信息框头/尾 +
去色码的完整输出 + 产物清单），点击查看，或打开最新一条。

## 优化方法与参数速查

| 优化器 | 说明 | 库 |
|---|---|---|
| `bo` | 顺序高斯过程 BO，样本效率高 | scikit-optimize |
| `parallel_bo` | 批量提议 + fork 并行评估 | scikit-optimize |
| `pso` | 粒子群，适合多峰/非光滑 | pyswarms |

| 参数 | 含义 |
|---|---|
| `r0_w` | 波函数约束半径 |
| `r0_d` | 密度约束半径 |
| `sigma_rep` | 排斥势尺度（内层 brute + L-BFGS-B 优化） |
| `p` | 约束势指数 `V=(r_cov/r0)^p`（带隙模式优化） |

| 目标模式 | 损失 | 需要 |
|---|---|---|
| energygeometry | μ_E+μ_V+μ_B（EOS） | 旁边有 `fit.json` |
| dataset | (1−c)·μ_E + c·μ_F | E0s（可自动拟合缓存） |
| reaction | 同 dataset + 每化学式分解 CSV | E0s |
| bandstructure | HOMO/LUMO 失配 | k 路径 `.json` |

## 已知限制

- 无 DFTB+/hotcent/pylibxc 时无法实际计算（运行按钮已接线，缺依赖只会在日志里报错）
- 结构预览为 2D 投影，非交互式 3D
- `bandstructure` 优化路径在 `cli/run.py` 中有已知 bug（本轮未修）；界面已就绪，修复后即可用
