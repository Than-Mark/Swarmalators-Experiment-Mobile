# Swarmalators-Experiment-Mobile

一个面向活性物质/Swarmalators 数值实验的多项目框架。仓库按课题拆分为多个子目录，共享一套基础建模模板、数据存储方式和后处理流程，支持：

- 二维/三维 swarmalator 模型仿真
- 周期边界、局域耦合、噪声、外驱动、障碍/路径等机制扩展
- 批量参数扫描（含多进程）
- HDF5 轨迹存储与 Notebook 可视化分析

## 1. 仓库结构

本仓库采用“公共库 + 多课题子工程”的组织方式：

- `swarmalatorlib/`
  - 通用模板与工具（核心基类在 `template.py`）
- `PJT_*`、`[PJT]*`
  - 各独立研究课题（每个目录通常有 `main.py` + `run*.py` + `*.ipynb`）
- `[Read]*`
  - 文献复现实验/阅读相关代码
- `MS_Thesis/`
  - 论文阶段的整合实验代码

典型子项目内部结构：

- `main.py`：定义模型类、动力学更新、数据记录、分析类
- `run.py` / `run*.py`：批量运行入口（参数扫描、多进程）
- `*.ipynb`：结果可视化、相图绘制、视频导出

## 2. 核心设计思想

### 2.1 基类驱动

多数模型继承自 `swarmalatorlib/template.py` 中的基类（如 `Swarmalators2D`、`Swarmalators`），常见重写内容包括：

- 相互作用核（空间/相位耦合项）
- `update()`（单步数值推进）
- `append()`（按快照间隔写入数据）
- `plot()`（状态可视化）
- `__str__()`（实验参数命名，决定输出文件名）

### 2.2 统一数据落盘

模型运行时通常通过 `pandas.HDFStore` 写入 `*.h5`，常见键包括：

- `positionX`
- `phaseTheta`
- 以及各模型特有中间量（如 `pointTheta`、`pointX`、`vecAnglePhi` 等）

这一设计使得同一模型可以被 Notebook/分析类重复读取，复用后处理代码。

### 2.3 快照机制

`shotsnaps` 参数用于控制存储间隔（每隔若干步写一次），在长期仿真时平衡存储开销与时间分辨率。

## 3. 环境准备

建议使用 Python 3.10+（3.9 也可尝试），并使用虚拟环境。

### 3.1 创建环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3.2 安装依赖

```bash
pip install numpy pandas matplotlib numba scipy seaborn tqdm imageio tables jupyter
```

说明：

- `tables` 用于 `pandas.HDFStore`（HDF5 读写）
- `scipy` 在部分项目中用于几何/邻域操作（如 Delaunay）
- 若需要导出 mp4，请安装 ffmpeg，并在代码中配置路径

## 4. 快速开始

### 4.1 运行单个课题

以某个子项目为例：

```bash
cd "[PJT_C] XXX"
python run.py
```

或直接在对应目录运行 `runSingle.py`、`runFreq.py`、`runNoise.py` 等脚本。

### 4.2 交互式分析

在项目目录中打开 Notebook（如 `main.ipynb` / `plotFigs.ipynb`），读取已保存的 `*.h5` 数据进行相图、序参量、轨迹可视化。

## 5. 数据与输出约定

- 输出文件主要为 `*.h5`、`*.mp4`、`*.png`、`*.csv`
- 文件名通常由模型 `__str__()` 生成，包含关键参数
- 建议将大规模数据目录配置为外部盘路径（例如脚本中的 `SAVE_PATH`）

仓库的 `.gitignore` 已忽略常见结果文件，避免大体量数据进入版本管理。

## 6. 新增一个模型的推荐流程

1. 在目标课题目录新建或修改 `main.py`，继承模板基类。
2. 明确并实现：
   - 状态变量初始化
   - 相互作用项/邻接关系
   - `update()` 数值推进
   - `append()` 数据记录
   - `__str__()` 参数化命名
3. 新建 `run*.py` 用于批量参数扫描。
4. 用 Notebook 做可视化和统计分析。

## 7. 性能建议

- 优先使用 `numba.njit` 包装高频计算核
- 降低快照频率（增大 `shotsnaps`）以减小 I/O 压力
- 参数扫描时结合 `multiprocessing.Pool`
- 大规模实验建议固定随机种子并记录参数网格，保证可复现

## 8. 常见问题

### Q1: 运行报 HDF5 相关错误？

确认安装了 `tables`（PyTables），并检查输出目录是否可写。

### Q2: 无法导出 mp4？

确认系统已安装 ffmpeg，并将 matplotlib 的 ffmpeg 路径设置为本机实际路径。

### Q3: 仿真太慢？

减少粒子数、缩短总步数、提高快照间隔，或先在小参数网格上调试。

## 9. 致谢

本仓库汇总了多个 swarmalator/活性物质方向的研究实验与复现实践，适合作为：

- 新模型快速原型平台
- 参数扫描与图像生产流水线
- 文献复现实验与方法比较基座
