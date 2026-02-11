<p align="center">
<img src="./assets/moses.png" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-GPLv3-blue?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Framework-C++17/20-green?style=for-the-badge" alt="Framework">
  <img src="https://img.shields.io/badge/AI_Engine-ONNX_Runtime-orange?style=for-the-badge" alt="AI">
</p>

---

## 摘要

Moses 是一个基于在线学习算法的全资产量化交易信号聚合 ensemble 框架。本项目融合了 Multiplicative Weights Update 的在线学习理论、GAF 的计算机视觉特征提取，以及 PPO 的深度强化学习控制方法。Moses 通过将信号映射至标准度量空间，利用遗憾界约束动态优化权重分配，实现了跨资产类别（股票、期货、期权等）的自适应风险对冲与策略集成。

---

## 数学构架

### 1. 信号编码与视觉嵌入
Moses 引入了时序图像化机制。对于任意给定的时间序列 $X = \{x_1, x_2, ..., x_n\}$，系统通过下式构建 GAF 矩阵：

$\cos(\phi_i) = \tilde{x}_i, \quad \tilde{x}_i \in [-1, 1]$

$G_{i,j} = \cos(\phi_i + \phi_j) = \tilde{x}_i \tilde{x}_j - \sqrt{1 - \tilde{x}_i^2} \sqrt{1 - \tilde{x}_j^2}$

该矩阵捕捉了价格动量中的非线性相位相关性，并由卷积神经网络转化为高维信号。

### 2. 在线聚合引擎
系统将每一路信号视为一个专家。权重的演化遵循指数级更新准则，以最小化预测误差的累积：

$W_{i,t+1} = W_{i,t} \cdot (1 - \eta \cdot L_{i,t})$

其中 $\eta$ 为学习率，$L_{i,t}$ 是基于实际市场回报 $R_{t}$ 定义的损失函数。该算法保证了系统在最坏情况下的性能界。

### 3. 元策略控制
利用 PPO 智能体作为超参数控制器，系统实时监测市场状态 $\mathcal{S}$（包括波动率熵、权重分布多样性等），并输出行动向量 $\mathcal{A} = \{\eta, \text{Scaling}\}$，动态调节聚合器的灵敏度与全局仓位暴露。

---

## Features

1. 多模态融合: 支持传统量价因子、基本面分数与 GAF 视觉因子的无缝集成。
2. 高性能并发: 核心引擎采用 C++17 编写，底层通过 Boost.Asio 实现异步 Socket 交易指令分发。
3. 量纲一致性: 内置自适应标准化模块，支持对不同频率和物理意义的资产信号进行 Z-Score 与 Tanh 压缩。
4. 结合了 PPO 动态仓位控制与基于预期缺口的硬性风控截断。

---

## Quick Start

### 编译环境要求
- **Compiler**: GCC 9.0+ / Clang 11.0+ (支持 C++17)
- **Dependencies**: Eigen3, Boost.Asio, ONNX Runtime, nlohmann-json

### 部署流程
1. 模型导出:
   ```bash
   python pipeline/train_gaf.py
   python pipeline/train_ppo.py
   ```
2. 内核构建:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j4
   ```
3. 服务启动:
   ```bash
   ./Moses_Server 8888 ./models/gaf.onnx ./models/ppo.onnx
   ```

---

## Citation

如果您在学术工作或实盘交易系统中参考了 Moses 的架构，请引用本项目：

```bibtex
@software{Moses2023,
  author = {Moses Contributors},
  title = {Moses: A Multi-modal Online Learning Framework for Signal Ensemble},
  year = {2025},
  url = {https://github.com/imbue-bit/Moses/}
}
```

---

<p align="center">
  <i>"The law of the wise is a fountain of life, to depart from the snares of death."</i> —— <b>Moses</b>
</p>
