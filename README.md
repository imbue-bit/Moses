<p align="center">
<img src="./assets/moses.png" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-GPLv3-blue?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Framework-C++17/20-green?style=for-the-badge" alt="Framework">
  <img src="https://img.shields.io/badge/AI_Engine-ONNX_Runtime-orange?style=for-the-badge" alt="AI">
</p>

---

> [!TIP]
> **什么是信号聚合？**
> 
> 在当今数据驱动的金融市场，单一维度的信号已无法满足复杂决策的需求。信号聚合是一种先进的数据融合策略，其核心价值在于将多个独立的、甚至弱相关的alpha因子，通过智能化的加权与集成，提炼成一个具有更高信息熵和更强预测能力的单一决策指令，带来更高的收益。

## 摘要

Moses 是一个基于在线学习算法的全资产量化交易信号聚合 ensemble 框架。本项目融合了 Multiplicative Weights Update 的在线学习理论、GAF 的计算机视觉特征提取，以及 PPO 的深度强化学习控制方法。Moses 通过将信号映射至标准度量空间，利用遗憾界约束动态优化权重分配，实现了跨资产类别（股票、期货、期权等）的自适应风险对冲与策略集成。

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
   python train/train_gaf.py
   python train/train_ppo.py
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
@software{Moses2025,
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
