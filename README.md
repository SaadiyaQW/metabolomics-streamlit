# 代谢组学数据分析平台

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 项目简介

本项目基于非靶向代谢组学数据，构建了一套完整的自动化分析流程，包括数据清洗、PCA降维、差异代谢物筛选和可视化。项目旨在快速识别不同实验组之间的代谢差异，为生物标志物发现提供数据支持。

## 📊 数据来源

- **数据库**：MetaboLights (MTBLS24)
- **样品数量**：106 例
- **代谢物数量**：701 个
- **分组**：Group1 (35例)，Group2 (35例)，Group3 (36例)

## 🛠️ 技术栈

| 类别 | 工具 |
|------|------|
| 数据处理 | pandas, numpy |
| 机器学习 | scikit-learn |
| 可视化 | matplotlib, seaborn, plotly |
| Web界面 | streamlit |
| 环境管理 | conda |

## 💡 结论

### 主要发现

1. **代谢谱差异显著**  
   PCA分析显示，Group1、Group2、Group3三组样品的代谢谱存在显著差异，前三个主成分累计解释了36.2%的代谢变异。

2. **70个显著差异代谢物**  
   通过t检验筛选出70个代谢物（p < 0.05），这些代谢物在Group3中的丰度**全部显著低于**Group1。

3. **Top 5 差异最显著的代谢物**

| 代谢物 | P值 | 变化方向 |
|--------|-----|----------|
| M118 | 8.81 × 10⁻⁵ | ↓ 下调 |
| M612 | 1.12 × 10⁻⁴ | ↓ 下调 |
| M527 | 1.34 × 10⁻³ | ↓ 下调 |
| M505 | 1.36 × 10⁻³ | ↓ 下调 |
| M529 | 2.03 × 10⁻³ | ↓ 下调 |

4. **生物学意义**  
   70个代谢物全部下调提示Group3可能处于**代谢抑制状态**，涉及能量代谢、氨基酸代谢等通路的广泛下调。具体代谢物如M118、M612等可作为潜在的生物标志物进行后续靶向验证。

### 统计摘要

| 指标 | 数值 |
|------|------|
| 总样品数 | 106 |
| 总代谢物数 | 701 |
| 显著差异代谢物（p<0.05） | 70 |
| 上调代谢物 | 0 |
| 下调代谢物 | 70 |
| PCA累计解释方差（PC1-3） | 36.2% |

## 快速开始
```bash
git clone https://github.com/yourname/metabolomics_project.git
cd metabolomics_project
conda create -n metabolomics python=3.10
conda activate metabolomics
pip install -r requirements.txt
python analysis.py
streamlit run app.py
