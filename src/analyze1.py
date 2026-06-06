import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（解决图表中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 1. 加载数据
# ============================================
df = pd.read_csv('data/metabolomics_ready.csv', index_col=0)
print(f"数据形状: {df.shape}")
print(f"样品数: {df.shape[0]}, 代谢物数: {df.shape[1]}")

# ============================================
# 2. 缺失值处理（如果有）
# ============================================
if df.isnull().sum().sum() > 0:
    print(f"发现缺失值，用0填充")
    df = df.fillna(0)

# ============================================
# 3. 数据标准化
# ============================================
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    index=df.index,
    columns=df.columns
)

# ============================================
# 4. PCA分析
# ============================================
pca = PCA(n_components=5)
pca_result = pca.fit_transform(df_scaled)

# 解释方差
exp_var = pca.explained_variance_ratio_
print(f"\nPCA解释方差:")
print(f"PC1: {exp_var[0]:.2%}")
print(f"PC2: {exp_var[1]:.2%}")
print(f"PC3: {exp_var[2]:.2%}")
print(f"前5主成分累计: {exp_var[:5].sum():.2%}")

# 绘制PCA图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 散点图
scatter = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=range(len(pca_result)), cmap='viridis', alpha=0.7)
axes[0].set_xlabel(f'PC1 ({exp_var[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({exp_var[1]:.1%})')
axes[0].set_title('PCA: 样品分布')
plt.colorbar(scatter, ax=axes[0], label='样品索引')

# 碎石图
axes[1].bar(range(1, 6), exp_var[:5] * 100)
axes[1].set_xlabel('主成分')
axes[1].set_ylabel('解释方差百分比 (%)')
axes[1].set_title('碎石图')
axes[1].set_xticks(range(1, 6))

plt.tight_layout()
plt.savefig('output/pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 5. 如果样品名包含分组信息，自动提取分组
# ============================================
# 从样品名中提取分组（例如：AKI_8_24_01_110722 → 提取数字部分）
df['Group'] = df.index.str.extract(r'(\d+_\d+_\d+)')[0]  # 提取 8_24_01 这样的模式
print(f"\n自动提取的分组:\n{df['Group'].value_counts()}")

# 保存结果
df.to_csv('data/metabolomics_with_group.csv')
print("\n✅ 已保存: data/metabolomics_with_group.csv")