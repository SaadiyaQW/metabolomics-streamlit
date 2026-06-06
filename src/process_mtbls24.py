import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================
# 1. 加载数据，检查列结构
# ============================================
print("=" * 60)
print("1. 加载 MTBLS24 数据")
print("=" * 60)

file_path = "data/m_MTBLS24_helena_metabolite_profiling_NMR_spectroscopy_15_10_2012_v2_maf.tsv"
df_raw = pd.read_csv(file_path, sep='\t')

print(f"原始数据形状: {df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")
print(f"\n前5行:")
print(df_raw.head())

# 查看所有列名，找到数据列的起始位置
print(f"\n所有列名（前20个）:")
print(list(df_raw.columns[:20]))

# ============================================
# 2. 定位数据列：找到第一个全数值的列
# ============================================
print("\n" + "=" * 60)
print("2. 定位数值数据列")
print("=" * 60)

# 尝试将每一列转换为数值，看哪一列开始能成功
sample_cols = []
for col in df_raw.columns:
    # 尝试转换为数值，忽略错误
    try:
        pd.to_numeric(df_raw[col])
        sample_cols.append(col)
    except:
        print(f"跳过非数值列: {col}")

print(f"\n找到 {len(sample_cols)} 个数值列（样品列）")
print(f"样品列名（前10个）: {sample_cols[:10]}")

# 从原始数据中只提取这些数值列
df_numeric = df_raw[sample_cols].copy()

# 提取代谢物名称（使用第一列作为代谢物标识）
metabolite_names = df_raw.iloc[:, 0].copy()
print(f"\n代谢物数量: {len(metabolite_names)}")
print(f"样品数量: {df_numeric.shape[1]}")

# ============================================
# 3. 转置：样品作为行，代谢物作为列
# ============================================
print("\n" + "=" * 60)
print("3. 转置数据")
print("=" * 60)

# 转置
df_transposed = df_numeric.T
df_transposed.columns = metabolite_names

print(f"转置后形状: {df_transposed.shape[0]} 样品 × {df_transposed.shape[1]} 代谢物")
print(f"样品名（前10个）: {list(df_transposed.index[:10])}")

# ============================================
# 4. 数据清洗：缺失值处理
# ============================================
print("\n" + "=" * 60)
print("4. 缺失值处理")
print("=" * 60)

# 检查缺失值
missing_ratio = df_transposed.isnull().sum().sum() / (df_transposed.shape[0] * df_transposed.shape[1])
print(f"缺失值比例: {missing_ratio:.2%}")

if missing_ratio > 0:
    # 代谢组学常用：用最小值的一半填充
    # 只对数值列计算最小值
    numeric_cols = df_transposed.select_dtypes(include=[np.number]).columns
    min_values = df_transposed[numeric_cols].min()
    for col in numeric_cols:
        df_transposed[col] = df_transposed[col].fillna(min_values[col] / 2)
    print("已用最小值/2填充缺失值")

# 确保所有数据都是数值类型
df_transposed = df_transposed.apply(pd.to_numeric, errors='coerce')
df_transposed = df_transposed.fillna(df_transposed.min() / 2)

print(f"清洗后形状: {df_transposed.shape}")
print(f"清洗后数据类型: {df_transposed.dtypes.value_counts()}")

# ============================================
# 5. 数据标准化
# ============================================
print("\n" + "=" * 60)
print("5. 数据标准化 (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_transposed),
    index=df_transposed.index,
    columns=df_transposed.columns
)
print(f"标准化后数据形状: {df_scaled.shape}")

# ============================================
# 6. PCA 分析
# ============================================
print("\n" + "=" * 60)
print("6. PCA 分析")
print("=" * 60)

pca = PCA(n_components=5)
pca_result = pca.fit_transform(df_scaled)

# 解释方差比例
explained_variance = pca.explained_variance_ratio_
print(f"PC1 解释方差: {explained_variance[0]:.2%}")
print(f"PC2 解释方差: {explained_variance[1]:.2%}")
print(f"前5个主成分累计解释方差: {explained_variance[:5].sum():.2%}")

# ============================================
# 7. 可视化
# ============================================
print("\n" + "=" * 60)
print("7. 生成 PCA 图")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：散点图
scatter = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=range(len(pca_result)), cmap='viridis', alpha=0.7)
axes[0].set_xlabel(f'PC1 ({explained_variance[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({explained_variance[1]:.1%})')
axes[0].set_title('PCA: 样品分布')
plt.colorbar(scatter, ax=axes[0], label='样品索引')

# 图2：碎石图
axes[1].bar(range(1, 6), explained_variance[:5] * 100)
axes[1].set_xlabel('主成分')
axes[1].set_ylabel('解释方差百分比 (%)')
axes[1].set_title('碎石图')
axes[1].set_xticks(range(1, 6))

plt.tight_layout()
plt.savefig('output/pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("PCA 图已保存到 output/pca_analysis.png")

# ============================================
# 8. 保存处理后的数据
# ============================================
print("\n" + "=" * 60)
print("8. 保存处理后数据")
print("=" * 60)

# 保存清洗后的数据
df_transposed.to_csv('data/processed_metabolomics_data.csv')
print("已保存: data/processed_metabolomics_data.csv")

# 保存标准化后的数据
df_scaled.to_csv('data/scaled_metabolomics_data.csv')
print("已保存: data/scaled_metabolomics_data.csv")

print("\n" + "=" * 60)
print("✅ 数据处理完成！")
print("=" * 60)
