import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# 设置 matplotlib 字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出文件夹
import os
os.makedirs('output', exist_ok=True)

print("=" * 60)
print("代谢组学数据分析 Pipeline")
print("=" * 60)

# ============================================
# 1. 加载数据
# ============================================
print("\n1. 加载数据...")
df = pd.read_csv('data/metabolomics_ready.csv', index_col=0)
print(f"   ✓ 数据形状: {df.shape[0]} 样品 × {df.shape[1]} 代谢物")

# 如果列名是 Unnamed，重新命名为 M1, M2, ...
if df.columns[0].startswith('Unnamed'):
    df.columns = [f'M{i+1}' for i in range(df.shape[1])]
    print(f"   ✓ 代谢物已重命名为: {df.columns[:5].tolist()}...")

# ============================================
# 2. 缺失值处理
# ============================================
print("\n2. 缺失值处理...")
missing_count = df.isnull().sum().sum()
if missing_count > 0:
    df = df.fillna(df.min() / 2)
    print(f"   ✓ 填充了 {missing_count} 个缺失值")
else:
    print("   ✓ 无缺失值")

# ============================================
# 3. 提取分组信息（从样品名）
# ============================================
print("\n3. 提取分组信息...")

# 从样品名中提取时间点（如 8_24_01 中的 01 可能是样品编号）
# 根据命名规律，提取最后一段数字作为分组
df['Sample_Time'] = df.index.str.extract(r'(\d+_\d+_\d+_\d+)')[0]
df['Sample_Simple'] = df.index.str.extract(r'AKI_\d+_\d+_(\d+)_')[0]

# 创建简单的分组（按样品索引分成三组）
n_samples = len(df)
df['Group'] = 'Group1'
df.iloc[n_samples//3:2*n_samples//3, df.columns.get_loc('Group')] = 'Group2'
df.iloc[2*n_samples//3:, df.columns.get_loc('Group')] = 'Group3'

print(f"   ✓ 分组分布:")
print(f"      Group1: {(df['Group']=='Group1').sum()} 样品")
print(f"      Group2: {(df['Group']=='Group2').sum()} 样品")
print(f"      Group3: {(df['Group']=='Group3').sum()} 样品")

# 保存带分组的数据
df.to_csv('data/metabolomics_with_group.csv')
print("   ✓ 已保存: data/metabolomics_with_group.csv")

# ============================================
# 4. 提取代谢物数据（排除分组列）
# ============================================
feature_cols = [col for col in df.columns if col.startswith('M')]
X = df[feature_cols].values
y = df['Group'].values

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n4. 数据标准化完成")
print(f"   特征矩阵形状: {X_scaled.shape}")

# ============================================
# 5. PCA 分析
# ============================================
print("\n5. PCA 分析...")

pca = PCA(n_components=5)
pca_result = pca.fit_transform(X_scaled)
exp_var = pca.explained_variance_ratio_

print(f"   PC1: {exp_var[0]:.2%}")
print(f"   PC2: {exp_var[1]:.2%}")
print(f"   PC3: {exp_var[2]:.2%}")
print(f"   前5主成分累计: {exp_var[:5].sum():.2%}")

# 绘制 PCA 图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 图1：PC1 vs PC2（按分组着色）
colors = {'Group1': 'red', 'Group2': 'green', 'Group3': 'blue'}
for group in df['Group'].unique():
    mask = df['Group'] == group
    axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], 
                    c=colors[group], label=group, alpha=0.7, s=50)
axes[0].set_xlabel(f'PC1 ({exp_var[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({exp_var[1]:.1%})')
axes[0].set_title('PCA: 样品分布（按分组）')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2：PC1 vs PC3
for group in df['Group'].unique():
    mask = df['Group'] == group
    axes[1].scatter(pca_result[mask, 0], pca_result[mask, 2], 
                    c=colors[group], label=group, alpha=0.7, s=50)
axes[1].set_xlabel(f'PC1 ({exp_var[0]:.1%})')
axes[1].set_ylabel(f'PC3 ({exp_var[2]:.1%})')
axes[1].set_title('PCA: PC1 vs PC3')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图3：碎石图
axes[2].bar(range(1, 6), exp_var[:5] * 100)
axes[2].set_xlabel('主成分')
axes[2].set_ylabel('解释方差百分比 (%)')
axes[2].set_title('碎石图')
axes[2].set_xticks(range(1, 6))

plt.tight_layout()
plt.savefig('output/pca_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ 已保存: output/pca_analysis.png")

# ============================================
# 6. 差异代谢物筛选（Group1 vs Group3）
# ============================================
print("\n6. 差异代谢物筛选 (Group1 vs Group3)...")

group1_idx = df['Group'] == 'Group1'
group3_idx = df['Group'] == 'Group3'

results = []
for i, col in enumerate(feature_cols):
    g1_values = X_scaled[group1_idx, i]
    g3_values = X_scaled[group3_idx, i]
    
    # t检验
    t_stat, p_val = ttest_ind(g1_values, g3_values)
    
    # 差异倍数
    fold_change = np.mean(g1_values) / np.mean(g3_values) if np.mean(g3_values) != 0 else np.inf
    
    results.append({
        'Metabolite': col,
        'Fold_Change': fold_change,
        'Log2_FC': np.log2(abs(fold_change)) if fold_change > 0 else 0,
        'P_value': p_val,
        'Direction': 'Up' if fold_change > 1 else 'Down'
    })

df_results = pd.DataFrame(results)
df_results['-Log10_P'] = -np.log10(df_results['P_value'])
df_results = df_results.sort_values('P_value')

# 筛选显著差异代谢物
sig_results = df_results[df_results['P_value'] < 0.05].copy()
print(f"   ✓ 找到 {len(sig_results)} 个显著差异代谢物 (p < 0.05)")

# 保存结果
sig_results.to_csv('output/differential_metabolites.csv', index=False)
print("   ✓ 已保存: output/differential_metabolites.csv")

# ============================================
# 7. 火山图
# ============================================
print("\n7. 绘制火山图...")

fig, ax = plt.subplots(figsize=(10, 8))

# 标记显著差异代谢物
sig_up = sig_results[sig_results['Log2_FC'] > 1]
sig_down = sig_results[sig_results['Log2_FC'] < -1]
non_sig = df_results[df_results['P_value'] >= 0.05]

ax.scatter(non_sig['Log2_FC'], non_sig['-Log10_P'], 
           alpha=0.5, s=30, label='Non-significant', color='gray')
ax.scatter(sig_up['Log2_FC'], sig_up['-Log10_P'], 
           alpha=0.7, s=50, label='Up-regulated (p<0.05)', color='red')
ax.scatter(sig_down['Log2_FC'], sig_down['-Log10_P'], 
           alpha=0.7, s=50, label='Down-regulated (p<0.05)', color='blue')

ax.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=-1, color='black', linestyle='--', alpha=0.5)

ax.set_xlabel('Log2 Fold Change')
ax.set_ylabel('-Log10 P-value')
ax.set_title('火山图: Group1 vs Group3')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/volcano_plot.png', dpi=300, bbox_inches='tight')
print("   ✓ 已保存: output/volcano_plot.png")

# ============================================
# 8. 热图（Top 30 差异代谢物）
# ============================================
print("\n8. 绘制热图...")

top_metabolites = sig_results.head(30)['Metabolite'].tolist()
if len(top_metabolites) > 5:
    # 提取对应数据
    top_indices = [feature_cols.index(m) for m in top_metabolites if m in feature_cols]
    heatmap_data = X_scaled[:, top_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_data.T, cmap='coolwarm', aspect='auto')
    ax.set_xlabel('样品')
    ax.set_ylabel('代谢物')
    ax.set_title(f'Top {len(top_metabolites)} 差异代谢物热图')
    plt.colorbar(im, ax=ax, label='标准化丰度')
    plt.tight_layout()
    plt.savefig('output/heatmap.png', dpi=300, bbox_inches='tight')
    print("   ✓ 已保存: output/heatmap.png")

# ============================================
# 9. 汇总报告
# ============================================
print("\n" + "=" * 60)
print("分析完成！输出文件汇总：")
print("=" * 60)
print("1. data/metabolomics_with_group.csv  - 带分组信息的数据")
print("2. output/pca_analysis.png           - PCA图")
print("3. output/volcano_plot.png           - 火山图")
print("4. output/differential_metabolites.csv - 差异代谢物列表")
print("5. output/heatmap.png                - 热图")
print("=" * 60)
