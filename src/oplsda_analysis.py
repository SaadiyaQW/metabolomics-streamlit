import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("OPLS-DA (PLS-DA) 分析")
print("=" * 60)

# ============================================
# 1. 加载数据
# ============================================
df = pd.read_csv('data/metabolomics_with_group.csv', index_col=0)
print(f"\n1. 加载数据: {df.shape[0]} 样品 × {df.shape[1]} 代谢物")

# 提取特征和标签
feature_cols = [col for col in df.columns if col.startswith('M')]
X = df[feature_cols].values
y = df['Group'].values

# 将标签转换为数值
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"   分组映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 只分析 Group1 vs Group3
mask = (y_encoded == 0) | (y_encoded == 2)
X_13 = X[mask]
y_13 = y_encoded[mask]
y_13_binary = (y_13 == 2).astype(int)

print(f"\n2. 分析 Group1 vs Group3: {len(y_13_binary)} 个样品")
print(f"   Group1: {(y_13_binary==0).sum()}, Group3: {(y_13_binary==1).sum()}")

# ============================================
# 2. 数据标准化
# ============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_13)
print(f"\n3. 数据标准化完成")

# ============================================
# 3. PLS-DA 建模（手动交叉验证）
# ============================================
print("\n4. 构建 PLS-DA 模型...")

best_n_components = 2
best_score = 0
cv_scores = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for n_comp in range(1, 6):
    pls = PLSRegression(n_components=n_comp)
    fold_scores = []
    for train_idx, val_idx in skf.split(X_scaled, y_13_binary):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_13_binary[train_idx], y_13_binary[val_idx]
        
        pls.fit(X_train, y_train)
        y_pred = pls.predict(X_val)
        y_pred_class = (y_pred > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_val, y_pred_class)
        fold_scores.append(acc)
    
    mean_score = np.mean(fold_scores)
    cv_scores.append(mean_score)
    print(f"   n_components={n_comp}, CV准确率={mean_score:.2%}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_n_components = n_comp

print(f"\n   最优主成分数: {best_n_components}")
print(f"   最佳CV准确率: {best_score:.2%}")

# 使用最优主成分数训练最终模型
pls = PLSRegression(n_components=best_n_components)
pls.fit(X_scaled, y_13_binary)

# ============================================
# 4. 模型评估
# ============================================
print("\n5. 模型评估...")

y_pred = pls.predict(X_scaled)
y_pred_class = (y_pred > 0.5).astype(int).flatten()
accuracy = accuracy_score(y_13_binary, y_pred_class)
print(f"   训练集准确率: {accuracy:.2%}")

y_cv_pred = cross_val_predict(pls, X_scaled, y_13_binary, cv=skf)
y_cv_class = (y_cv_pred > 0.5).astype(int).flatten()
cv_accuracy = accuracy_score(y_13_binary, y_cv_class)
print(f"   交叉验证准确率: {cv_accuracy:.2%}")

cm = confusion_matrix(y_13_binary, y_cv_class)
print(f"\n   混淆矩阵:")
print(f"   Group1 预测正确: {cm[0,0]}, 预测错误: {cm[0,1]}")
print(f"   Group3 预测正确: {cm[1,1]}, 预测错误: {cm[1,0]}")

# ============================================
# 5. 计算 VIP 值
# ============================================
print("\n6. 计算 VIP 值...")

def calculate_vip(pls_model):
    w = pls_model.x_weights_
    q = pls_model.y_loadings_
    
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    
    p = w.shape[0]
    vip = np.zeros(p)
    
    ss = np.sum(w**2, axis=0) * np.sum(q**2, axis=0)
    
    for j in range(p):
        weighted_sum = np.sum(w[j, :]**2 * ss)
        vip[j] = np.sqrt(p * weighted_sum / np.sum(ss))
    
    return vip

vip_scores = calculate_vip(pls)

vip_df = pd.DataFrame({
    'Metabolite': feature_cols,
    'VIP': vip_scores,
    'Coefficient': pls.coef_.flatten()
})
vip_df = vip_df.sort_values('VIP', ascending=False)

important_features = vip_df[vip_df['VIP'] > 1]
print(f"   VIP > 1 的代谢物数量: {len(important_features)}")
print(f"\n   Top 10 VIP 代谢物:")
for i, row in important_features.head(10).iterrows():
    print(f"      {row['Metabolite']}: VIP = {row['VIP']:.3f}")

vip_df.to_csv('output/vip_scores.csv', index=False)
print(f"\n   ✓ 已保存: output/vip_scores.csv")

# ============================================
# 6. 可视化（修复索引错误）
# ============================================
print("\n7. 生成可视化图表...")

n_components_actual = pls.x_scores_.shape[1]  # 实际主成分数
print(f"   实际主成分数: {n_components_actual}")

# 根据实际主成分数确定子图布局
fig = plt.figure(figsize=(16, 10))

# 图1：PLS-DA 得分图（如果只有1个成分，用一维散点图）
ax1 = fig.add_subplot(2, 2, 1)
t = pls.x_scores_
colors = ['red' if y_13_binary[i] == 0 else 'blue' for i in range(len(y_13_binary))]

if n_components_actual >= 2:
    ax1.scatter(t[:, 0], t[:, 1], c=colors, alpha=0.7, s=60)
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
else:
    # 只有1个主成分时，用一维散点图
    x_pos = np.zeros(len(t))
    ax1.scatter(t[:, 0], x_pos, c=colors, alpha=0.7, s=60)
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('')
    ax1.set_yticks([])

ax1.set_title('PLS-DA 得分图 (Group1 vs Group3)')
ax1.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Group1'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Group3')
])
ax1.grid(True, alpha=0.3)

# 图2：VIP 柱状图（Top 15）
ax2 = fig.add_subplot(2, 2, 2)
top_vip = vip_df.head(15)
ax2.barh(top_vip['Metabolite'], top_vip['VIP'], color='steelblue')
ax2.axvline(x=1, color='red', linestyle='--', label='VIP = 1')
ax2.set_xlabel('VIP Value')
ax2.set_title('Top 15 代谢物 VIP 值')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='x')

# 图3：ROC 曲线
ax3 = fig.add_subplot(2, 2, 3)
y_score = pls.predict(X_scaled).flatten()
fpr, tpr, _ = roc_curve(y_13_binary, y_score)
roc_auc = auc(fpr, tpr)

ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC 曲线')
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.3)

# 图4：CV 准确率 vs 主成分数
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(range(1, 6), cv_scores, marker='o', linestyle='-', color='green', linewidth=2, markersize=8)
ax4.set_xlabel('Number of Components')
ax4.set_ylabel('CV Accuracy')
ax4.set_title('交叉验证准确率 vs 主成分数')
ax4.set_xticks(range(1, 6))
ax4.set_ylim([0, 1])
ax4.grid(True, alpha=0.3)

# 标注最佳点
best_idx = best_n_components - 1
ax4.plot(best_n_components, cv_scores[best_idx], 'ro', markersize=12, markeredgecolor='darkred')
ax4.annotate(f'Best: n={best_n_components}\nAcc={cv_scores[best_idx]:.1%}',
             xy=(best_n_components, cv_scores[best_idx]),
             xytext=(best_n_components+0.5, cv_scores[best_idx]-0.1),
             arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('output/oplsda_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ 已保存: output/oplsda_analysis.png")

# 保存分组信息
group_info = pd.DataFrame({
    'Sample': df.index[mask],
    'Group': ['Group1' if x==0 else 'Group3' for x in y_13_binary]
})
group_info.to_csv('output/group1_vs_group3_info.csv', index=False)

# ============================================
# 7. 汇总结果
# ============================================
print("\n" + "=" * 60)
print("OPLS-DA 分析完成！")
print("=" * 60)
print(f"\n📊 模型性能:")
print(f"   - 最优主成分数: {best_n_components}")
print(f"   - 交叉验证准确率: {cv_accuracy:.2%}")
print(f"   - AUC: {roc_auc:.3f}")
print(f"\n🔬 VIP > 1 的代谢物: {len(important_features)} 个")
print(f"\n📁 输出文件:")
print("   - output/vip_scores.csv              - VIP 值表")
print("   - output/oplsda_analysis.png         - OPLS-DA 可视化")
print("   - output/group1_vs_group3_info.csv   - 分组信息")
print("=" * 60)