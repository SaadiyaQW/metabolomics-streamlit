import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gseapy import enrichr
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("基于48个鉴定代谢物的KEGG通路富集分析")
print("=" * 60)

# ============================================
# 1. 加载有ChEBI ID的代谢物
# ============================================
map_df = pd.read_csv('output/metabolite_name_mapping_with_chebi.csv')
valid = map_df[map_df['ChEBI_ID'].notna()].copy()

# 提取ChEBI ID的数字部分
gene_list = []
for chebi in valid['ChEBI_ID']:
    if 'CHEBI:' in str(chebi):
        gene_list.append(chebi.replace('CHEBI:', ''))
    elif str(chebi).isdigit():
        gene_list.append(str(chebi))

print(f"\n1. 可用代谢物数量: {len(gene_list)}")
print(f"   前10个ChEBI ID: {gene_list[:10]}")

# ============================================
# 2. 使用Enrichr进行KEGG富集分析
# ============================================
print("\n2. 正在查询KEGG通路...")

try:
    enr = enrichr(gene_list=gene_list,
                  gene_sets='KEGG_2021_Human',
                  organism='Human',
                  description='48_metabolites_kegg',
                  outdir='output/kegg_48_analysis',
                  cutoff=0.5
                  )
    
    results = enr.results
    print(f"\n   分析完成！找到 {len(results)} 个通路。")
    
    # ============================================
    # 3. 筛选显著富集通路
    # ============================================
    sig_results = results[results['Adjusted P-value'] < 0.05].copy()
    print(f"\n3. 显著富集通路 (p_adj < 0.05): {len(sig_results)} 个")
    
    if len(sig_results) == 0:
        # 尝试使用更宽松的阈值
        sig_results = results[results['P-value'] < 0.05].copy()
        print(f"   未校正 p < 0.05 的通路: {len(sig_results)} 个")
    
    # ============================================
    # 4. 显示结果
    # ============================================
    if len(sig_results) > 0:
        print("\n📊 显著富集的通路:")
        for i, row in sig_results.head(10).iterrows():
            genes_in_pathway = row['Genes'].split(';') if pd.notna(row['Genes']) else []
            print(f"\n   {i+1}. {row['Term']}")
            print(f"      Adjusted P-value: {row['Adjusted P-value']:.4f}")
            print(f"      Odds Ratio: {row['Odds Ratio']:.2f}")
            print(f"      Overlap: {row['Overlap']}")
            print(f"      Genes: {', '.join(genes_in_pathway[:5])}...")
    else:
        print("\n⚠️ 没有找到显著富集的通路")
        print("   可能的原因:")
        print("   1. 48个代谢物分布太分散")
        print("   2. 这些代谢物主要不来自KEGG注释的通路")
        print("   3. 需要更多的鉴定代谢物")
    
    # ============================================
    # 5. 保存结果
    # ============================================
    results.to_csv('output/kegg_48_results.csv', index=False)
    print(f"\n✅ 已保存: output/kegg_48_results.csv")
    
    # ============================================
    # 6. 可视化
    # ============================================
    if len(sig_results) > 0:
        plot_df = sig_results.head(10).copy()
    else:
        plot_df = results.head(10).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 图1：气泡图
    ax1 = axes[0]
    plot_df_sorted = plot_df.sort_values('Adjusted P-value')
    x = -np.log10(plot_df_sorted['Adjusted P-value'] + 1e-10)
    y = np.arange(len(plot_df_sorted))
    
    # 气泡大小根据Overlap数量
    overlap_counts = plot_df_sorted['Overlap'].apply(
        lambda x: int(x.split('/')[0]) if pd.notna(x) else 1
    )
    sizes = overlap_counts * 50
    
    scatter = ax1.scatter(x, y, s=sizes, alpha=0.7, c=plot_df_sorted['Odds Ratio'], 
                          cmap='RdYlGn', edgecolors='black', linewidth=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels([term[:40] for term in plot_df_sorted['Term']])
    ax1.set_xlabel('-Log10(Adjusted P-value)')
    ax1.set_title('KEGG通路富集气泡图')
    ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax1, label='Odds Ratio')
    
    # 图2：富集倍数柱状图
    ax2 = axes[1]
    colors = ['red' if p < 0.05 else 'gray' for p in plot_df['Adjusted P-value']]
    ax2.barh(plot_df['Term'][:10], plot_df['Odds Ratio'][:10], color=colors[:10])
    ax2.axvline(x=1, color='red', linestyle='--', label='Odds Ratio = 1')
    ax2.set_xlabel('Odds Ratio')
    ax2.set_title('通路富集倍数')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('output/kegg_48_enrichment.png', dpi=300, bbox_inches='tight')
    print("✅ 已保存: output/kegg_48_enrichment.png")
    
    # ============================================
    # 7. 汇总报告
    # ============================================
    print("\n" + "=" * 60)
    print("KEGG富集分析完成")
    print("=" * 60)
    print(f"\n📊 分析概况:")
    print(f"   - 输入代谢物数: {len(gene_list)}")
    print(f"   - 总通路数: {len(results)}")
    print(f"   - 显著富集通路: {len(sig_results)}")
    
    if len(sig_results) > 0:
        print(f"\n🔬 显著富集通路列表:")
        for i, row in sig_results.head(5).iterrows():
            print(f"   {i+1}. {row['Term']} (p_adj={row['Adjusted P-value']:.4f})")
    
except Exception as e:
    print(f"❌ 分析出错: {e}")
    print("\n请检查:")
    print("1. 网络连接是否正常")
    print("2. gseapy是否安装: pip install gseapy")
    print("3. ChEBI ID格式是否正确")

print("\n" + "=" * 60)