import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gseapy import enrichr
import warnings
import os
import requests
import urllib.request

# ============================================
# 代理配置（保持不变）
# ============================================
PROXY_PORT = "7897"
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL

proxy_handler = urllib.request.ProxyHandler({'http': PROXY_URL, 'https': PROXY_URL})
opener = urllib.request.build_opener(proxy_handler)
urllib.request.install_opener(opener)

warnings.filterwarnings('ignore')

print("=" * 60)
print("KEGG 通路富集分析（最终版）")
print(f"代理地址: {PROXY_URL}")
print("=" * 60)

# ============================================
# 1. 准备 ChEBI ID 列表
# ============================================
gene_list = ['57589', '15347', '48669', '8087', '17368', '18089', '32636', '18183']
print(f"\n输入代谢物 (ChEBI ID): {gene_list}")

# ============================================
# 2. 测试网络连接
# ============================================
print("\n测试网络连接...")
try:
    proxies = {'http': PROXY_URL, 'https': PROXY_URL}
    r = requests.get('https://maayanlab.cloud', proxies=proxies, timeout=10)
    print("✅ 网络连通")
except Exception as e:
    print(f"❌ 网络不通: {e}")
    exit()

# ============================================
# 3. 使用 enrichr 函数（不设 cutoff，让结果显示所有）
# ============================================
print("\n正在查询 KEGG_2021_Human 数据库...")
print("（这可能需要 1-2 分钟，请耐心等待）")

try:
    # 关键：不设置 cutoff 参数，让它返回所有结果
    # 如果报错，再尝试 cutoff=1.0
    enr = enrichr(gene_list=gene_list,
                  gene_sets='KEGG_2021_Human',
                  organism='human',
                  outdir='output/kegg_final',
                  description='8_metabolites'
                  )
    
    # 获取结果
    results = enr.results
    print(f"\n✅ 查询成功！找到 {len(results)} 个通路。")
    
    if len(results) > 0:
        # 按 P-value 排序
        results = results.sort_values('P-value')
        
        print("\n📊 通路富集结果（按P值排序，前15个）:")
        print("-" * 80)
        for i, row in results.head(15).iterrows():
            print(f"\n{i+1}. {row['Term']}")
            print(f"   P-value: {row['P-value']:.4f}")
            if 'Adjusted P-value' in row:
                print(f"   Adj P-val: {row['Adjusted P-value']:.4f}")
            print(f"   Odds Ratio: {row['Odds Ratio']:.2f}")
            print(f"   Overlap: {row['Overlap']}")
            print(f"   Genes: {row['Genes']}")
        
        # 保存完整结果
        results.to_csv('output/kegg_final_results.csv', index=False)
        print("\n✅ 结果已保存至 output/kegg_final_results.csv")
        
        # 可视化
        if len(results) > 0:
            plot_df = results.head(15).copy()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_df = plot_df.sort_values('P-value')
            y_pos = np.arange(len(plot_df))
            
            # 根据 P-value 着色
            colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'steelblue' 
                      for p in plot_df['P-value']]
            
            ax.barh(y_pos, -np.log10(plot_df['P-value'] + 1e-10), color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([term[:40] for term in plot_df['Term']])
            ax.set_xlabel('-Log10(P-value)')
            ax.set_title('KEGG 通路富集分析')
            ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p = 0.05')
            ax.axvline(x=-np.log10(0.1), color='orange', linestyle='--', label='p = 0.1')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig('output/kegg_final_enrichment.png', dpi=300, bbox_inches='tight')
            print("✅ 图片已保存至 output/kegg_final_enrichment.png")
    else:
        print("⚠️ 没有找到任何通路结果")
        print("可能原因：这8个ChEBI ID在KEGG数据库中匹配不到通路")
        print("\n备选方案：手动查询各个ID")
        print("访问 https://www.kegg.jp/kegg/compound/ 输入每个ID查看")
        
except Exception as e:
    print(f"\n❌ 分析出错: {e}")
    print("\n可能是 gseapy 版本或网络问题")
    print("请尝试在终端执行：")
    print("   pip install --upgrade gseapy")
    print("\n或者将结果文件发给我，我手动帮你分析")

print("\n" + "=" * 60)
print("分析完成")
print("=" * 60)