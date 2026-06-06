import pandas as pd

# 读取 TSV，pandas 会自动处理制表符分隔
df = pd.read_csv(
    "data/m_MTBLS24_helena_metabolite_profiling_NMR_spectroscopy_15_10_2012_v2_maf.tsv",
    sep="\t",
    encoding="utf-8"
)

print("✅ 数据加载成功")
print(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"\n前 5 列: {list(df.columns[:5])}")
print(f"最后 5 列: {list(df.columns[-5:])}")

# 自动找样品列（列名类似 AKI_xxx）
sample_cols = [col for col in df.columns if col.startswith("AKI")]
print(f"\n找到 {len(sample_cols)} 个样品列")

if len(sample_cols) > 0:
    # 提取代谢物名称（第一列）
    metabolite_names = df.iloc[:, 0]
    
    # 提取样品数据
    df_samples = df[sample_cols]
    
    # 转置，变成 样品 × 代谢物
    df_ready = df_samples.T
    df_ready.columns = metabolite_names
    df_ready.index.name = "Sample_ID"
    
    # 保存为可直接分析的 CSV
    df_ready.to_csv("data/metabolomics_ready.csv")
    print(f"\n✅ 已保存: data/metabolomics_ready.csv")
    print(f"最终形状: {df_ready.shape[0]} 样品 × {df_ready.shape[1]} 代谢物")
else:
    print("\n⚠️ 没有找到以 AKI 开头的样品列，请检查列名格式")