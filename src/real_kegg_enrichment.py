import pandas as pd
from gseapy import enrichr

# 1. 加载原始数据
df_raw = pd.read_csv(
    'data/m_MTBLS24_helena_metabolite_profiling_NMR_spectroscopy_15_10_2012_v2_maf.tsv',
    sep='\t',
    nrows=701  # 只读前701行
)

# 2. 提取信息
# 列索引需要根据你的文件调整（建议先打印 df_raw.columns）
print("原始列名:", list(df_raw.columns[:10]))

# 假设 database_identifier 是第0列，metabolite_identification 是第4列
chebi_ids = df_raw.iloc[:, 0].tolist()      # database_identifier 列
real_names = df_raw.iloc[:, 4].tolist()     # metabolite_identification 列

# 3. 加载 VIP 结果
vip_df = pd.read_csv('output/vip_scores.csv')
important_ms = vip_df[vip_df['VIP'] > 1]['Metabolite'].tolist()

# 4. 建立映射
mapping_results = []
for m_id in important_ms:
    idx = int(m_id.replace('M', '')) - 1
    if 0 <= idx < len(chebi_ids):
        chebi = chebi_ids[idx] if pd.notna(chebi_ids[idx]) else None
        name = real_names[idx] if pd.notna(real_names[idx]) else None
    else:
        chebi, name = None, None
    
    mapping_results.append({
        'M_ID': m_id,
        'ChEBI_ID': chebi,
        'Real_Name': name
    })

df_map = pd.DataFrame(mapping_results)
df_map.to_csv('output/metabolite_name_mapping_with_chebi.csv', index=False)
print("已保存带 ChEBI ID 的映射表")
print("\n前10行:")
print(df_map.head(10))

# 5. 统计有多少有 ChEBI ID
has_chebi = df_map['ChEBI_ID'].notna().sum()
has_name = df_map['Real_Name'].notna().sum()
print(f"\n统计: {has_chebi} 个代谢物有 ChEBI ID, {has_name} 个有名称")
