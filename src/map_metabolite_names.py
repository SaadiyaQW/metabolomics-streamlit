import pandas as pd

# 1. 加载原始数据（名称来源）
print("正在加载原始代谢物名称...")
df_raw = pd.read_csv(
    'data/m_MTBLS24_helena_metabolite_profiling_NMR_spectroscopy_15_10_2012_v2_maf.tsv',
    sep='\t', nrows=701  # 只读前701行，因为后面都是样品数据了
)

# 找到名称列（通常是 metabolite_identification）
name_col = None
for col in df_raw.columns:
    if 'metabolite_identification' in col:
        name_col = col
        break

if name_col is None:
    name_col = df_raw.columns[0]  # 如果找不到，就用第一列
    print("未找到 metabolite_identification，使用第一列作为名称。")

# 提取所有的真实名称（按顺序：第1个代谢物 -> 第701个代谢物）
true_names = df_raw[name_col].tolist()
print(f"成功加载 {len(true_names)} 个代谢物名称。")

# 2. 加载你的 VIP 结果（M编号列表）
vip_df = pd.read_csv('output/vip_scores.csv')
important_ms = vip_df[vip_df['VIP'] > 1]['Metabolite'].tolist()
print(f"需要映射的代谢物数量: {len(important_ms)}")

# 3. 建立映射关系
# 注意：你的 M1 对应的就是原始数据中的第 1 个代谢物（索引0），以此类推。
mapping_results = []
for m_id in important_ms:
    # 提取数字编号，例如 'M118' -> 118
    idx = int(m_id.replace('M', '')) - 1  # 减1是因为列表索引从0开始
    if 0 <= idx < len(true_names):
        real_name = true_names[idx]
        mapping_results.append({'M_ID': m_id, 'Real_Name': real_name})
    else:
        mapping_results.append({'M_ID': m_id, 'Real_Name': 'Not Found'})

# 4. 保存映射结果
df_map = pd.DataFrame(mapping_results)
df_map.to_csv('output/metabolite_name_mapping.csv', index=False)
print(f"映射完成！已保存到 output/metabolite_name_mapping.csv")
print("\n前10个映射结果：")
print(df_map.head(10))
