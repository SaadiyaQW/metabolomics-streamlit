import pandas as pd

# 加载映射表
map_df = pd.read_csv('output/metabolite_name_mapping_with_chebi.csv')
valid = map_df[map_df['ChEBI_ID'].notna()].copy()

print(f"可用代谢物数量: {len(valid)}")
print("\n前20个可用的代谢物:")
for i, row in valid.head(20).iterrows():
    print(f"  {row['M_ID']}: {row['ChEBI_ID']} -> {row['Real_Name']}")