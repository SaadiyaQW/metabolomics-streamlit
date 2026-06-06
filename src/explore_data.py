import pandas as pd
import os

# 设置数据文件夹路径（根据你实际存放的位置修改）
data_folder = "data"  # 如果你的 data 文件夹在项目根目录下

# 获取所有 .tsv 文件
tsv_files = [f for f in os.listdir(data_folder) if f.endswith('.tsv')]

print(f"找到 {len(tsv_files)} 个 TSV 文件\n")
print("=" * 80)

for file in tsv_files:
    file_path = os.path.join(data_folder, file)
    
    print(f"\n📁 文件名: {file}")
    
    # 读取前 5 行，先不加载全部（有些文件可能很大）
    try:
        df_sample = pd.read_csv(file_path, sep='\t', nrows=5)
        print(f"   列数: {df_sample.shape[1]}")
        print(f"   列名前5个: {list(df_sample.columns[:5])}")
        
        # 读取全部数据（如果文件不大）
        df_full = pd.read_csv(file_path, sep='\t')
        print(f"   总行数: {df_full.shape[0]}")
        print(f"   总列数: {df_full.shape[1]}")
        
        # 检查是否有分组信息列
        possible_group_cols = [col for col in df_full.columns if 'group' in col.lower() or 'class' in col.lower() or 'treatment' in col.lower()]
        if possible_group_cols:
            print(f"   🔍 发现可能的分组列: {possible_group_cols[:3]}")
        
        # 显示前 2 行（转置显示，方便看列名和值）
        print("\n   📋 前2行数据（转置）:")
        print(df_full.head(2).T)
        
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")
    
    print("-" * 80)
    