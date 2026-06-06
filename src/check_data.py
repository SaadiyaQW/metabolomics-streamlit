import pandas as pd

df = pd.read_csv('data/metabolomics_ready.csv', index_col=0)
print(f"数据形状: {df.shape[0]} 样品 × {df.shape[1]} 代谢物")
print(f"\n前5个样品名:\n{df.index[:5]}")
print(f"\n前5个代谢物名:\n{df.columns[:5]}")
print(f"\n数据前5行前5列:\n{df.iloc[:5, :5]}")