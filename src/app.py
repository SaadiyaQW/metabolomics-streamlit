import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# 页面配置
st.set_page_config(page_title="代谢组学分析平台", layout="wide")

st.title("🔬 代谢组学数据分析平台")
st.markdown("基于 MTBLS24 数据集的非靶向代谢组学分析")

# 侧边栏
with st.sidebar:
    st.header("📁 数据加载")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    st.markdown("---")
    st.header("⚙️ 分析参数")
    pca_components = st.slider("PCA 主成分数", 2, 5, 2)
    st.markdown("---")
    st.header("📊 项目信息")
    st.markdown("- 样品数: 106")
    st.markdown("- 代谢物数: 701")
    st.markdown("- 分组: Group1/2/3")

# 主区域
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.success(f"✅ 数据加载成功！{df.shape[0]} 个样品，{df.shape[1]} 个代谢物")
    
    # 数据预览
    with st.expander("📋 数据预览"):
        st.dataframe(df.head())
    
    # PCA 分析
    st.subheader("📈 PCA 分析")
    
    # 标准化
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    
    # PCA
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(df_scaled)
    
    # 解释方差
    exp_var = pca.explained_variance_ratio_
    
    # 使用 plotly 绘制交互式 PCA 图
    fig = px.scatter(
        x=pca_result[:, 0], y=pca_result[:, 1],
        title=f"PCA 分析 (PC1: {exp_var[0]:.1%}, PC2: {exp_var[1]:.1%})",
        labels={'x': 'PC1', 'y': 'PC2'},
        width=800, height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 碎石图
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(exp_var)+1), exp_var * 100)
    ax.set_xlabel('主成分')
    ax.set_ylabel('解释方差百分比 (%)')
    ax.set_title('碎石图')
    st.pyplot(fig2)
    
else:
    # 未上传文件时，显示示例数据
    st.info("👈 请上传 CSV 格式的代谢组数据（行为样品，列为代谢物）")
    
    with st.expander("📖 查看示例数据格式"):
        st.code("""
        Sample,Metabolite1,Metabolite2,...,Group,Bitter_Value
        Sample1,12.3,45.6,...,Group1,5.2
        Sample2,13.4,46.7,...,Group1,5.5
        ...
        """, language="text")

# 页脚
st.markdown("---")
st.markdown("📧 联系方式 | 📚 项目说明 | 🔗 GitHub 仓库")