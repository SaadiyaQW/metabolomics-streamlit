import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# 设置页面配置
st.set_page_config(page_title="代谢组学分析平台", layout="wide")

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

st.title("🔬 代谢组学数据分析平台")
st.markdown("基于 MTBLS24 数据集的非靶向代谢组学分析")

# ============================================
# 侧边栏：文件上传和参数设置
# ============================================
with st.sidebar:
    st.header("📁 数据加载")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
    
    st.header("⚙️ 分析参数")
    pca_components = st.slider("PCA 主成分数", 2, 5, 2)
    show_heatmap = st.checkbox("显示热图", value=True)
    show_table = st.checkbox("显示差异代谢物表格", value=True)
    
    st.header("📊 项目信息")
    st.markdown("""
    - **样品数**: 106
    - **代谢物数**: 701
    - **分组**: Group1 / Group2 / Group3
    - **差异代谢物**: 70个（全部下调）
    - **PLS-DA AUC**: 0.885
    """)

# ============================================
# 主区域：数据分析
# ============================================
if uploaded_file is not None:
    # 读取数据
    df = pd.read_csv(uploaded_file, index_col=0)
    st.success(f"✅ 数据加载成功！{df.shape[0]} 个样品，{df.shape[1]} 个代谢物")
    
    # 数据预览
    with st.expander("📋 数据预览"):
        st.dataframe(df.head())
    
    # 数据标准化
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    
    # ========================================
    # PCA 分析
    # ========================================
    st.header("📈 PCA 分析")
    
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(df_scaled)
    exp_var = pca.explained_variance_ratio_
    
    # 使用 Plotly 绘制交互式 PCA 图
    fig_pca = px.scatter(
        x=pca_result[:, 0], y=pca_result[:, 1],
        title=f"PCA 分析 (PC1: {exp_var[0]:.1%}, PC2: {exp_var[1]:.1%})",
        labels={'x': f'PC1 ({exp_var[0]:.1%})', 'y': f'PC2 ({exp_var[1]:.1%})'},
        width=800, height=600,
        text=df.index
    )
    fig_pca.update_traces(textposition='top center', marker=dict(size=10))
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # 碎石图
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(exp_var)+1), exp_var * 100, color='steelblue')
    ax.set_xlabel('主成分')
    ax.set_ylabel('解释方差百分比 (%)')
    ax.set_title('碎石图')
    ax.set_xticks(range(1, len(exp_var)+1))
    st.pyplot(fig)
    
    # ========================================
    # 热图（可选）
    # ========================================
    if show_heatmap:
        st.header("🔥 差异代谢物热图")
        
        # 计算差异最大的前30个代谢物（按方差排序）
        variances = df_scaled.var()
        top_metabolites = variances.nlargest(30).index
        heatmap_data = df_scaled[top_metabolites].T
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(heatmap_data, cmap='coolwarm', aspect='auto')
        ax.set_xlabel('样品')
        ax.set_ylabel('代谢物')
        ax.set_title('Top 30 变异代谢物热图')
        plt.colorbar(im, ax=ax, label='标准化丰度')
        st.pyplot(fig)
    
    # ========================================
    # 差异代谢物表格（可选）
    # ========================================
    if show_table:
        st.header("📋 差异代谢物列表")
        
        # 尝试读取预先生成的差异代谢物列表
        if os.path.exists("output/differential_metabolites.csv"):
            df_diff = pd.read_csv("output/differential_metabolites.csv")
            st.dataframe(df_diff.head(20))
        else:
            # 如果没有预先生成文件，显示方差分析结果
            variance_df = pd.DataFrame({
                'Metabolite': df.columns,
                'Variance': df.var().values,
                'Mean': df.mean().values
            }).sort_values('Variance', ascending=False)
            st.dataframe(variance_df.head(20))
            st.info("💡 提示：上传完整的差异代谢物文件（differential_metabolites.csv）可获得更详细的结果")
    
    # ========================================
    # 下载结果
    # ========================================
    st.header("📥 下载分析结果")
    
    # 生成 PCA 结果表
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca_components)],
        index=df.index
    )
    
    csv = pca_df.to_csv().encode('utf-8')
    st.download_button(
        label="下载 PCA 结果 (CSV)",
        data=csv,
        file_name="pca_results.csv",
        mime="text/csv"
    )

else:
    # 未上传文件时的提示
    st.info("👈 请上传 CSV 格式的代谢组数据（行为样品，列为代谢物）")
    
    with st.expander("📖 查看示例数据格式"):
        st.code("""
        Sample,Metabolite1,Metabolite2,...,Group
        Sample1,12.3,45.6,...,Group1
        Sample2,13.4,46.7,...,Group1
        ...
        """, language="text")

# ============================================
# 页脚
# ============================================
st.markdown("---")
st.markdown("📧 联系方式 | [GitHub 仓库](https://github.com/SaadiyaQW/metabolomics-streamlit)")
