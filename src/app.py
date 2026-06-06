import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os

# 设置页面配置
st.set_page_config(page_title="Metabolomics Analysis Platform", layout="wide")

st.title("🔬 Metabolomics Data Analysis Platform")
st.markdown("Non-targeted metabolomics analysis based on MTBLS24 dataset")

# ============================================
# 侧边栏：多文件上传
# ============================================
with st.sidebar:
    st.header("📁 Data Upload")
    
    # 多文件上传器
    uploaded_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload your main data file (required) and optional files like differential_metabolites.csv"
    )
    
    st.header("⚙️ Analysis Parameters")
    pca_components = st.slider("Number of PCA components", 2, 5, 2)
    show_heatmap = st.checkbox("Show Heatmap", value=True)
    show_table = st.checkbox("Show Differential Metabolites Table", value=True)
    
    st.header("📊 Project Info")
    st.markdown("""
    - **Samples**: 106
    - **Metabolites**: 701
    - **Groups**: Group1 / Group2 / Group3
    - **Differential Metabolites**: 70 (all down-regulated)
    - **PLS-DA AUC**: 0.885
    """)

# ============================================
# 识别上传的文件
# ============================================
main_data = None
diff_data = None

if uploaded_files:
    for file in uploaded_files:
        filename = file.name.lower()
        if "differential" in filename or "diff" in filename:
            diff_data = pd.read_csv(file)
            st.sidebar.success(f"✅ Loaded: {file.name} (differential metabolites)")
        elif "metabol" in filename or "data" in filename or "ready" in filename:
            main_data = pd.read_csv(file, index_col=0)
            st.sidebar.success(f"✅ Loaded: {file.name} (main data)")
        else:
            # 默认当作主数据
            if main_data is None:
                main_data = pd.read_csv(file, index_col=0)
                st.sidebar.success(f"✅ Loaded: {file.name} (main data)")

# ============================================
# 主区域：数据分析
# ============================================
if main_data is not None:
    df = main_data
    st.success(f"✅ Main data loaded! {df.shape[0]} samples, {df.shape[1]} metabolites")
    
    # 数据预览
    with st.expander("📋 Data Preview"):
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
    st.header("📈 PCA Analysis")
    
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(df_scaled)
    exp_var = pca.explained_variance_ratio_
    
    # 交互式 PCA 图
    fig_pca = px.scatter(
        x=pca_result[:, 0], y=pca_result[:, 1],
        title=f"PCA Analysis (PC1: {exp_var[0]:.1%}, PC2: {exp_var[1]:.1%})",
        labels={'x': f'PC1 ({exp_var[0]:.1%})', 'y': f'PC2 ({exp_var[1]:.1%})'},
        width=800, height=600,
        text=df.index
    )
    fig_pca.update_traces(textposition='top center', marker=dict(size=10))
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # 碎石图
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(exp_var)+1), exp_var * 100, color='steelblue')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Scree Plot')
    ax.set_xticks(range(1, len(exp_var)+1))
    st.pyplot(fig)
    
    # ========================================
    # 热图
    # ========================================
    if show_heatmap:
        st.header("🔥 Heatmap")
        
        variances = df_scaled.var()
        top_metabolites = variances.nlargest(30).index
        heatmap_data = df_scaled[top_metabolites].T
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(heatmap_data, cmap='coolwarm', aspect='auto')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Metabolites')
        ax.set_title('Top 30 Metabolites Heatmap')
        plt.colorbar(im, ax=ax, label='Normalized Abundance')
        st.pyplot(fig)
    
    # ========================================
    # 差异代谢物表格
    # ========================================
    if show_table:
        st.header("📋 Differential Metabolites")
        
        if diff_data is not None:
            # 使用上传的差异代谢物文件
            st.dataframe(diff_data.head(20))
            st.caption(f"📁 Showing first 20 rows from uploaded differential metabolites file")
        else:
            # 没有上传差异代谢物文件时的友好提示
            st.info("""
            💡 **No differential metabolites file uploaded.**
            
            To see the differential metabolites table, please upload a CSV file containing:
            - `differential_metabolites.csv` (from your analysis)
            
            Or, if you don't have this file, you can still view variance analysis below:
            """)
            
            # 显示方差分析作为替代
            variance_df = pd.DataFrame({
                'Metabolite': df.columns,
                'Variance': df.var().values,
                'Mean': df.mean().values
            }).sort_values('Variance', ascending=False)
            
            st.dataframe(variance_df.head(20))
            st.caption("📊 Top 20 metabolites by variance (higher variance = more variation across samples)")
    
    # ========================================
    # 下载结果
    # ========================================
    st.header("📥 Download Results")
    
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca_components)],
        index=df.index
    )
    
    csv = pca_df.to_csv().encode('utf-8')
    st.download_button(
        label="Download PCA Results (CSV)",
        data=csv,
        file_name="pca_results.csv",
        mime="text/csv"
    )

else:
    # 未上传文件时的提示
    st.info("""
    👈 **Please upload your data files**
    
    **Required:**
    - Main metabolomics data file (samples as rows, metabolites as columns)
    
    **Optional:**
    - `differential_metabolites.csv` - Pre-computed differential metabolites (for better display)
    
    You can upload multiple files at once using the file uploader above.
    """)
    
    with st.expander("📖 Example data format"):
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
st.markdown("[📁 GitHub Repository](https://github.com/SaadiyaQW/metabolomics-streamlit)")
