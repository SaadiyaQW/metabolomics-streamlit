import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os

# ============================================
# 页面配置（必须在最前面）
# ============================================
st.set_page_config(page_title="Metabolomics Analysis Platform", layout="wide")

# ============================================
# 读取 URL 参数，识别用户类型
# ============================================
query_params = st.query_params
token = query_params.get("token", "")

# 定义你的秘密 token
UNLIMITED_TOKEN = "admin456"   # 给你自己用（无限制）
LIMITED_TOKEN = "abc123"       # 给一般用户（限制版）

if token == UNLIMITED_TOKEN:
    mode = "unlimited"
    st.info("🔓 无限制模式")
elif token == LIMITED_TOKEN:
    mode = "limited"
    st.info("🔒 限制模式（只能上传2次）")
else:
    mode = "limited"
    st.warning("⚠️ 默认限制模式")

# 初始化上传计数器
if 'upload_count' not in st.session_state:
    st.session_state.upload_count = 0

st.title("🔬 Metabolomics Data Analysis Platform")
st.markdown("Non-targeted metabolomics analysis based on MTBLS24 dataset")

# ============================================
# 侧边栏
# ============================================
with st.sidebar:
    st.header("📁 Data Upload")
    
    # 限制版：显示剩余次数
    if mode == "limited":
        remaining = 2 - st.session_state.upload_count
        st.info(f"📊 剩余上传次数: {remaining}")
    
    # 上传逻辑
    if mode == "limited" and st.session_state.upload_count >= 2:
        st.warning("⚠️ 已达到上传次数上限（2次）。刷新页面可重置。")
        uploaded_files = None
    else:
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload your main data file (required) and optional files like differential_metabolites.csv"
        )
        
        if uploaded_files and mode == "limited":
            st.session_state.upload_count += 1
            st.rerun()
    
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
            st.sidebar.success(f"✅ Loaded: {file.name}")
        elif "metabol" in filename or "data" in filename or "ready" in filename:
            main_data = pd.read_csv(file, index_col=0)
            st.sidebar.success(f"✅ Loaded: {file.name}")
        else:
            if main_data is None:
                main_data = pd.read_csv(file, index_col=0)
                st.sidebar.success(f"✅ Loaded: {file.name}")

# ============================================
# 主区域分析（与之前相同，略）
# ============================================
if main_data is not None:
    df = main_data
    st.success(f"✅ Main data loaded! {df.shape[0]} samples, {df.shape[1]} metabolites")
    
    with st.expander("📋 Data Preview"):
        st.dataframe(df.head())
    
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    
    st.header("📈 PCA Analysis")
    
    pca = PCA(n_components=pca_components)
    pca_result = pca.fit_transform(df_scaled)
    exp_var = pca.explained_variance_ratio_
    
    fig_pca = px.scatter(
        x=pca_result[:, 0], y=pca_result[:, 1],
        title=f"PCA Analysis (PC1: {exp_var[0]:.1%}, PC2: {exp_var[1]:.1%})",
        labels={'x': f'PC1 ({exp_var[0]:.1%})', 'y': f'PC2 ({exp_var[1]:.1%})'},
        width=800, height=600,
        text=df.index
    )
    fig_pca.update_traces(textposition='top center', marker=dict(size=10))
    st.plotly_chart(fig_pca, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(exp_var)+1), exp_var * 100, color='steelblue')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Scree Plot')
    ax.set_xticks(range(1, len(exp_var)+1))
    st.pyplot(fig)
    
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
    
    if show_table:
        st.header("📋 Differential Metabolites")
        if diff_data is not None:
            st.dataframe(diff_data.head(20))
            st.caption("📁 Showing first 20 rows from uploaded differential metabolites file")
        else:
            st.info("💡 No differential metabolites file uploaded. Showing variance analysis instead.")
            variance_df = pd.DataFrame({
                'Metabolite': df.columns,
                'Variance': df.var().values,
                'Mean': df.mean().values
            }).sort_values('Variance', ascending=False)
            st.dataframe(variance_df.head(20))
    
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
    st.info("""
    👈 **Please upload your data files**
    
    **Required:**
    - Main metabolomics data file (samples as rows, metabolites as columns)
    
    **Optional:**
    - `differential_metabolites.csv` - Pre-computed differential metabolites
    """)
    
    with st.expander("📖 Example data format"):
        st.code("""
        Sample,Metabolite1,Metabolite2,...,Group
        Sample1,12.3,45.6,...,Group1
        Sample2,13.4,46.7,...,Group1
        ...
        """, language="text")

st.markdown("---")
st.markdown("[📁 GitHub Repository](https://github.com/SaadiyaQW/metabolomics-streamlit)")
