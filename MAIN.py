import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Set page configuration
st.set_page_config(
    page_title="QSAR Drug Target Analyzer",
    page_icon="🧬",
    layout="wide"
)

# ----------------------------------------
# TITLE & INTRODUCTION
# ----------------------------------------
st.title("🧬 QSAR Drug Target Activity Analyzer")
st.markdown("""
**Welcome to the QSAR Data Dashboard.** This application is designed to analyze Quantitative Structure-Activity Relationship (QSAR) datasets, specifically focusing on how different molecules interact with various drug targets. 
""")

st.sidebar.header("Setup & Filters")

# ----------------------------------------
# DATA LOADING & CLEANING (Optimized)
# ----------------------------------------
@st.cache_data
def load_and_clean_data(file_path):
    try:
        preview = pd.read_csv(file_path, nrows=1)
        molecule_col = preview.columns[0]
        
        all_cols = preview.columns.tolist()
        numeric_cols = preview.select_dtypes(include="number").columns.tolist()
        keep_cols = [molecule_col] + numeric_cols
        
        df = pd.read_csv(file_path, usecols=keep_cols)
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
    except Exception as e:
        return None
    
    df.drop_duplicates(inplace=True)
    return df

GITHUB_RELEASE_URL = "https://github.com/morlockmaimai-ui/Drug-Target-Activity-Analyzer-/releases/download/v1.0.0/QSAR.csv"

with st.spinner("Streaming dataset efficiently from GitHub assets... Please wait."):
    df = load_and_clean_data(GITHUB_RELEASE_URL)

if df is None:
    st.error("⚠️ Failed to load data. Verify your Release Link format or check if your repository is public.")
    st.stop()

molecule_col = df.columns[0]
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Sidebar: Filter Molecule globally or view all
all_molecules = df[molecule_col].unique().tolist()
selected_molecule = st.sidebar.selectbox("Select a Specific Molecule to Focus On:", ["All"] + all_molecules)

filtered_df = df if selected_molecule == "All" else df[df[molecule_col] == selected_molecule]

# ----------------------------------------
# METRICS & KEY STATISTICS
# ----------------------------------------
st.header("📊 Key Statistics & Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Unique Molecules", len(all_molecules))
with col2:
    st.metric("Total Drug Targets Tested", len(numeric_cols))
with col3:
    st.metric("Dataset Rows (Cleaned)", df.shape[0])
with col4:
    st.metric("Current Filter View", selected_molecule)

with st.expander("View Raw Cleaned Data Summary"):
    st.dataframe(filtered_df.describe(), use_container_width=True)


# ----------------------------------------
# DATA VISUALIZATION
# ----------------------------------------
st.header("📈 Data Visualization & Distribution")
tab1, tab2 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])

with tab1:
    st.subheader("Distribution of Primary Metric")
    if len(numeric_cols) > 0:
        primary_metric = st.selectbox("Select metric to plot distribution:", numeric_cols, index=0)
        fig_hist = px.histogram(
            filtered_df, 
            x=primary_metric, 
            marginal="rug",  
            title=f"Distribution of {primary_metric}",
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("No numeric columns available for distribution analysis.")

with tab2:
    st.subheader("Feature Relationships")
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("Select X-axis metric:", numeric_cols, index=0)
        y_axis = st.selectbox("Select Y-axis metric:", numeric_cols, index=min(1, len(numeric_cols)-1))
        
        fig_scatter = px.scatter(
            filtered_df, 
            x=x_axis, 
            y=y_axis, 
            hover_name=molecule_col,
            title=f"{x_axis} vs {y_axis}",
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Need at least 2 numeric features to display a scatter relationship.")


# ----------------------------------------
# DRUG RANKING SYSTEM 
# ----------------------------------------
st.header("🏆 Drug Targets Ranked By Activity Score")
st.markdown("This section reshapes the dataset to order the tested drug targets from highest score (Rank 1) to lowest score for each molecule.")

orig_order_map = {mol: i for i, mol in enumerate(df[molecule_col])}

numeric_df = filtered_df.select_dtypes(include="number").copy()
numeric_df[molecule_col] = filtered_df[molecule_col]

# Melt to long format
long_df = numeric_df.melt(
    id_vars=molecule_col,
    var_name="Drug",
    value_name="Score"
).dropna(subset=["Score"])

long_df = long_df.sort_values([molecule_col, "Score"], ascending=[True, False])
long_df["Rank"] = long_df.groupby(molecule_col).cumcount() + 1

drug_wide = long_df.pivot(index=molecule_col, columns="Rank", values="Drug")
drug_wide.columns = [f"Rank_{int(c)}" for c in drug_wide.columns]
drug_wide = drug_wide.reset_index()

drug_wide["_orig_sort"] = drug_wide[molecule_col].map(orig_order_map)
drug_wide = drug_wide.sort_values("_orig_sort").drop(columns=["_orig_sort"])

csv_data = drug_wide.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Ranked Drugs CSV",
    data=csv_data,
    file_name="qsar_ranked_drugs.csv",
    mime="text/csv"
)


# ----------------------------------------
# INTERACTIVE MOLECULE QUERY FORM
# ----------------------------------------
st.header("🔍 Molecule Activity Inspector")
st.markdown("Select a specific molecule below to see a sorted breakdown of its ideal drug targets alongside their exact calculated values.")

selected_inspect = st.selectbox("Inspect specific molecule details:", all_molecules)
molecule_data = long_df[long_df[molecule_col] == selected_inspect].sort_values("Rank")

if not molecule_data.empty:
    fig_bar = px.bar(
        molecule_data,
        x="Score",
        y="Drug",
        orientation='h',
        color="Score",
        text="Rank",
        title=f"Drug Target Activity Hierarchy for Molecule: {selected_inspect}",
        color_continuous_scale="Viridis"
    )
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)
    st.table(molecule_data[["Rank", "Drug", "Score"]])
else:
    st.info("No active data entries found for this molecule.")
