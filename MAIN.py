import streamlit as st
import polars as pl
import plotly.express as px

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
**Welcome to the QSAR Data Dashboard.** This application is optimized to analyze massive Quantitative Structure-Activity Relationship (QSAR) datasets efficiently.
""")

st.sidebar.header("Setup & Filters")

# ----------------------------------------
# DATA LOADING & CLEANING (Polars-Optimized)
# ----------------------------------------
@st.cache_data
def load_and_clean_data(file_path):
    try:
        # Polars scans files instantly to find columns
        preview = pl.read_csv(file_path, n_rows=1)
        molecule_col = preview.columns[0]
        
        # Identify numeric columns
        numeric_cols = [col for col in preview.columns if preview[col].dtype.is_numeric()]
        keep_cols = [molecule_col] + numeric_cols
        
        # Read file with only the necessary columns and drop duplicates efficiently
        df = pl.read_csv(file_path, columns=keep_cols)
        df = df.unique()
        
        # Cast numeric columns to float32 to save 50% memory
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in numeric_cols])
        return df
    except Exception as e:
        return None

GITHUB_RELEASE_URL = "https://github.com/morlockmaimai-ui/Drug-Target-Activity-Analyzer-/releases/download/v1.0.0/QSAR.csv"

with st.spinner("Streaming dataset efficiently via Polars... Please wait."):
    df = load_and_clean_data(GITHUB_RELEASE_URL)

if df is None:
    st.error("⚠️ Failed to load data. Verify your Release Link format or check if your repository is public.")
    st.stop()

# Track schema information
molecule_col = df.columns[0]
numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]

# Sidebar Filter
all_molecules = df[molecule_col].unique().to_list()
selected_molecule = st.sidebar.selectbox("Select a Specific Molecule to Focus On:", ["All"] + all_molecules)

# Filter safely
if selected_molecule != "All":
    filtered_df = df.filter(pl.col(molecule_col) == selected_molecule)
else:
    filtered_df = df

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
    st.metric("Dataset Rows (Cleaned)", df.height)
with col4:
    st.metric("Current Filter View", selected_molecule)

with st.expander("View Raw Cleaned Data Summary"):
    # Describe uses minimal memory and converts to pandas only for display
    st.dataframe(filtered_df.describe().to_pandas(), use_container_width=True)

# ----------------------------------------
# DATA VISUALIZATION (Memory Protected)
# ----------------------------------------
st.header("📈 Data Visualization & Distribution")
tab1, tab2 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])

# To protect the browser and server from crashing, sample the data for plotting if it's huge
MAX_PLOT_ROWS = 25000
if filtered_df.height > MAX_PLOT_ROWS:
    plot_df = filtered_df.sample(n=MAX_PLOT_ROWS, seed=42)
    st.caption(f"⚠️ Data contains {filtered_df.height} rows. Plot dynamically sampled to {MAX_PLOT_ROWS} rows to prevent memory crashes.")
else:
    plot_df = filtered_df

with tab1:
    st.subheader("Distribution of Primary Metric")
    if len(numeric_cols) > 0:
        primary_metric = st.selectbox("Select metric to plot distribution:", numeric_cols, index=0)
        
        # Plot using native pandas conversion right at chart generation
        fig_hist = px.histogram(
            plot_df.select([primary_metric]).to_pandas(), 
            x=primary_metric, 
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
            plot_df.select([molecule_col, x_axis, y_axis]).to_pandas(), 
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
# DRUG RANKING SYSTEM (Optimized Long-Melt)
# ----------------------------------------
st.header("🏆 Drug Targets Ranked By Activity Score")
st.markdown("This section reshapes the dataset to order the tested drug targets from highest score (Rank 1) to lowest score for each molecule.")

# Perform Melt and Rank operations inside Polars (uses significantly less memory than pandas)
long_df_pl = (
    filtered_df.unpivot(index=molecule_col, on=numeric_cols, variable_name="Drug", value_name="Score")
    .drop_nulls(subset=["Score"])
    .sort([molecule_col, "Score"], descending=[False, True])
)

# Calculate Rank grouping by Molecule
long_df_pl = long_df_pl.with_columns(
    pl.int_range(1, pl.len() + 1).over(molecule_col).alias("Rank")
)

# Compute wide dynamic ranking table only if safe, or display optimization message
if filtered_df.height < 5000:
    drug_wide_pl = (
        long_df_pl.with_columns(pl.format("Rank_{}", pl.col("Rank")).alias("Rank_Str"))
        .pivot(on="Rank_Str", index=molecule_col, values="Drug")
    )
    # Keep original file sorting order
    orig_order = filtered_df.select(molecule_col)
    drug_wide_pl = orig_order.join(drug_wide_pl, on=molecule_col, how="left")
    
    csv_data = drug_wide_pl.write_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Ranked Drugs CSV",
        data=csv_data,
        file_name="qsar_ranked_drugs.csv",
        mime="text/csv"
    )
else:
    st.warning("⚠️ Full reshape-to-wide table for downloding is blocked because your active workspace data is too massive. Filter your molecule focus on the sidebar to enable download features.")

# ----------------------------------------
# INTERACTIVE MOLECULE QUERY FORM
# ----------------------------------------
st.header("🔍 Molecule Activity Inspector")
st.markdown("Select a specific molecule below to see a sorted breakdown of its ideal drug targets alongside their exact calculated values.")

selected_inspect = st.selectbox("Inspect specific molecule details:", all_molecules)

# Filter the already-melted Polars dataframe directly 
molecule_data_pl = long_df_pl.filter(pl.col(molecule_col) == selected_inspect).sort("Rank")
molecule_data_pd = molecule_data_pl.to_pandas()

if not molecule_data_pd.empty:
    fig_bar = px.bar(
        molecule_data_pd,
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
    st.table(molecule_data_pd[["Rank", "Drug", "Score"]])
else:
    st.info("No active data entries found for this molecule.")
