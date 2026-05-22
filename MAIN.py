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

By calculating and ranking activity scores, this tool helps researchers instantly identify which drug targets are most effective for a given molecule. Use the sidebar to upload data, filter specific molecules, and explore molecular distributions, feature relationships, and ranking matrices in real time.
""")

st.sidebar.header("Setup & Filters")

# ----------------------------------------
# DATA LOADING & CLEANING (Functionality 1)
# ----------------------------------------
@st.cache_data
def load_and_clean_data(file_path):
    try:
        # Load data directly from the web URL
        df = pd.read_csv(file_path)
    except Exception as e:
        return None
    
    # Data Cleaning
    df.drop_duplicates(inplace=True)
    return df

# Your Google Drive view link converted into a direct download stream URL
GOOGLE_DRIVE_CSV_URL = "https://drive.google.com/uc?export=download&id=1fnPYHehOo-DhgJRCjGD07ozM7Tw75CjL"

# Add a visual loading spinner since a 200MB file takes a moment to fetch
with st.spinner("Downloading and processing dataset from Google Drive... Please wait."):
    df = load_and_clean_data(GOOGLE_DRIVE_CSV_URL)

if df is None:
    st.error("⚠️ Failed to load data from Google Drive. Please verify your connection or your file link sharing permissions.")
    st.stop()

molecule_col = df.columns[0]
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# Sidebar: Filter Molecule globally or view all
all_molecules = df[molecule_col].unique().tolist()
selected_molecule = st.sidebar.selectbox("Select a Specific Molecule to Focus On:", ["All"] + all_molecules)

# Filter dataset based on selection
filtered_df = df if selected_molecule == "All" else df[df[molecule_col] == selected_molecule]

# ----------------------------------------
# METRICS & KEY STATISTICS (Functionality 2)
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
# DATA VISUALIZATION (Functionality 3)
# ----------------------------------------
st.header("📈 Data Visualization & Distribution")

tab1, tab2 = st.tabs(["Univariate Analysis", "Bivariate Analysis"])

with tab1:
    st.subheader("Distribution of Primary Metric")
    if len(numeric_cols) > 0:
        primary_metric = st.selectbox("Select metric to plot distribution:", numeric_cols, index=0)
        
        # Plotly Express histogram
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
# DRUG RANKING SYSTEM (Functionality 4)
# ----------------------------------------
st.header("🏆 Drug Targets Ranked By Activity Score")
st.markdown("This section reshapes the dataset to order the tested drug targets from highest score (Rank 1) to lowest score for each molecule.")

# Perform processing dynamically based on user selections
orig_order_map = {mol: i for i, mol in enumerate(df[molecule_col])}

numeric_df = filtered_df.select_dtypes(include="number").copy()
numeric_df[molecule_col] = filtered_df[molecule_col]

# Melt to long format
long_df = numeric_df.melt(
    id_vars=molecule_col,
    var_name="Drug",
    value_name="Score"
).dropna(subset=["Score"])

# Sort by molecule and highest score
long_df = long_df.sort_values([molecule_col, "Score"], ascending=[True, False])
long_df["Rank"] = long_df.groupby(molecule_col).cumcount() + 1

# Pivot back to wide format for the ranking matrix
drug_wide = long_df.pivot(index=molecule_col, columns="Rank", values="Drug")
drug_wide.columns = [f"Rank_{int(c)}" for c in drug_wide.columns]
drug_wide = drug_wide.reset_index()

# Re-align original dataframe order
drug_wide["_orig_sort"] = drug_wide[molecule_col].map(orig_order_map)
drug_wide = drug_wide.sort_values("_orig_sort").drop(columns=["_orig_sort"])

# Allow download of the processed rankings
csv_data = drug_wide.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Ranked Drugs CSV",
    data=csv_data,
    file_name="qsar_ranked_drugs.csv",
    mime="text/csv"
)


# ----------------------------------------
# INTERACTIVE MOLECULE QUERY FORM (Functionality 5)
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
    # Reverse y-axis so Rank 1 stays at the top of the bar chart
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.table(molecule_data[["Rank", "Drug", "Score"]])
else:
    st.info("No active data entries found for this molecule.")
