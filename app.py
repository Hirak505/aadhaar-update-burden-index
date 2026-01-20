import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Project Sentinel – Admin Dashboard", layout="wide")


@st.cache_data
def load_results():
    return pd.read_csv("data/processed/statewise_anomaly_alerts.csv")


df = load_results()

st.title("🛡️ Project Sentinel – Admin Monitoring Dashboard")
st.caption("Visual Identification of Normal, Hot Spot & Dark Spot Districts")

st.sidebar.header("Filters")

state_filter = st.sidebar.selectbox(
    "Select State", ["All"] + sorted(df["state"].unique())
)

severity_filter = st.sidebar.multiselect(
    "Risk Severity", ["High", "Medium", "Normal"], default=["High", "Medium"]
)

type_filter = st.sidebar.multiselect(
    "Anomaly Type",
    ["Hot Spot", "Dark Spot", "Normal"],
    default=["Hot Spot", "Dark Spot"],
)

filtered = df.copy()

if state_filter != "All":
    filtered = filtered[filtered["state"] == state_filter]

filtered = filtered[filtered["risk_severity"].isin(severity_filter)]
filtered = filtered[filtered["anomaly_type"].isin(type_filter)]

st.header("📊 National Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", len(filtered))
col2.metric("🔴 High Risk", (filtered["risk_severity"] == "High").sum())
col3.metric("🟠 Medium Risk", (filtered["risk_severity"] == "Medium").sum())
col4.metric("🔵 Normal", (filtered["risk_severity"] == "Normal").sum())

st.subheader("🚩 District Risk Status Overview")

status_counts = df["risk_severity"].value_counts()

fig1, ax1 = plt.subplots()
ax1.bar(status_counts.index, status_counts.values, color=["red", "orange", "green"])
ax1.set_ylabel("Number of District Records")
ax1.set_title("Normal vs Red-Flagged Districts")
st.pyplot(fig1)

st.subheader("🔥 Hot Spots vs 🌑 Dark Spots")

type_counts = df["anomaly_type"].value_counts()

fig2, ax2 = plt.subplots()
ax2.bar(type_counts.index, type_counts.values, color=["red", "blue", "green"])
ax2.set_ylabel("Count")
ax2.set_title("Anomaly Type Distribution")
st.pyplot(fig2)

st.subheader("📈 Z-Score Deviation (Anomaly Intensity)")

fig3, ax3 = plt.subplots(figsize=(10, 4))

colors = []
for _, row in filtered.iterrows():
    if row["anomaly_type"] == "Hot Spot":
        colors.append("red")
    elif row["anomaly_type"] == "Dark Spot":
        colors.append("blue")
    else:
        colors.append("green")

ax3.scatter(filtered["district"], filtered["z_score"], c=colors)

ax3.axhline(3, linestyle="--", color="black", label="High Threshold")
ax3.axhline(2, linestyle=":", color="black", label="Medium Threshold")
ax3.axhline(-2, linestyle=":", color="black")
ax3.axhline(-3, linestyle="--", color="black")

ax3.set_ylabel("Z-Score")
ax3.set_xlabel("District")
ax3.set_title("District Deviation from State Baseline")
ax3.tick_params(axis="x", rotation=90)
ax3.legend()

st.pyplot(fig3)

st.subheader("🗺️ State-wise Anomaly Density")

heatmap_df = df.groupby(["state", "risk_severity"]).size().unstack(fill_value=0)

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_df, cmap="Reds", annot=True, fmt="d", ax=ax4)
ax4.set_title("Anomaly Density by State and Severity")
st.pyplot(fig4)

st.subheader("🚨 Top Red-Flagged Districts")

top_red = df[df["risk_severity"] == "High"].sort_values("z_score", ascending=False)

st.dataframe(
    top_red[
        [
            "state",
            "district",
            "time_period",
            "total_activity",
            "z_score",
            "anomaly_type",
            "final_flag",
            "recommendation",
        ]
    ].head(20)
)

with st.expander("📋 View Full Anomaly Report"):
    st.dataframe(df)

st.markdown("---")
st.caption("Project Sentinel | Visualization Layer")
