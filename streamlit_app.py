import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide", page_title="Disaster Forecast Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("results/all_incidents_2025.csv", parse_dates=["Date"])

df = load_data()

df["LSTM %"] = (df["LSTM Predictions"] / df["LSTM Predictions"].max()) * 100
df["XGBoost %"] = (df["XGBoost Predictions"] / df["XGBoost Predictions"].max()) * 100
df["Hybrid %"] = (df["Hybrid Predictions"] / df["Hybrid Predictions"].max()) * 100

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
        .dropdown-container {
            padding: 0.5rem 1rem;
        }
        .chart-box {
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 10px;
            height: 100%;
        }
    </style>
""", unsafe_allow_html=True)

st.title("2025 Disaster Forecast")

st.sidebar.header("Filters")
incident_type = st.sidebar.selectbox("Select Incident Type", df["Incident Type"].unique())
model_selection = st.sidebar.radio("Select Model", ["LSTM Predictions", "XGBoost Predictions", "Hybrid Predictions"])

col_filter1, col_filter2 = st.columns([3, 2])

with col_filter1:
    st.markdown("### Incident Choropleth Map")

with col_filter2:
    st.markdown("### Top N Risky States (This Month)")
    month_options = df["Date"].dt.to_period("M").unique().astype(str)
    selected_month = st.selectbox("Select Month", sorted(month_options), key="month_filter")

col1, col2 = st.columns([3, 2])

with col1:
    filtered = df[df["Incident Type"] == incident_type]
    color_scale_map = {
        "Fire": "OrRd",
        "Flood": "Blues",
        "Hurricane": "YlGnBu",
        "Severe Storm": "Purples",
    }
    color_scale = color_scale_map.get(incident_type, "Viridis")
    fig1 = px.choropleth(
        filtered,
        locations="State",
        locationmode="USA-states",
        color=model_selection,
        animation_frame=filtered["Date"].dt.strftime("%b-%Y"),
        scope="usa",
        color_continuous_scale=color_scale,
    )
    fig1.update_layout(
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    filtered_month = filtered[filtered["Date"].dt.to_period("M").astype(str) == selected_month]
    top_states = filtered_month.groupby("State")[model_selection].sum().nlargest(10).reset_index()
    fig2 = px.pie(
        top_states,
        names="State",
        values=model_selection,
        title=f"Top 10 Risky States in {selected_month} ({model_selection})",
        color_discrete_sequence=px.colors.sequential.__getattribute__(color_scale)
    )
    fig2.update_traces(textinfo='percent+label', hole=0.4)
    fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# DROPDOWNS for Row 2
col_filter3, col_filter4 = st.columns([3, 2])

with col_filter3:
    st.markdown("### Incident Data Table")
    available_states = df[df["Incident Type"] == incident_type]["State"].unique()
    state_filter = st.selectbox("Select State", sorted(available_states))

with col_filter4:
    st.markdown("### Prediction Range (Max - Min)")

col3, col4 = st.columns([3, 2])

with col3:
    table_df = df[(df["State"] == state_filter) & (df["Incident Type"] == incident_type)].copy()
    table_df["Month"] = table_df["Date"].dt.strftime("%b %Y")
    table_df_display = table_df[["Month", "Incident Type", "State", model_selection]].sort_values("Month")
    st.dataframe(table_df_display, use_container_width=True)

with col4:
    box_df = df[df["Incident Type"] == incident_type]
    range_df = box_df.groupby("State")[model_selection].agg(["max", "min"]).reset_index()
    range_df["Range"] = range_df["max"] - range_df["min"]
    fig4 = px.bar(
        range_df,
        x="State",
        y="Range",
        title=f"Prediction Range (Max - Min) of {model_selection} for {incident_type}",
        color="Range",
        color_continuous_scale=color_scale
    )
    fig4.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig4, use_container_width=True)
