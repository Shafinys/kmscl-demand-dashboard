import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import warnings
from datetime import datetime

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="KMSCL Strategic Dashboard",
    page_icon="⚕️",
    layout="wide"
)

def local_css(file_name):
    """Function to load a local CSS file for styling."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

local_css("style.css")
warnings.filterwarnings('ignore')

st.title("⚕️ KMSCL Strategic Command Center")

# ==============================================================================
# Data Loading and Caching
# ==============================================================================
@st.cache_data
def load_data(filepath):
    """Loads and prepares the enhanced drug demand data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df['Year'] = df['Date'].dt.year
        return df
    except FileNotFoundError:
        st.error(f"ERROR: Data file not found. Please ensure '{filepath}' is in the same folder.")
        return None

# Use st.cache_resource for the model itself, as it's a resource that doesn't change
@st.cache_resource
def run_prophet_forecast(_data, warehouse, drug, horizon):
    """
    Filters data, runs a Prophet forecast, and returns the forecast and model.
    Handles 'All Kerala' aggregation.
    """
    if warehouse == 'All Kerala':
        ts_df = _data[_data['Drug'] == drug].groupby('Date')['Stock_Out'].sum().reset_index()
    else:
        ts_df = _data[(_data['Warehouse'] == warehouse) & (_data['Drug'] == drug)].copy()

    if ts_df.empty or ts_df['Stock_Out'].sum() == 0:
        return None, None
        
    prophet_df = ts_df.rename(columns={'Date': 'ds', 'Stock_Out': 'y'})[['ds', 'y']]
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return forecast, model

df = load_data('synthetic_drug_demand_data_enhanced.csv')

if df is not None:
    # --- Timestamps and KPIs ---
    latest_date = df['Date'].max()
    current_date = datetime.now()
    col_date1, col_date2, _ = st.columns([1, 2, 2])
    with col_date1: st.caption(f"**Today:** {current_date.strftime('%d %B %Y')}")
    with col_date2: st.caption(f"**Data Updated Through:** {latest_date.strftime('%d %B %Y')}")
    st.header("Statewide Operations Overview")
    df_latest = df[df['Date'] == latest_date]
    total_inventory_value = df_latest['Value_Closing_Stock'].sum()
    df_recent = df[df['Date'] > latest_date - pd.Timedelta(days=90)]
    avg_daily_demand = df_recent.groupby(['Warehouse', 'Drug'])['Stock_Out'].mean().reset_index()
    avg_daily_demand.rename(columns={'Stock_Out': 'Avg_Daily_Demand'}, inplace=True)
    df_latest_with_demand = pd.merge(df_latest, avg_daily_demand, on=['Warehouse', 'Drug'], how='left').fillna(0)
    df_latest_with_demand['Days_of_Supply'] = df_latest_with_demand['Closing_Stock'] / (df_latest_with_demand['Avg_Daily_Demand'] + 0.001)
    at_risk_df = df_latest_with_demand[df_latest_with_demand['Days_of_Supply'] < 30].copy()
    at_risk_items_count = at_risk_df.shape[0]
    df_last_year = df[df['Date'] > latest_date - pd.Timedelta(days=365)]
    total_value_dispensed_yearly = df_last_year['Value_Stock_Out'].sum()
    avg_inventory_value_yearly = df_last_year['Value_Closing_Stock'].mean()
    cost_of_goods_sold_daily = total_value_dispensed_yearly / 365
    avg_days_on_hand = avg_inventory_value_yearly / (cost_of_goods_sold_daily + 0.001)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Inventory Value", f"₹{total_inventory_value/1_00_00_000:.2f} Cr")
    kpi2.metric("At-Risk SKUs (<30 Days Supply)", f"{at_risk_items_count} SKUs", delta=f"{at_risk_items_count} inventory lines need attention", delta_color="inverse")
    kpi3.metric("Average Days on Hand", f"{avg_days_on_hand:.1f} Days")
    if not at_risk_df.empty:
        with st.expander("⚠️ Click to View and Filter At-Risk SKU Details"):
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1: selected_warehouses = st.multiselect("Filter by Warehouse:", sorted(at_risk_df['Warehouse'].unique()))
            with filter_col2: selected_drugs = st.multiselect("Filter by Drug:", sorted(at_risk_df['Drug'].unique()))
            filtered_at_risk_df = at_risk_df.copy()
            if selected_warehouses: filtered_at_risk_df = filtered_at_risk_df[filtered_at_risk_df['Warehouse'].isin(selected_warehouses)]
            if selected_drugs: filtered_at_risk_df = filtered_at_risk_df[filtered_at_risk_df['Drug'].isin(selected_drugs)]
            display_cols = {'Warehouse': 'Warehouse', 'Drug': 'Drug', 'Closing_Stock': 'Stock on Hand (Units)', 'Days_of_Supply': 'Est. Days of Supply'}
            at_risk_display = filtered_at_risk_df[list(display_cols.keys())].rename(columns=display_cols)
            at_risk_display['Est. Days of Supply'] = at_risk_display['Est. Days of Supply'].astype(int)
            if at_risk_display.empty: st.warning("No at-risk items match your filter criteria.")
            else: st.dataframe(at_risk_display.sort_values(by='Est. Days of Supply'), use_container_width=True)
    st.markdown("---")

    # --- Tabs for Detailed View ---
    tab1, tab2 = st.tabs(["**📊 Demand Analytics**", "**📈 Predictive Forecasting**"])

    # ==========================================================================
    # --- DEMAND ANALYTICS TAB (Restored) ---
    # ==========================================================================
    with tab1:
        st.subheader("Interactive Demand Analysis")
        col1, _ = st.columns([2, 3])
        with col1:
            warehouse_list = ['All Kerala'] + sorted(df['Warehouse'].unique())
            selected_warehouse = st.selectbox("Select a Warehouse to analyze:", warehouse_list)

        if selected_warehouse == 'All Kerala':
            st.markdown("#### Statewide Demand Overview")
            display_df = df
        else:
            st.markdown(f"#### Demand Overview for: **{selected_warehouse}**")
            display_df = df[df['Warehouse'] == selected_warehouse]

        chart1, chart2 = st.columns(2)
        with chart1:
            st.markdown("###### Total Demand by Warehouse (Last 365 Days)")
            demand_by_warehouse = df_last_year.groupby('Warehouse')['Stock_Out'].sum().sort_values(ascending=False)
            colors = ['#1f77b4'] * len(demand_by_warehouse)
            if selected_warehouse != 'All Kerala':
                try: idx = demand_by_warehouse.index.get_loc(selected_warehouse); colors[idx] = '#ff7f0e'
                except KeyError: pass
            fig_wh = px.bar(demand_by_warehouse, x=demand_by_warehouse.index, y='Stock_Out', labels={'Stock_Out': 'Total Units Dispensed', 'x': 'Warehouse'}, text_auto='.2s')
            fig_wh.update_traces(marker_color=colors, textposition='outside')
            fig_wh.update_layout(yaxis_title="Total Units", xaxis_title=None)
            st.plotly_chart(fig_wh, use_container_width=True)

        with chart2:
            st.markdown(f"###### Top 10 Moving Drugs (Last 365 Days)")
            demand_by_drug = display_df[display_df['Date'] > latest_date - pd.Timedelta(days=365)].groupby('Drug')['Stock_Out'].sum().nlargest(10).sort_values(ascending=True)
            fig_drug = px.bar(demand_by_drug, y=demand_by_drug.index, x='Stock_Out', orientation='h', labels={'Stock_Out': 'Total Units Dispensed', 'y': ''}, text_auto='.2s', color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_drug.update_traces(textposition='outside')
            fig_drug.update_layout(xaxis_title="Total Units", yaxis_title=None)
            st.plotly_chart(fig_drug, use_container_width=True)

    # ==========================================================================
    # --- PREDICTIVE FORECASTING TAB (with Upgrades) ---
    # ==========================================================================
    with tab2:
        st.subheader("Future Demand Prediction")
        subtab1, subtab2 = st.tabs(["**Single Item Forecast**", "**Bulk Forecast Calculator**"])

        with subtab1:
            st.markdown("##### Analyze and visualize the forecast for a single SKU.")
            filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 3])
            with filter_col1:
                warehouse_list_fc = ['All Kerala'] + sorted(df['Warehouse'].unique())
                selected_warehouse_fc = st.selectbox("Select Location", warehouse_list_fc, key="fc_wh")
            with filter_col2:
                drug_list_fc = sorted(df['Drug'].unique())
                selected_drug_fc = st.selectbox("Select Drug", drug_list_fc, key="fc_drug")
            with filter_col3:
                forecast_horizon = st.slider("Forecast Horizon (Days)", 30, 365, 90, key="fc_horizon")
            
            st.markdown("---")
            st.write(f"#### Forecasting for **{selected_drug_fc}** in **{selected_warehouse_fc}**")
            with st.spinner("Training model and generating forecast..."):
                forecast_data, model = run_prophet_forecast(df, selected_warehouse_fc, selected_drug_fc, forecast_horizon)

            if forecast_data is None:
                st.error(f"No data available for the selected combination.")
            else:
                future_forecast = forecast_data.tail(forecast_horizon)
                total_predicted_demand = future_forecast['yhat'].sum()
                st.metric(label=f"Total Predicted Demand (Next {forecast_horizon} Days)", value=f"{total_predicted_demand:,.0f} units")
                
                from prophet.plot import plot_plotly
                fig_forecast = plot_plotly(model, forecast_data)
                fig_forecast.update_layout(title=f"Predicted Demand vs. Actuals", title_x=0.5)
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                with st.expander("View Forecast Data Table"):
                    forecast_display = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    forecast_display['yhat'] = forecast_display['yhat'].round(0).astype(int)
                    forecast_display['yhat_lower'] = forecast_display['yhat_lower'].round(0).astype(int)
                    forecast_display['yhat_upper'] = forecast_display['yhat_upper'].round(0).astype(int)
                    forecast_display = forecast_display.rename(columns={'ds': 'Date', 'yhat': 'Predicted Demand', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
                    st.dataframe(forecast_display, use_container_width=True)

        with subtab2:
            st.markdown("##### Calculate total predicted demand for multiple SKUs at once.")
            bulk_filter1, bulk_filter2, bulk_filter3 = st.columns([2, 2, 3])
            with bulk_filter1:
                bulk_warehouses = st.multiselect("Select Warehouses", sorted(df['Warehouse'].unique()))
            with bulk_filter2:
                bulk_drugs = st.multiselect("Select Drugs", sorted(df['Drug'].unique()))
            with bulk_filter3:
                bulk_horizon = st.slider("Forecast Horizon (Days)", 30, 180, 60, key="bulk_horizon")

            if st.button("Run Bulk Forecast", type="primary"):
                if not bulk_warehouses or not bulk_drugs:
                    st.warning("Please select at least one warehouse and one drug.")
                else:
                    results = []
                    total_combinations = len(bulk_warehouses) * len(bulk_drugs)
                    progress_bar = st.progress(0, text="Initializing bulk forecast...")
                    
                    for i, wh in enumerate(bulk_warehouses):
                        for j, drug in enumerate(bulk_drugs):
                            current_prog = (i * len(bulk_drugs) + j + 1) / total_combinations
                            progress_bar.progress(current_prog, text=f"Forecasting {drug} in {wh}... ({i * len(bulk_drugs) + j + 1}/{total_combinations})")
                            
                            forecast_data, _ = run_prophet_forecast(df, wh, drug, bulk_horizon)
                            
                            if forecast_data is not None:
                                total_demand = forecast_data.tail(bulk_horizon)['yhat'].sum()
                                results.append({
                                    "Warehouse": wh,
                                    "Drug": drug,
                                    f"Predicted Demand (Next {bulk_horizon} Days)": int(total_demand)
                                })
                    
                    progress_bar.progress(1.0, text="Bulk forecast complete!")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
else:
    st.error("Dashboard could not be loaded. Please ensure the enhanced data file is available.")

