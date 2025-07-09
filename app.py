import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Set page configuration
st.set_page_config(layout="wide", page_title="Inventory Demand Forecasting")

# Sample data (replace with your actual DataFrames)
# Historical data
df_weekly = pd.read_csv('weekly_sales.csv')  # Ensure this file exists with appropriate data
df_weekly['Date'] = pd.to_datetime(df_weekly['Date'])

# XGBoost forecast
future_forecast_xgboost = pd.read_csv('xgb_sales_forecast.csv')  # Ensure this file exists with appropriate data
future_forecast_xgboost['Date'] = pd.to_datetime(future_forecast_xgboost['Date'])

# ARIMA forecast (replace with actual ARIMA data)
arima_forecast = pd.read_csv('arima_sales_forecast.csv')  # Ensure this file exists with appropriate data
arima_forecast['Date'] = pd.to_datetime(arima_forecast['Date'])


# MAPE values
mape_arima = 7.98  # %
mape_xgb = 6.14    # %

# Inventory optimization function
def calculate_reorder_point(forecast, lead_time_weeks=2, safety_stock=500000):
    lead_time_demand = forecast['predicted_sales'][:lead_time_weeks].sum()
    reorder_point = lead_time_demand + safety_stock
    return reorder_point

# Sidebar for inputs
st.sidebar.header("Settings")
lead_time_weeks = st.sidebar.slider("Lead Time (weeks)", 1, 4, 2)
safety_stock = st.sidebar.number_input("Safety Stock (units)", min_value=0, value=500000, step=100000)

# Title
st.title("Inventory Demand Forecasting")

# 1. Current Sales Trend
st.header("1. Current Sales Trend")
st.line_chart(df_weekly.set_index('Date')['Sales'] / 1e7, use_container_width=True)
st.write("Historical sales trend from 2013 to 2015-07-31, scaled to 1e7 units.")

# 2. Demanded Sales Trend
st.header("2. Demanded Sales Trend")
st.subheader("XGBoost Forecast")
st.line_chart(future_forecast_xgboost.set_index('Date')['predicted_sales'] / 1e7, use_container_width=True)
st.write("Forecasted demand using XGBoost for the next 4 weeks, scaled to 1e7 units.")

# 3. Comparison of Two Models
st.header("3. Comparison of Two Models")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(arima_forecast['Date'], arima_forecast['predicted_sales'] / 1e7, label='ARIMA Forecast', color='blue')
ax.plot(future_forecast_xgboost['Date'], future_forecast_xgboost['predicted_sales'] / 1e7, label='XGBoost Forecast', color='orange')
ax.set_xlabel('Date')
ax.set_ylabel('Sales (1e7)')
ax.set_title('ARIMA vs. XGBoost Forecast')
ax.legend()
ax.grid(True)
st.pyplot(fig)
st.write(f"ARIMA MAPE: {mape_arima}%, XGBoost MAPE: {mape_xgb}%. Lower MAPE indicates better accuracy.")

# 4. UI of Reorder Points
st.header("4. Reorder Points")
if mape_arima <= mape_xgb:
    st.write("Using ARIMA forecast for inventory optimization (better MAPE)")
    forecast_for_inventory = arima_forecast
else:
    st.write("Using XGBoost forecast for inventory optimization (better MAPE)")
    forecast_for_inventory = future_forecast_xgboost

reorder_point = calculate_reorder_point(forecast_for_inventory, lead_time_weeks, safety_stock)
st.write(f"**Reorder Point:** {reorder_point:,.2f} units")
st.write(f"**Lead Time:** {lead_time_weeks} weeks")
st.write(f"**Safety Stock:** {safety_stock:,.2f} units")
# st.write(f"**Current Date and Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S %z')}")

# Run the app
if __name__ == "__main__":
    st.rerun()