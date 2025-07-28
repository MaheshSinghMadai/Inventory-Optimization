import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Set page configuration
st.set_page_config(layout="wide", page_title="Inventory Demand Forecasting")

# Cache data loading
@st.cache_data
def load_data():
    # Load historical daily sales from online_retail.csv
    try:
        df = pd.read_csv('online_retail.csv', encoding='latin-1')
        df = df.dropna(subset=['StockCode', 'Quantity', 'InvoiceDate'])
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df = df[df['Quantity'] > 0]
        daily_sales = df.groupby(['StockCode', pd.Grouper(key='InvoiceDate', freq='D')])['Quantity'].sum().reset_index()
        st.write(f"Loaded and processed historical data shape: {daily_sales.shape}")
    except FileNotFoundError:
        st.error("online_retail.csv not found. Please ensure the file is in the working directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading online_retail.csv: {str(e)}")
        st.stop()

    # Load precomputed forecasts from CSV files
    forecasts_dir = 'forecasts'
    if not os.path.exists(forecasts_dir):
        st.error(f"Directory {forecasts_dir} does not exist")
        st.stop()

    arima_forecasts = {}
    xgb_forecasts = {}
    try:
        for f in os.listdir(forecasts_dir):
            if f.startswith('arima_forecast_') and f.endswith('.csv'):
                stockcode = f.replace('arima_forecast_', '').replace('.csv', '')
                df = pd.read_csv(os.path.join(forecasts_dir, f))
                df['Date'] = pd.to_datetime(df['Date'])
                arima_forecasts[stockcode] = df
            elif f.startswith('xgb_forecast_') and f.endswith('.csv'):
                stockcode = f.replace('xgb_forecast_', '').replace('.csv', '')
                df = pd.read_csv(os.path.join(forecasts_dir, f))
                df['Date'] = pd.to_datetime(df['Date'])
                xgb_forecasts[stockcode] = df
    except Exception as e:
        st.error(f"Error loading forecast files: {str(e)}")
        st.stop()

    return daily_sales, arima_forecasts, xgb_forecasts

# Load performance metrics from CSV
metrics_file = os.path.join('metrics', 'performance_metrics.csv')
try:
    metrics_df = pd.read_csv(metrics_file)
    performance_metrics = {row['Model']: {'MAE': 0, 'RMSE': 0, 'MAPE': row['Average_MAPE (%)']} for index, row in metrics_df.iterrows()}
except FileNotFoundError:
    st.error(f"Performance metrics file {metrics_file} not found")
    st.stop()
except Exception as e:
    st.error(f"Error loading performance_metrics.csv: {str(e)}")
    st.stop()

# Load data once
daily_sales, arima_forecasts, xgb_forecasts = load_data()

# Inventory optimization function for item-level
@st.cache_data
def calculate_reorder_point_item(forecast, lead_time_days=2, safety_stock=50):
    # Convert lead_time_days to weeks' worth of data (assuming 7 days per week)
    lead_time_weeks = lead_time_days // 7 if lead_time_days >= 7 else 1
    lead_time_demand = forecast['predicted_sales'][:lead_time_weeks].sum()
    reorder_point = lead_time_demand + safety_stock
    return reorder_point

# Initialize session state
if 'reorder_points' not in st.session_state:
    st.session_state.reorder_points = {}
if 'lead_time' not in st.session_state:
    st.session_state.lead_time = 2
if 'safety_stock' not in st.session_state:
    st.session_state.safety_stock = 50

# Sidebar for inputs
st.sidebar.header("Settings")
# Get top 10 StockCodes by frequency
stockcode_counts = daily_sales['StockCode'].value_counts()
stockcodes = sorted(list(set(arima_forecasts.keys()).union(set(xgb_forecasts.keys()))))
if not stockcodes:
    st.sidebar.error("No stock codes found in daily_sales.")
    st.stop()
selected_stockcode = st.sidebar.selectbox("Select StockCode", stockcodes, key="stockcode")
lead_time_days = st.sidebar.slider("Lead Time (days)", 1, 7, 2, key="lead_time_input")
safety_stock = st.sidebar.number_input("Safety Stock per Item (units)", min_value=0, value=50, step=10, key="safety_stock_input")

# Title
st.title("Inventory Demand Forecasting UI")

# Display demand, forecast, and reorder point for selected StockCode
if selected_stockcode:
    try:
        # Determine best model based on lower MAPE from performance_metrics
        use_arima = performance_metrics['ARIMA']['MAPE'] < performance_metrics['XGBoost']['MAPE']
        best_model = 'ARIMA' if use_arima else 'XGBoost'
        forecast = arima_forecasts.get(selected_stockcode) if use_arima else xgb_forecasts.get(selected_stockcode)
        if forecast is None:
            st.error(f"No {best_model} forecast available for StockCode {selected_stockcode}")
            st.stop()

        # Historical Demand
        st.header("1. Historical Demand Trend")
        historical_data = daily_sales[daily_sales['StockCode'] == selected_stockcode].set_index('InvoiceDate')['Quantity']
        historical_data.index = pd.to_datetime(historical_data.index)
        fig1, ax1 = plt.subplots(figsize=(15, 6))
        ax1.plot(historical_data.index, historical_data / 1000, label=f'Historical: StockCode {selected_stockcode}', color='blue')
        ax1.set_title(f'Historical Demand for StockCode {selected_stockcode}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Quantity (Thousands)')
        ax1.legend()
        ax1.grid(True)
        plt.tight_layout()
        st.pyplot(fig1)
        st.write("Historical sales trend (quantity in thousands) from 2010-12-01 to 2011-12-09.")

        # Forecasted Demand
        st.header("2. Forecasted Demand Trend")
        fig2, ax2 = plt.subplots(figsize=(15, 6))
        ax2.plot(forecast['Date'], forecast['predicted_sales'] / 1000, label=f'{best_model} Forecast', color='red' if use_arima else 'green', linestyle='--')
        ax2.set_title(f'Forecasted Demand for StockCode {selected_stockcode}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Quantity (Thousands)')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        st.pyplot(fig2)
        st.write(f"Forecasted demand using {best_model} (quantity in thousands) for the next 7 days.")

        # Reorder Point
        st.header("3. Reorder Point")
        if (st.session_state.reorder_points.get(selected_stockcode) is None or
            st.session_state.lead_time != lead_time_days or
            st.session_state.safety_stock != safety_stock):
            st.session_state.reorder_points[selected_stockcode] = calculate_reorder_point_item(forecast, lead_time_days, safety_stock)
            st.session_state.lead_time = lead_time_days
            st.session_state.safety_stock = safety_stock
        st.write(f"**Reorder Point for StockCode {selected_stockcode}:** {st.session_state.reorder_points[selected_stockcode]:,.2f} units")
        st.write(f"**Lead Time:** {lead_time_days} days")
        st.write(f"**Safety Stock:** {safety_stock:,.2f} units")
        st.write(f"**Best Model:** {best_model} (Lower MAPE: {performance_metrics[best_model]['MAPE']:.2f}%)")
    except Exception as e:
        st.error(f"Error processing StockCode {selected_stockcode}: {str(e)}")

# Comparison of Two Models
st.header("4. Comparison of Two Models")
fig3, ax3 = plt.subplots(figsize=(12, 6))
historical_total = daily_sales.groupby('InvoiceDate')['Quantity'].sum().reset_index()
ax3.plot(historical_total['InvoiceDate'], historical_total['Quantity'] / 1000, label='Historical Sales', color='blue')
if arima_forecasts:
    total_arima = pd.concat(arima_forecasts.values()).groupby('Date')['predicted_sales'].sum().reset_index()
    ax3.plot(total_arima['Date'], total_arima['predicted_sales'] / 1000, label='ARIMA Forecast', color='skyblue', linestyle='--')
if xgb_forecasts:
    total_xgb = pd.concat(xgb_forecasts.values()).groupby('Date')['predicted_sales'].sum().reset_index()
    ax3.plot(total_xgb['Date'], total_xgb['predicted_sales'] / 1000, label='XGBoost Forecast', color='lightgreen', linestyle='--')
ax3.set_xlabel('Date')
ax3.set_ylabel('Sales (Thousands)')
ax3.set_title('Historical vs. Forecasted Sales (All StockCodes)')
ax3.legend()
ax3.grid(True)
plt.tight_layout()
st.pyplot(fig3)
st.write(f"ARIMA MAPE: {performance_metrics['ARIMA']['MAPE']:.2f}%, XGBoost MAPE: {performance_metrics['XGBoost']['MAPE']:.2f}%. Note: Values from performance_metrics.csv.")

# Performance Comparison Charts
st.header("5. Performance Comparison of Models")
fig4, (ax_mse, ax_mape) = plt.subplots(1, 2, figsize=(12, 5))
width = 0.35
x = np.arange(2)

# MSE Chart
ax_mse.bar(x[0], metrics_df[metrics_df['Model'] == 'ARIMA']['Average_MSE'].values[0] / 1e6, width, label='ARIMA', color='skyblue')
ax_mse.bar(x[1], metrics_df[metrics_df['Model'] == 'XGBoost']['Average_MSE'].values[0] / 1e6, width, label='XGBoost', color='lightgreen')
ax_mse.set_xlabel('Models')
ax_mse.set_ylabel('MSE (Millions)')
ax_mse.set_title('Mean Squared Error Comparison')
ax_mse.set_xticks(x)
ax_mse.set_xticklabels(['ARIMA', 'XGBoost'])
ax_mse.legend()
ax_mse.grid(True, which='both', linestyle='--', alpha=0.7)

# MAPE Chart
ax_mape.bar(x[0], metrics_df[metrics_df['Model'] == 'ARIMA']['Average_MAPE (%)'].values[0], width, label='ARIMA', color='skyblue')
ax_mape.bar(x[1], metrics_df[metrics_df['Model'] == 'XGBoost']['Average_MAPE (%)'].values[0], width, label='XGBoost', color='lightgreen')
ax_mape.set_xlabel('Models')
ax_mape.set_ylabel('MAPE (%)')
ax_mape.set_title('Mean Absolute Percentage Error Comparison')
ax_mape.set_xticks(x)
ax_mape.set_xticklabels(['ARIMA', 'XGBoost'])
ax_mape.legend()
ax_mape.grid(True, which='both', linestyle='--', alpha=0.7)

plt.tight_layout()
st.pyplot(fig4)
st.write("MSE is scaled to millions (1e6) for readability. Values are from performance_metrics.csv.")
