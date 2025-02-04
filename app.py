import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore, skew

# Page configuration
st.set_page_config(page_title="Insurance Premium Forecasting", layout="wide")

# App title and description
st.title("Insurance Premium Forecasting System")
st.markdown("""
This app performs comprehensive analysis and forecasting of insurance premiums using:
- Time series decomposition
- Outlier detection
- Stationarity checks
- Facebook Prophet forecasting
""")

# Sidebar controls
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "zip"])
    
    st.header("Filter Options")
    state_options = ['TO', 'SP', 'RJ', 'MG', 'BA']
    selected_state = st.selectbox("Select State", state_options)
    
    st.header("Preprocessing Options")
    handle_missing = st.radio("Handle Missing Values", ['drop', 'ffill'])
    remove_outliers = st.checkbox("Remove Outliers (IQR Method)")
    scale_data = st.checkbox("Normalize Data (MinMax Scaling)")
    
    st.header("Prophet Configuration")
    seasonality_mode = st.selectbox("Seasonality Mode", ['additive', 'multiplicative'])
    changepoint_scale = st.slider("Changepoint Flexibility", 0.001, 0.5, 0.05)
    forecast_periods = st.number_input("Forecast Months", 1, 36, 12)

# Data loading and preprocessing
@st.cache
def load_and_filter_data(uploaded_file, selected_state):
    if uploaded_file is not None:
         if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                # Extract the first file in the zip (assuming it contains only one CSV)
                file_name = zip_ref.namelist()[0]
                with zip_ref.open(file_name) as f:
                    df = pd.read_csv(f)
        else:
            df = pd.read_csv(uploaded_file,encoding='ISO-8859-1')
        
        # Apply filters
        filtered_df = df[
            (df['company_code'] == 6785) &
            (df['product'] == '0993 - Vida em Grupo') &
            (df['state'] == selected_state)
        ]
        
        # Convert and set datetime index
        filtered_df['year_month_f'] = pd.to_datetime(filtered_df['year_month'])
        filtered_df.set_index('year_month_f', inplace=True)
        filtered_df = filtered_df.asfreq("MS").sort_index()
        
        return filtered_df[['premiums']]
    return None

if uploaded_file:
    data = load_and_filter_data(uploaded_file, selected_state)
    
    # Handle missing values
    if handle_missing == 'drop':
        data = data.dropna()
    else:
        data = data.ffill()
    
    # Data Exploration Section
    st.header("Data Exploration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(data.tail(), height=200)
    
    with col2:
        st.subheader("Basic Statistics")
        st.write(data.describe())
    
    # Outlier Handling
    if remove_outliers:
        Q1 = data['premiums'].quantile(0.25)
        Q3 = data['premiums'].quantile(0.75)
        IQR = Q3 - Q1
        data = data[(data['premiums'] >= Q1 - 1.5*IQR) & (data['premiums'] <= Q3 + 1.5*IQR)]
        st.success(f"Outliers removed using IQR method. New shape: {data.shape}")
    
    # Data Scaling
    if scale_data:
        scaler = MinMaxScaler()
        data[['premiums']] = scaler.fit_transform(data[['premiums']])
        st.success("Data normalized using MinMax scaling")
    
    # Time Series Analysis
    st.header("Time Series Analysis")
    
    # Rolling Statistics Plot
    rolling_window = st.slider("Rolling Window Size", 3, 24, 12)
    rolling_mean = data.rolling(window=rolling_window).mean()
    rolling_std = data.rolling(window=rolling_window).std()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['premiums'], name='Original'))
    fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean['premiums'], name='Rolling Mean'))
    fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std['premiums'], name='Rolling Std'))
    st.plotly_chart(fig, use_container_width=True)
    
    # Stationarity Check
    st.subheader("Stationarity Analysis")
    adf_result = adfuller(data['premiums'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ADF Statistic", f"{adf_result[0]:.2f}")
    col2.metric("p-value", f"{adf_result[1]:.4f}")
    col3.metric("Critical Value (1%)", f"{adf_result[4]['1%']:.2f}")
    
    # Differencing if non-stationary
    if adf_result[1] > 0.05:
        data['premiums'] = data['premiums'].diff().dropna()
        st.warning("Applied differencing to make data stationary")
    
    # Prophet Model Setup
    st.header("Prophet Forecasting")
    
    if st.button("Train Forecast Model"):
        with st.spinner("Training Prophet model..."):
            # Prepare Prophet data
            prophet_df = data.reset_index().rename(columns={'year_month_f': 'ds', 'premiums': 'y'})
            
            # Split data
            train_size = int(len(prophet_df) * 0.8)
            train = prophet_df[:train_size]
            test = prophet_df[train_size:]
            
            # Initialize and train model
            model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_scale,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            
            model.fit(train)
            
            # Generate forecasts
            future = model.make_future_dataframe(periods=forecast_periods, freq='MS')
            forecast = model.predict(future)
            
            # Store results in session state
            st.session_state.model = model
            st.session_state.forecast = forecast
            st.session_state.test = test
            
            st.success("Model training completed!")

    # Display results if model exists
    if 'model' in st.session_state:
        # Forecast Visualization
        st.subheader("Forecast Results")
        fig1 = plot_plotly(st.session_state.model, st.session_state.forecast)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Components Plot
        st.subheader("Forecast Components")
        fig2 = plot_components_plotly(st.session_state.model, st.session_state.forecast)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Model Evaluation
        st.subheader("Model Performance")
        merged = pd.merge(st.session_state.test, st.session_state.forecast, on='ds')
        
        if not merged.empty:
            # Calculate metrics
            mae = np.mean(np.abs(merged['y'] - merged['yhat']))
            rmse = np.sqrt(np.mean((merged['y'] - merged['yhat'])**2))
            mape = np.mean(np.abs((merged['y'] - merged['yhat']) / merged['y'])) * 100
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{mae:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            col3.metric("MAPE", f"{mape:.2f}%")
            
            # Actual vs Predicted Plot
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=merged['ds'], y=merged['y'], name='Actual'))
            fig3.add_trace(go.Scatter(x=merged['ds'], y=merged['yhat'], name='Predicted'))
            st.plotly_chart(fig3, use_container_width=True)
        
        # Download forecast results
        st.subheader("Download Forecasts")
        forecast_csv = st.session_state.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
        st.download_button("Download Forecast Data", forecast_csv, "prophet_forecast.csv")

else:
    st.info("Please upload your insurance dataset to begin analysis")

# Documentation Section
with st.expander("App Guide"):
    st.markdown("""
    **App Workflow:**
    1. Upload dataset (CSV/ZIP format)
    2. Select state for analysis
    3. Configure data preprocessing options
    4. Adjust model parameters
    5. Train model and view forecasts
    6. Download results
    
    **Dataset Requirements:**
    - Must contain columns: company_code, product, state, year_month, premiums
    - Historical data should cover multiple years for seasonality detection
    """)
