import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Tractor Sales Forecaster", layout="centered")

st.title("🚜 Tractor Sales Forecasting Dashboard")
st.write("""
This application uses a pre-trained **SARIMAX model** to predict monthly tractor sales.
Select the forecasting horizon below to see predicted values and an interactive chart.
""")

# Load the dumped model
@st.cache_resource
def load_model():
    return joblib.load('tractor_sales_arima_model.pkl')

try:
    model = load_model()
    st.success("Model successfully loaded!")
except Exception as e:
    st.error(f"Failed to load the model file. Please ensure 'tractor_sales_arima_model.pkl' is in the same directory. Error: {e}")
    st.stop()

# User Inputs
st.sidebar.header("Forecast Settings")
forecast_years = st.sidebar.slider("Years to Forecast", min_value=1, max_value=5, value=1)
n_periods = forecast_years * 12

if st.button("Generate Forecast"):
    with st.spinner("Calculating predictions..."):
        # Training data ended in Dec 2012. 
        # Model forecasts from 2013-01-01 onwards. 
        forecast_indices = pd.date_range(start='2013-01-01', periods=n_periods, freq='MS')
        
        # Generate forecast
        predictions = model.predict(n_periods=n_periods)
        forecast_df = pd.DataFrame({
            'Month-Year': forecast_indices,
            'Predicted Sales': predictions
        })
        
        # Focus on the most recent predictions or full forecast window
        forecast_df['Formatted Month'] = forecast_df['Month-Year'].dt.strftime('%B %Y')
        
        st.subheader(f"Sales Forecast for the next {forecast_years} Year(s)")
        
        # Plot interactive forecast line
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['Month-Year'], 
            y=forecast_df['Predicted Sales'],
            mode='lines+markers',
            name='Forecasted Sales',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Future Forecast Trend",
            xaxis_title="Date",
            yaxis_title="Tractors Sold",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display forecast table
        st.write("### Prediction Table")
        display_df = forecast_df[['Formatted Month', 'Predicted Sales']].copy()
        display_df['Predicted Sales'] = display_df['Predicted Sales'].round(2)
        st.dataframe(display_df, use_container_width=True)
