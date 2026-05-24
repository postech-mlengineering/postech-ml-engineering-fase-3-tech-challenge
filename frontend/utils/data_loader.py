import streamlit as st
import pandas as pd
import os
import joblib

@st.cache_data
def load_aggregated_data():
    """Carrega os dados pré-agregados em cache para performance."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    
    route_delays = pd.read_parquet(os.path.join(data_dir, 'route_delays.parquet'))
    airport_delays = pd.read_parquet(os.path.join(data_dir, 'airport_delays.parquet'))
    airline_stats = pd.read_parquet(os.path.join(data_dir, 'airline_stats.parquet'))
    
    return route_delays, airport_delays, airline_stats

@st.cache_resource
def load_models():
    """Carrega o modelo treinado e os encoders em cache."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    
    model = joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    le_airline = joblib.load(os.path.join(models_dir, 'le_airline.pkl'))
    
    return model, scaler, le_airline
