import joblib

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_pickle("../data/curated/data.pkl")
    df_features = pd.read_pickle("../data/curated/features.pkl")
    return df, df_features


@st.cache_resource(show_spinner=False)
def load_resources():
    model = joblib.load("../models/model_xgboost.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    return model, scaler


def get_best_k(scaled_data, max_k=10):
    k_range = range(1, max_k + 1)
    sse = []

    for i in k_range:
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)
    
    kl = KneeLocator(k_range, sse, curve='convex', direction='decreasing')
    return kl.elbow


def get_cluster(df_input, group_col, target_col='IS_DELAYED'):
    """Executa o processo de clustering conforme sua lógica fornecida"""
    df_profile = df_input.groupby(group_col).agg({
        target_col: ['mean', 'count']
    }).reset_index()
    
    df_profile.columns = [group_col, f'{group_col}_DELAY_RATE', 'FLIGHT_VOLUME']
    
    scaler = StandardScaler()
    features_to_scale = [f'{group_col}_DELAY_RATE', 'FLIGHT_VOLUME']
    df_scaled = scaler.fit_transform(df_profile[features_to_scale])
    
    n_clusters = get_best_k(df_scaled)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_profile[f'{group_col}_PROFILE'] = kmeans.fit_predict(df_scaled)
    
    return df_profile

