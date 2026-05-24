import streamlit as st
import pandas as pd
import sys
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_aggregated_data

st.set_page_config(page_title="Modelo Não Supervisionado", page_icon="🔍", layout="wide")

st.title("🔍 Clusterização de Companhias Aéreas")
st.markdown("Identifique perfis operacionais das empresas aéreas agrupando-as por características de atraso e tempo de manobra.")

try:
    with st.spinner("Carregando dados..."):
        _, _, airline_stats = load_aggregated_data()
        
    st.sidebar.header("Parâmetros do Modelo")
    
    # Seleção do número de clusters (K)
    k_clusters = st.sidebar.slider("Número de Clusters (K)", min_value=2, max_value=8, value=4)
    
    # Seleção de features para clusterizar
    available_features = ['TAXI_OUT', 'TAXI_IN', 'AIRLINE_DELAY', 'WEATHER_DELAY', 'FLIGHT_VOLUME']
    selected_features = st.sidebar.multiselect(
        "Features para Clusterização", 
        options=available_features, 
        default=['TAXI_OUT', 'TAXI_IN', 'AIRLINE_DELAY', 'WEATHER_DELAY']
    )
    
    if len(selected_features) < 2:
        st.warning("Selecione pelo menos 2 features para realizar a clusterização.")
    else:
        # Treinar K-Means em tempo real
        X = airline_stats[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)
        airline_stats['CLUSTER'] = kmeans.fit_predict(X_scaled)
        airline_stats['CLUSTER_NAME'] = airline_stats['CLUSTER'].apply(lambda x: f"Perfil {x}")
        
        # Plotly 3D ou 2D
        st.subheader("Visualização dos Clusters")
        
        if len(selected_features) >= 3:
            fig = px.scatter_3d(
                airline_stats, 
                x=selected_features[0], 
                y=selected_features[1], 
                z=selected_features[2],
                color='CLUSTER_NAME',
                hover_name='AIRLINE_NAME',
                size='FLIGHT_VOLUME' if 'FLIGHT_VOLUME' not in selected_features else None,
                title=f"Agrupamento 3D de Companhias Aéreas (K={k_clusters})"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(
                airline_stats,
                x=selected_features[0],
                y=selected_features[1],
                color='CLUSTER_NAME',
                hover_name='AIRLINE_NAME',
                size='FLIGHT_VOLUME' if 'FLIGHT_VOLUME' not in selected_features else None,
                title=f"Agrupamento 2D de Companhias Aéreas (K={k_clusters})"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Explicando os Perfis
        st.markdown("---")
        st.subheader("Entendendo os Perfis (Médias por Cluster)")
        
        # Calcular médias reais por cluster para explicar
        cluster_means = airline_stats.groupby('CLUSTER_NAME')[selected_features].mean().reset_index()
        
        # Mostrar em cards ou dataframe
        st.dataframe(cluster_means.style.highlight_max(axis=0, color='lightcoral').highlight_min(axis=0, color='lightgreen'))
        st.caption("Verde = Menor Média (Geralmente Melhor) | Vermelho = Maior Média (Geralmente Pior)")
        
        # Tabela de companhias por cluster
        st.markdown("### Companhias por Perfil")
        for cluster in sorted(airline_stats['CLUSTER_NAME'].unique()):
            companies = airline_stats[airline_stats['CLUSTER_NAME'] == cluster]['AIRLINE_NAME'].tolist()
            st.write(f"**{cluster}:** {', '.join(companies)}")

except Exception as e:
    st.error(f"Erro ao carregar dados. Executou o `aggregate_data.py`? Erro: {e}")
