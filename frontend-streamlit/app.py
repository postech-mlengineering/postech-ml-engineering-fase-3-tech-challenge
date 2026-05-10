import streamlit as st
import sys
import os

# Configuração da página principal
st.set_page_config(
    page_title="Dashboard de Atrasos - Tech Challenge",
    page_icon="✈️",
    layout="wide"
)

# Adicionando path para imports locais
sys.path.append(os.path.dirname(__file__))

from utils.data_loader import load_aggregated_data

st.title("✈️ Tech Challenge Fase 3: Atrasos de Voos")
st.markdown("---")

st.markdown("""
### Bem-vindo ao Dashboard de Análise de Voos!
Este painel foi desenvolvido para a Fase 3 do Tech Challenge, focando na análise do grande volume de dados de voos dos EUA e seus respectivos atrasos.

Aqui você poderá explorar:
- **📍 Análise Geográfica**: Visualize rotas com maiores atrasos e descubra aeroportos críticos através de um mapa de calor.
- **🤖 Modelo Supervisionado**: Faça predições em tempo real para saber se um voo hipotético vai atrasar ou não.
- **🔍 Modelo Não Supervisionado**: Explore os perfis operacionais das companhias aéreas utilizando clusterização (K-Means).

Utilize o menu lateral para navegar entre as páginas.
""")

st.markdown("---")
st.subheader("📊 Visão Geral dos Dados")

try:
    with st.spinner("Carregando dados agregados..."):
        route_delays, airport_delays, airline_stats = load_aggregated_data()
    
    total_flights = airport_delays['FLIGHT_COUNT'].sum()
    avg_delay = airport_delays['AVG_DEPARTURE_DELAY'].mean()
    worst_airline = airline_stats.loc[airline_stats['AIRLINE_DELAY'].idxmax()]['AIRLINE_NAME']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Voos Analisados", f"{total_flights:,.0f}".replace(',', '.'))
    
    with col2:
        st.metric("Média de Atraso na Partida", f"{avg_delay:.2f} min")
        
    with col3:
        st.metric("Companhia c/ Maior Atraso (Causa Própria)", worst_airline)
        
except Exception as e:
    st.error(f"Erro ao carregar os dados. Certifique-se de que o script `aggregate_data.py` foi executado. Detalhes: {e}")
