import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_aggregated_data, load_models

st.set_page_config(page_title="Modelo Supervisionado", page_icon="🤖", layout="wide")

st.title("🤖 Predição de Atrasos (Modelo Supervisionado)")
st.markdown("Preveja se um voo hipotético sofrerá atrasos com base nos dados históricos.")

try:
    with st.spinner("Carregando modelos e dados..."):
        model, scaler, le_airline = load_models()
        route_delays, airport_delays, airline_stats = load_aggregated_data()
        
    st.sidebar.header("Parâmetros do Voo")
    
    # Inputs para a predição
    month = st.sidebar.slider("Mês do voo", min_value=1, max_value=12, value=1)
    day_of_week = st.sidebar.selectbox("Dia da Semana", options=[1, 2, 3, 4, 5, 6, 7], format_func=lambda x: ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo'][x-1])
    
    # Hora de partida (formato HHMM -> 0 a 2359)
    hour = st.sidebar.slider("Hora de Partida", min_value=0, max_value=23, value=12)
    minute = st.sidebar.slider("Minuto de Partida", min_value=0, max_value=59, value=30, step=5)
    scheduled_departure = hour * 100 + minute
    
    distance = st.sidebar.number_input("Distância Estimada (Milhas)", min_value=50, max_value=5000, value=1000, step=100)
    
    available_airlines = list(le_airline.classes_)
    airline_name = st.sidebar.selectbox("Companhia Aérea", available_airlines)
    
    if st.sidebar.button("Fazer Predição", type="primary"):
        # Preparar dados para o modelo
        input_data = pd.DataFrame({
            'MONTH': [month],
            'DAY_OF_WEEK': [day_of_week],
            'SCHEDULED_DEPARTURE': [scheduled_departure],
            'DISTANCE': [distance],
            'AIRLINE_NAME': [le_airline.transform([airline_name])[0]]
        })
        
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        st.subheader("Resultado da Predição")
        if prediction == 1:
            st.error(f"⚠️ Alta probabilidade de atraso! (Confiança: {probability[1]*100:.2f}%)")
        else:
            st.success(f"✅ Voo deve ocorrer no horário! (Confiança: {probability[0]*100:.2f}%)")
            
        st.markdown("---")
        
    # Mostrar feature importance simulado (já que LogisticRegression usa coeficientes)
    st.subheader("Importância das Variáveis (Feature Importance)")
    st.markdown("Descubra o quanto cada variável contribui para a ocorrência de atrasos.")
    
    coefs = pd.DataFrame({
        'Feature': ['Mês', 'Dia da Semana', 'Horário de Partida', 'Distância', 'Companhia Aérea'],
        'Coeficiente': model.coef_[0]
    })
    
    # Ordenar por valor absoluto para ver o impacto
    coefs['Impacto Absoluto'] = coefs['Coeficiente'].abs()
    coefs = coefs.sort_values(by='Impacto Absoluto', ascending=False)
    
    fig = px.bar(coefs, x='Coeficiente', y='Feature', orientation='h', 
                 color='Coeficiente', color_continuous_scale='RdBu_r',
                 title="Impacto Positivo = Aumenta Atraso | Impacto Negativo = Reduz Atraso")
                 
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Os modelos ainda não foram gerados. Por favor, execute o script `aggregate_data.py`. Erro: {e}")
