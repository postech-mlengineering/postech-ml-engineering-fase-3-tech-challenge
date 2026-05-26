import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from services.data_service import load_resources


def prediction_page():
    st.title("🔮 Preditor")

    st.divider()

    try:
        model, scaler = load_resources()
    except:
        st.error("Recursos (Model/Scaler) não encontrados em /models")
        return

    st.subheader("Configurações")

    with st.form("ml_form"):
        col1, col2 = st.columns(2)
        with col1:
            airplane_delayed = st.selectbox("Aeronave veio de atraso?", [0, 1])
            m_origin = st.slider("Momentum Origem", 0.0, 1.0, 0.1)
            m_dest = st.slider("Momentum Destino", 0.0, 1.0, 0.1)
            flights_window = st.number_input("Voos na mesma hora", 0, 150, 20)
        
        with col2:
            month = st.number_input("Mês", 1, 12, 6)
            hour = st.number_input("Hora (HH)", 0, 23, 14)
            weekend = st.selectbox("Fim de Semana?", [0, 1])
            holiday = st.selectbox("Feriado?", [0, 1])
            haul = st.selectbox("Tipo de Rota", ["SHORT", "MEDIUM", "LONG"])

        btn = st.form_submit_button("Prever Atraso", type="primary")

    if btn:
        # Preparação dos dados seguindo seu FeatureEngineer
        # 1. Numéricos
        num_data = pd.DataFrame([[
            airplane_delayed, m_origin, m_dest, flights_window, 
            month, hour, weekend, holiday
        ]], columns=['AIRPLANE_WAS_DELAYED', 'ORIGIN_AIRPORT_DELAY_MOMENTUM', 
                     'DESTINATION_AIRPORT_DELAY_MOMENTUM', 'ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW', 
                     'MONTH', 'SCHEDULED_DEPARTURE_HOUR', 'IS_WEEKEND', 'IS_HOLIDAY'])
        
        num_scaled = scaler.transform(num_data)

        # 2. Dummies para HAUL_TYPE
        h_short = 1 if haul == "SHORT" else 0
        h_medium = 1 if haul == "MEDIUM" else 0
        h_long = 1 if haul == "LONG" else 0

        # 3. Concatenar (11 colunas total: 8 scaled + 3 dummies)
        final_input = np.hstack([num_scaled, [[h_short, h_medium, h_long]]])
        
        # 4. Predição
        prob = model.predict_proba(final_input)[0][1]

        # Gauge Chart
        st.info(f"O voo tem {(prob * 100):.2f} de probabilidade atraso.")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            title = {'text': "Probabilidade de Atraso (%)"},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': "#EF4444" if prob > 0.5 else "#10B981"}}
        ))
        st.plotly_chart(fig)