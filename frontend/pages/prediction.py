import streamlit as st
import pandas as pd
import numpy as np

from services.data_service import load_resources
from charts.plots import plot_prediction_gauge


def prediction_page():
    st.title('Preditor')

    st.divider()

    try:
        model, scaler = load_resources()
    except:
        st.error('Recursos não encontrados.')
        return

    st.subheader('Configurações')

    with st.form('ml_form'):
        col1, col2 = st.columns(2)
        with col1:
            airplane_delayed = st.selectbox('Aeronave veio de atraso?', [0, 1])
            m_origin = st.slider('Momentum Origem', 0.0, 1.0, 0.1)
            m_dest = st.slider('Momentum Destino', 0.0, 1.0, 0.1)
            flights_window = st.number_input('Voos na mesma hora', 0, 150, 20)
        
        with col2:
            month = st.number_input('Mês', 1, 12, 6)
            hour = st.number_input('Hora (HH)', 0, 23, 14)
            weekend = st.selectbox('Fim de Semana?', [0, 1])
            holiday = st.selectbox('Feriado?', [0, 1])
            haul = st.selectbox('Tipo de Rota', ['SHORT', 'MEDIUM', 'LONG'])

        btn = st.form_submit_button('Prever Atraso', type='primary')

    if btn:
        num_data = pd.DataFrame([[
            airplane_delayed, m_origin, m_dest, flights_window, 
            month, hour, weekend, holiday
        ]], columns=['AIRPLANE_WAS_DELAYED', 'ORIGIN_AIRPORT_DELAY_MOMENTUM', 
                     'DESTINATION_AIRPORT_DELAY_MOMENTUM', 'ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW', 
                     'MONTH', 'SCHEDULED_DEPARTURE_HOUR', 'IS_WEEKEND', 'IS_HOLIDAY'])
        
        num_scaled = scaler.transform(num_data)

        h_short = 1 if haul == 'SHORT' else 0
        h_medium = 1 if haul == 'MEDIUM' else 0
        h_long = 1 if haul == 'LONG' else 0

        final_input = np.hstack([num_scaled, [[h_short, h_medium, h_long]]])
        
        prob = model.predict_proba(final_input)[0][1]

        plot_prediction_gauge(prob)

        st.info(f'O voo tem {(prob * 100):.2f} de probabilidade atraso.')

        st.plotly_chart(plot_prediction_gauge(prob))