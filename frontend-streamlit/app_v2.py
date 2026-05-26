import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Configuração premium da página
st.set_page_config(
    page_title="Analytics & Preditor de Atrasos",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização e fontes modernas
st.markdown(    """
    <style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stMetric {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 15px 25px;
        border: 1px solid #334155;
    }
    div[data-testid="stMetricValue"] {
        color: #38BDF8;
        font-size: 2rem;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94A3B8;
        padding: 12px 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True)

# Cache de carregamento de dados e modelos
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    route_delays = pd.read_parquet(os.path.join(base_dir, 'data', 'route_delays.parquet'))
    airport_delays = pd.read_parquet(os.path.join(base_dir, 'data', 'airport_delays.parquet'))
    airline_stats = pd.read_parquet(os.path.join(base_dir, 'data', 'airline_stats.parquet'))
    return route_delays, airport_delays, airline_stats

@st.cache_resource
def load_ml_resources():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model = joblib.load(os.path.join(base_dir, 'models', 'model_xgboost.pkl'))
    scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
    return model, scaler

# Carregamento seguro
try:
    route_delays, airport_delays, airline_stats = load_data()
    model, scaler = load_ml_resources()
    resources_loaded = True
except Exception as e:
    st.error(f"Erro ao carregar recursos. Certifique-se de que os dados e modelos estejam gerados. Detalhes: {e}")
    resources_loaded = False

if resources_loaded:
    st.title("✈️ Analytics & Predição de Atrasos de Voos")
    st.markdown("---")

    # Definição das Abas
    tab1, tab2 = st.tabs(["📊 Análise Descritiva", "🔮 Preditor de Atraso (XGBoost)"])

    # ==================== ABA 1 ====================
    with tab1:
        st.subheader("📊 Estatísticas e Indicadores Históricos")
        st.markdown("Visão geral e análises descritivas interativas dos voos e seus atrasos nos EUA.")

        # Indicadores principais (KPI Cards)
        col1, col2, col3 = st.columns(3)
        
        # Média de atrasos
        total_flights = airport_delays['FLIGHT_COUNT'].sum()
        avg_delay = airport_delays['AVG_DEPARTURE_DELAY'].mean()
        worst_airline = airline_stats.loc[airline_stats['AIRLINE_DELAY'].idxmax()]['AIRLINE_NAME']

        with col1:
            st.metric(label="Total de Voos no Dataset", value=f"{total_flights:,.0f}".replace(",", "."))
        with col2:
            st.metric(label="Média de Atraso na Partida", value=f"{avg_delay:.2f} min")
        with col3:
            st.metric(label="Companhia com Maior Atraso Próprio", value=worst_airline)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gráficos da Aba 1 (Utilizando APENAS Plotly)
        layout_col1, layout_col2 = st.columns(2)

        with layout_col1:
            st.markdown("#### ✈️ Aeroportos com Maiores Atrasos Médios na Partida")
            top_airports = airport_delays.sort_values(by='AVG_DEPARTURE_DELAY', ascending=False).head(10)
            fig_airports = px.bar(
                top_airports,
                x='AVG_DEPARTURE_DELAY',
                y='ORIGIN_AIRPORT',
                orientation='h',
                labels={'AVG_DEPARTURE_DELAY': 'Atraso Médio (min)', 'ORIGIN_AIRPORT': 'Aeroporto'},
                color='AVG_DEPARTURE_DELAY',
                color_continuous_scale='Reds'
            )
            fig_airports.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#F8FAFC",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_airports, use_container_width=True)

        with layout_col2:
            st.markdown("#### 🏢 Desempenho por Companhia Aérea (Atraso da Companhia)")
            top_airlines = airline_stats.sort_values(by='AIRLINE_DELAY', ascending=False)
            fig_airlines = px.bar(
                top_airlines,
                x='AIRLINE_NAME',
                y='AIRLINE_DELAY',
                labels={'AIRLINE_DELAY': 'Minutos Acumulados', 'AIRLINE_NAME': 'Companhia Aérea'},
                color='AIRLINE_DELAY',
                color_continuous_scale='Blues'
            )
            fig_airlines.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="#F8FAFC",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_airlines, use_container_width=True)

        st.markdown("#### 🗺️ Rotas (Origem ➔ Destino) com Maiores Atrasos")
        top_routes = route_delays.sort_values(by='AVG_ARRIVAL_DELAY', ascending=False).head(15)
        top_routes['ROUTE'] = top_routes['ORIGIN_AIRPORT'] + " ➔ " + top_routes['DESTINATION_AIRPORT']
        fig_routes = px.bar(
            top_routes,
            x='ROUTE',
            y='AVG_ARRIVAL_DELAY',
            labels={'AVG_ARRIVAL_DELAY': 'Atraso Médio na Chegada (min)', 'ROUTE': 'Rota'},
            color='AVG_ARRIVAL_DELAY',
            color_continuous_scale='Oranges'
        )
        fig_routes.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#F8FAFC",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_routes, use_container_width=True)

    # ==================== ABA 2 ====================
    with tab2:
        st.subheader("🔮 Estimativa de Atraso via XGBoost")
        st.markdown("Insira as variáveis exatas de ambiente operacional para calcular a probabilidade de atraso do voo.")

        with st.form("form_prediction"):
            col_a, col_b = st.columns(2)

            with col_a:
                airplane_was_delayed_input = st.selectbox(
                    "Aeronave veio de voo com atraso anterior? (sem tempo de recuperação)",
                    options=[0, 1],
                    format_func=lambda x: "Sim" if x == 1 else "Não"
                )
                
                origin_airport_delay_momentum_input = st.slider(
                    "Momentum de atraso no aeroporto de ORIGEM (última 1h)",
                    min_value=0.0, max_value=1.0, value=0.15, step=0.01,
                    help="Proporção de voos atrasados na última hora neste aeroporto."
                )

                dest_airport_delay_momentum_input = st.slider(
                    "Momentum de atraso no aeroporto de DESTINO (última 1h)",
                    min_value=0.0, max_value=1.0, value=0.12, step=0.01,
                    help="Proporção de voos atrasados na última hora no aeroporto de destino."
                )

                flights_same_window_input = st.number_input(
                    "Quantidade de voos simultâneos no mesmo aeroporto (mesma janela de 1h)",
                    min_value=0, max_value=200, value=15, step=1
                )

                haul_type_input = st.selectbox(
                    "Tipo de Rota (Distância)",
                    options=["SHORT", "MEDIUM", "LONG"]
                )

            with col_b:
                month_input = st.slider("Mês do Voo", min_value=1, max_value=12, value=6)
                
                hour_input = st.slider("Hora Programada de Partida", min_value=0, max_value=23, value=14)

                is_weekend_input = st.selectbox(
                    "É Fim de Semana?",
                    options=[0, 1],
                    format_func=lambda x: "Sim" if x == 1 else "Não"
                )

                is_holiday_input = st.selectbox(
                    "É Feriado nos EUA?",
                    options=[0, 1],
                    format_func=lambda x: "Sim" if x == 1 else "Não"
                )

            # Botão de submissão do formulário
            btn_predict = st.form_submit_button("Estimar Probabilidade de Atraso", type="primary")

            if btn_predict:
                # 1. Mapeamento do Haul Type para variáveis Dummy
                h_short = 1 if haul_type_input == "SHORT" else 0
                h_medium = 1 if haul_type_input == "MEDIUM" else 0
                h_long = 1 if haul_type_input == "LONG" else 0

                # 2. Escalonamento das variáveis numéricas
                # As 8 variáveis numéricas na ordem correta do scaler.pkl:
                numerical_features = pd.DataFrame([[
                    airplane_was_delayed_input,
                    origin_airport_delay_momentum_input,
                    dest_airport_delay_momentum_input,
                    flights_same_window_input,
                    month_input,
                    hour_input,
                    is_weekend_input,
                    is_holiday_input
                ]], columns=[
                    'AIRPLANE_WAS_DELAYED', 'ORIGIN_AIRPORT_DELAY_MOMENTUM', 'DESTINATION_AIRPORT_DELAY_MOMENTUM',
                    'ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW', 'MONTH', 'SCHEDULED_DEPARTURE_HOUR', 
                    'IS_WEEKEND', 'IS_HOLIDAY'
                ])

                scaled_numerical = scaler.transform(numerical_features)[0]

                # 3. Montar vetor final de features para predição (11 no total)
                final_features = np.array([[
                    scaled_numerical[0], # AIRPLANE_WAS_DELAYED
                    scaled_numerical[1], # ORIGIN_AIRPORT_DELAY_MOMENTUM
                    scaled_numerical[2], # DESTINATION_AIRPORT_DELAY_MOMENTUM
                    scaled_numerical[3], # ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW
                    scaled_numerical[4], # MONTH
                    scaled_numerical[5], # SCHEDULED_DEPARTURE_HOUR
                    scaled_numerical[6], # IS_WEEKEND
                    scaled_numerical[7], # IS_HOLIDAY
                    h_short,             # HAUL_TYPE_SHORT
                    h_medium,            # HAUL_TYPE_MEDIUM
                    h_long               # HAUL_TYPE_LONG
                ]])

                # DataFrame com nomes de colunas exatos exigidos pelo XGBClassifier
                feature_names = [
                    'AIRPLANE_WAS_DELAYED', 
                    'ORIGIN_AIRPORT_DELAY_MOMENTUM',
                    'DESTINATION_AIRPORT_DELAY_MOMENTUM',
                    'ORIGIN_AIRPORT_FLIGHTS_ON_THE_SAME_WINDOW', 
                    'MONTH',
                    'SCHEDULED_DEPARTURE_HOUR', 
                    'IS_WEEKEND', 
                    'IS_HOLIDAY', 
                    'HAUL_TYPE_SHORT',
                    'HAUL_TYPE_MEDIUM', 
                    'HAUL_TYPE_LONG'
                ]
                df_predict = pd.DataFrame(final_features, columns=feature_names)

                # Realizar predição
                prediction = model.predict(df_predict)[0]
                probabilities = model.predict_proba(df_predict)[0]

                st.markdown("---")
                st.subheader("🎯 Resultado da Estimativa")

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    if prediction == 1:
                        st.error(f"⚠️ **Atraso Provável!**")
                        st.write(f"O modelo XGBoost estimou que as condições operacionais informadas favorecem um atraso no voo.")
                    else:
                        st.success(f"✅ **Voo no Horário!**")
                        st.write("Excelente! Condições operacionais indicam que o voo tem altíssima chance de partir sem atrasos significativos.")

                with col_res2:
                    # Gauge gráfico elegante da probabilidade de atraso
                    fig_prob = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probabilities[1] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probabilidade de Atraso (%)", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                            'bar': {'color': "#EF4444" if prediction == 1 else "#10B981"},
                            'bgcolor': "#1E293B",
                            'borderwidth': 2,
                            'bordercolor': "#334155",
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.1)'},
                                {'range': [50, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                            ]
                        }
                    ))
                    fig_prob.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color="#F8FAFC",
                        height=250,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
