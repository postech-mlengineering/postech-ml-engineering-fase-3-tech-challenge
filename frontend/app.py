import streamlit as st

from pages.about import about_page
from pages.analytics import analytics_page
from pages.prediction import prediction_page
from pages.model_performance import model_performance_page


st.set_page_config(
    page_title="Flight Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
    <style>
    .material-symbols-outlined {
        font-size: 24px;
        vertical-align: middle;
        margin-right: 10px;
    }
    /* Estilização Premium */
    [data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        padding: 15px;
        border-radius: 10px;
    }
    .stApp {
        background-color: #0F172A;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Definição das Páginas
def show_intro():
    st.title("✈️ Flight Analytics System")
    st.markdown("""
    ### Bem-vindo ao sistema de controle de atrasos.
    Utilize o menu lateral para navegar:
    - **Analytics:** Visualize tendências históricas.
    - **Preditor:** Estime riscos de atrasos em tempo real.
    """)

# 4. Configuração do st.navigation
pages = {
    "Info": [
        st.Page(about_page, title="Sobre", icon=":material/info:"),
    ],
    "Dashboard": [
        st.Page(analytics_page, title="Estatísticas", icon=":material/analytics:"),
        st.Page(model_performance_page, title="Performance", icon=":material/rocket:")
    ],
    "Operacional": [
        st.Page(prediction_page, title="Predição de Atraso", icon=":material/online_prediction:"),
    ]
}

pg = st.navigation(pages)
pg.run()