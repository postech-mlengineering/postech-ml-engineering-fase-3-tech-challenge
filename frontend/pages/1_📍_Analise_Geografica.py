import streamlit as st
import pydeck as pdk
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.data_loader import load_aggregated_data

st.set_page_config(page_title="Análise Geográfica", page_icon="📍", layout="wide")

st.title("📍 Análise Geográfica de Atrasos")
st.markdown("Explore as rotas e aeroportos com maiores índices de atraso.")

with st.spinner("Carregando dados..."):
    route_delays, airport_delays, _ = load_aggregated_data()

# Filtros no Sidebar
st.sidebar.header("Filtros")

# Filtro de Companhia Aérea
airlines = sorted(route_delays['AIRLINE_NAME'].dropna().unique().tolist())
selected_airline = st.sidebar.selectbox("Empresa (Companhia Aérea)", ["Todas"] + airlines)

# Filtro de Estado
states = sorted(airport_delays['ORIGIN_STATE'].dropna().unique().tolist())
selected_state = st.sidebar.selectbox("Estado (Origem)", ["Todos"] + states)

# Aplicar filtros
filtered_routes = route_delays.copy()
filtered_airports = airport_delays.copy()

if selected_airline != "Todas":
    filtered_routes = filtered_routes[filtered_routes['AIRLINE_NAME'] == selected_airline]

if selected_state != "Todos":
    filtered_routes = filtered_routes[filtered_routes['ORIGIN_STATE'] == selected_state]
    filtered_airports = filtered_airports[filtered_airports['ORIGIN_STATE'] == selected_state]

# --- 1. Mapa de Rotas (Arcos) ---
st.subheader("✈️ Rotas e Atrasos (Arcos)")

# Para evitar travar o navegador, limitamos as top 1000 piores rotas se houver muitos dados
if len(filtered_routes) > 2000:
    st.warning("Muitas rotas encontradas. Exibindo as 2000 rotas com maiores atrasos de chegada para manter a performance.")
    filtered_routes = filtered_routes.sort_values(by='AVG_ARRIVAL_DELAY', ascending=False).head(2000)

# Definir cor do arco com base no atraso (Vermelho para atraso, Verde para no horário)
def get_color(delay):
    if delay > 15:
        return [255, 0, 0, 150] # Red
    elif delay > 0:
        return [255, 165, 0, 150] # Orange
    else:
        return [0, 255, 0, 150] # Green

filtered_routes['COLOR'] = filtered_routes['AVG_ARRIVAL_DELAY'].apply(get_color)

arc_layer = pdk.Layer(
    "ArcLayer",
    data=filtered_routes,
    get_source_position=["ORIGIN_LONGITUDE", "ORIGIN_LATITUDE"],
    get_target_position=["DEST_LONGITUDE", "DEST_LATITUDE"],
    get_source_color="COLOR",
    get_target_color="COLOR",
    get_width="FLIGHT_COUNT / 100", # Largura baseada no volume de voos
    width_min_pixels=1,
    width_max_pixels=5,
    pickable=True,
    auto_highlight=True,
)

# Tooltip para o mapa de arcos
tooltip_arc = {
    "html": "<b>Origem:</b> {ORIGIN_CITY} ({ORIGIN_AIRPORT}) <br/>"
            "<b>Destino:</b> {DEST_CITY} ({DESTINATION_AIRPORT}) <br/>"
            "<b>Companhia:</b> {AIRLINE_NAME} <br/>"
            "<b>Atraso Médio Chegada:</b> {AVG_ARRIVAL_DELAY} min <br/>"
            "<b>Volume de Voos:</b> {FLIGHT_COUNT}",
    "style": {"backgroundColor": "steelblue", "color": "white"}
}

view_state_arc = pdk.ViewState(
    latitude=39.8283,
    longitude=-98.5795,
    zoom=3,
    pitch=45
)

st.pydeck_chart(pdk.Deck(
    layers=[arc_layer],
    initial_view_state=view_state_arc,
    tooltip=tooltip_arc,
    map_style=None
))

# --- 2. Heatmap de Atrasos nos Aeroportos ---
st.markdown("---")
st.subheader("🔥 Mapa de Calor: Atrasos na Origem")
st.markdown("Os pontos mais intensos representam aeroportos com maiores atrasos médios na partida.")

# Filtrar para exibir no heatmap apenas quem tem um atraso médio positivo para o peso ser válido
heatmap_data = filtered_airports[filtered_airports['AVG_DEPARTURE_DELAY'] > 0]

heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=heatmap_data,
    get_position=["ORIGIN_LONGITUDE", "ORIGIN_LATITUDE"],
    get_weight="AVG_DEPARTURE_DELAY", # Peso pelo tempo de atraso
    radiusPixels=50,
)

view_state_heat = pdk.ViewState(
    latitude=39.8283,
    longitude=-98.5795,
    zoom=3,
    pitch=0
)

st.pydeck_chart(pdk.Deck(
    layers=[heatmap_layer],
    initial_view_state=view_state_heat,
    map_style=None
))
