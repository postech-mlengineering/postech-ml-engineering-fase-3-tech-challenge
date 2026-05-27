import streamlit as st

from services.data_service import load_data, get_cluster
from charts.plots import (
    plot_scatter, 
    plot_pie, 
    plot_columns, 
    plot_columns_lines
)


def analytics_page():
    st.title('Estatísticas')
    
    df, _ = load_data()

    if df is None or df.empty:
        st.warning('Nenhum dado encontrado. Verifique a fonte de dados.')
        return

    st.divider()

    tab1, tab2 = st.tabs(['Estatísticas', 'Perfis'])

    with tab1:
        st.subheader('Estatísticas Descritivas')
        
        stat_view = st.selectbox(
            'Selecione:',
            [
                'Análise de Atrasos', 
                'Análise de Atrasos por Causa',
                'Análise Temporal de Atrasos'
            ]
        )

        if stat_view == 'Distribuição de Pontualidade':
            fig = plot_pie(df)
            st.plotly_chart(fig, width='stretch')
            
        elif stat_view == 'Análise Temporal de Atrasos':
            fig = plot_columns_lines(df)
            st.plotly_chart(fig, width='stretch')
            
        elif stat_view == 'Análise de Atrasos':
            delay_cols = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY']
            delay_labels = ['Atraso Partida', 'Atraso Chegada']
            fig = plot_columns(df, delay_cols, delay_labels, 'Distribuição e Volatilidade dos Atrasos')
            st.plotly_chart(fig, width='stretch')
            
        elif stat_view == 'Análise de Atrasos por Causa':
            cause_cols = ['AIR_SYSTEM_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'SECURITY_DELAY', 'WEATHER_DELAY']
            cause_labels = ['Sist. Aéreo', 'Cia Aérea', 'Aeronave Tardia', 'Segurança', 'Clima']
            fig = plot_columns(df, cause_cols, cause_labels, 'Distribuição e Volatilidade dos Atrasos por Causa')
            st.plotly_chart(fig, width='stretch')

    with tab2:
        st.subheader('Perfis de Aeroportos, Rotas e Companhias')
        
        cluster_option = st.selectbox(
            'Selecione o perfil:',
            ['Aeroporto de Origem', 'Rota (Origem-Destino)', 'Companhia Aérea']
        )

        if cluster_option == 'Aeroporto de Origem':
            n_items = df['ORIGIN_AIRPORT'].nunique()
            group_key = 'ORIGIN_AIRPORT'
        elif cluster_option == 'Rota (Origem-Destino)':
            df['ROUTE'] = df['ORIGIN_AIRPORT'] + '_' + df['DESTINATION_AIRPORT']
            n_items = df['ROUTE'].nunique()
            group_key = 'ROUTE'
        else:
            n_items = df['AIRLINE'].nunique()
            group_key = 'AIRLINE'

        if n_items < 5:
            st.error(f'Dados insuficientes ({n_items}) para gerar clusters significativos.')
        else:
            with st.spinner('Processando clusters...'):
                try:
                    df_res = get_cluster(df, group_key)
                    
                    rate_col = f'{group_key}_DELAY_RATE'
                    profile_col = f'{group_key}_PROFILE'
                    
                    fig = plot_scatter(df_res, 'FLIGHT_VOLUME', rate_col, profile_col, group_key)
                    st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f'Erro ao processar clusters: {e}')