import streamlit as st


def about_page():
    st.title('Sobre')
    st.divider()

    st.markdown('''
    Este projeto implementa um pipeline end-to-end de **Machine Learning Engineering** para prever atrasos em voos domésticos nos EUA. 
    A solução aborda desde o processamento brutos até a disponibilização de um modelo preditivo otimizado.
    ''')

    st.header('Limpeza e Consolidação dos Dados (`DataCleaner`)', divider='blue')
    st.write('''
    A etapa inicial consolida dados de voos, aeroportos e companhias aéreas para criar um contexto operacional completo.
    ''')
    
    with st.expander('Detalhes', expanded=False):
        st.markdown('''
        *   **Tratamento de Nulos:** Colunas de categoria de atraso (`AIR_SYSTEM_DELAY`, `WEATHER_DELAY`, etc.) preenchidas com **0**.
        *   **Filtragem:** Remoção de voos cancelados (`CANCELLED`) ou desviados (`DIVERTED`), mantendo apenas a performance real de voo.
        *   **Definição do Target:** A variável alvo `IS_DELAYED` é binária, considerando **1** para atrasos de chegada superiores a **15 minutos**.
        ''')

    st.header('Engenharia de Features (`FeatureEngineer`)', divider='blue')
    st.write('A inteligência do modelo baseia-se em estatísticas derivadas do conjunto de dados.')

    col1, col2 = st.columns(2)
    with col1:
        st.info('**Dinâmica Operacional**')
        st.write('''
        - **Rolling:** Médias móveis de 1 hora para capturar o status de atraso atual dos aeroportos de origem e destino.
        - **Simultaneidade:** Contagem de decolagens na mesma janela horária no aeroporto de origem.
        - **Desempenho da Aeronave:** Verifica se o voo anterior da aeromave atrasou em uma janela menor que 6h.
        ''')
            
    with col2:
        st.info('**Perfis**')
        st.write('''
        - **Clustering:** Perfis de risco para aeroportos e companhias aéreas baseados em volume vs. taxa de atraso histórica.
        ''')

    with st.expander('Outras', expanded=False):
        st.markdown('''
        *   **Tempo:** Integração com a biblioteca `holidays` para detectar feriados nos EUA, além de sazonalidade e finais de semana.
        *   **Distância:** Categorização automática da rota (Curta, Média ou Longa) baseada em quantis de distância.
        ''')

    with st.expander('Pré-processamento', expanded=False):
        st.markdown('''
        *   **Balanceamento:** Aplicação de **undersampling** (1:1) para evitar viés em favor de voos pontuais.
        *   **Normalização:** Uso do `StandardScaler` persistido para garantir que variáveis de escala diferente (ex: mês vs. volume de voos) tenham peso equilibrado.
        ''')

    st.header('Modelagem e Performance (`ModelTrainer`)', divider='blue')
    
    st.success('**Modelo: XGBoost Classifier**')
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric('Acurácia', '74.84%')
    col_m2.metric('F1-Score', '0.7460')
    col_m3.metric('Precisão', '0.7581')
    col_m4.metric('Recall', '0.7484')

    with st.expander('Pipeline de Treinamento'):
        st.markdown('''
        O modelo foi selecionado através de uma competição rigorosa:
        1.  **Model Selection:** Comparação entre **XGBoost**, **Random Forest** e **Gradient Boosting** utilizando `RandomizedSearchCV` com validação cruzada (CV=3).
        2.  **Fine-Tuning:** Otimização de hiperparâmetros (como `max_depth`, `learning_rate` e `gamma`) no melhor modelo utilizando CV=5.
        ''')

    st.divider()

    col_tech, col_team = st.columns([2, 1])
    with col_tech:
        st.markdown('**Tecnologias:**')
        st.code('Python 3.11 | XGBoost | Scikit-Learn | Pandas | KMeans | Joblib | Streamlit', language='text')
    
    with col_team:
        st.markdown('**Colaboradores:**')
        st.markdown('''
        - Hugo Rodrigues
        - Leandro Delis
        - Jorge Platero
        ''')

    st.caption('Pós-Graduação em Machine Learning Engineering - FIAP | Fase 3 - Tech Challenge')