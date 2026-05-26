import streamlit as st

def about_page():
    st.title("✈️ Sistema de Análise e Predição de Atrasos Aéreos")
    
    st.markdown("""
    ### 1. Visão Geral do Projeto
    Este sistema foi desenvolvido como solução para o **Tech Challenge - Fase 3**, utilizando a base de dados de transporte aéreo dos EUA. O objetivo principal é identificar padrões de atraso e construir um modelo preditivo capaz de classificar a probabilidade de um voo sofrer atraso superior a 15 minutos (critério padrão da FAA).

    ### 2. Metodologia e Pipeline de Dados
    O projeto foi estruturado em um pipeline modular para garantir escalabilidade e reprodutibilidade:

    #### **A. Limpeza e Consolidação (Data Cleaning)**
    - **Merge Relacional:** Consolidação de três fontes de dados (`flights`, `airlines` e `airports`).
    - **Tratamento de Outliers e Nulos:** Imputação de valor zero para colunas de causas de atraso e remoção de voos cancelados ou desviados para focar estritamente na performance operacional.
    - **Definição de Target:** Criação da variável binária `IS_DELAYED` baseada no campo `ARRIVAL_DELAY`.

    #### **B. Engenharia de Features (Feature Engineering)**
    Esta é a camada mais crítica do projeto, onde foram aplicadas estratégias de:
    - **Time-Series Momentum:** Cálculo de "atraso em cadeia" através do rastreamento do prefixo da aeronave (`TAIL_NUMBER`) e da inércia de atrasos no aeroporto de origem nas últimas 1h.
    - **Análise de Contexto:** Integração com feriados nacionais dos EUA e definição de estações do ano (Sazonalidade).
    - **Densidade Operacional:** Cálculo de voos simultâneos na mesma janela horária para medir saturação da infraestrutura.

    #### **C. Modelagem Não Supervisionada (Clustering)**
    - **Estratégia:** Aplicação do algoritmo **K-Means** para agrupar aeroportos e companhias aéreas.
    - **Critério:** Os grupos foram definidos pela relação entre **Volume de Voos vs. Taxa de Atraso**, permitindo que o modelo supervisionado entenda o "perfil de risco" da entidade sem a necessidade de centenas de colunas de One-Hot Encoding.

    #### **D. Modelagem Supervisionada (Classification)**
    - **Algoritmos:** Comparação sistemática entre **XGBoost, Random Forest e Gradient Boosting**.
    - **Tratamento de Dados:** Aplicação de *Random Undersampling* para balancear as classes de atraso e *StandardScaler* para normalização das variáveis contínuas.
    - **Otimização:** Uso de *RandomizedSearchCV* para ajuste de hiperparâmetros, focando na métrica **F1-Score** para garantir equilíbrio entre Precisão e Recall.

    ### 3. Resultados e Conclusões
    - **Performance:** O modelo final (XGBoost) apresentou solidez na identificação de atrasos sistêmicos, sendo sensível a variáveis de horário de pico e histórico imediato da aeronave.
    - **Insights de Negócio:** A análise demonstrou que o "atraso de aeronave tardia" (Late Aircraft Delay) é o principal fator de propagação de atrasos, validando a estratégia de engenharia de features focada em `TAIL_NUMBER`.
    
    ### 4. Tecnologias Utilizadas
    - **Linguagem:** Python 3.x
    - **Processamento:** Pandas, Scikit-learn, XGBoost
    - **Interface:** Streamlit
    - **Visualização:** Plotly (Gráficos interativos com tema Synthwave personalizado)
    """)

    st.divider()
    st.caption("Documentação técnica do Tech Challenge - Grupo de Análise de Dados.")