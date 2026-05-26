import joblib

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from services.data_service import load_data

from charts.evaluation_plots import (
    plot_bars, 
    plot_confusion_matrix, 
    plot_roc_curve
)


def model_performance_page():

    st.title("Performance")
    
    df, df_features = load_data()

    if df is None or df.empty:
        st.warning("Nenhum dado encontrado. Verifique a fonte de dados.")
        return

    st.divider()

    tab0, tab1, tab2, tab3 = st.tabs(["Relatório de Classificação", "Matriz de Confusão", "Curva ROC", "Peso das Variáveis"])

    try:
        df_features = pd.read_pickle('../data/curated/features.pkl')
        model = joblib.load("../models/model_xgboost.pkl")
        
        X = df_features.drop(columns=['IS_DELAYED'])
        y = df_features['IS_DELAYED']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        with tab0:
            report = classification_report(y_test, y_pred, output_dict=True)
            df = pd.DataFrame(report)
            df = df.rename(columns={'0': 'Pontual', '1': 'Atrasado'})
            st.dataframe(df.transpose())

        with tab1:
            fig = plot_confusion_matrix(y_test, y_pred)
            st.plotly_chart(fig, width='stretch')

        with tab2:
            fig = plot_roc_curve(y_test, y_probs)
            st.plotly_chart(fig, width='stretch')

        with tab3:
            feat_imp_df = pd.DataFrame({
                'Variável': model.feature_names_in_,
                'Importância': model.feature_importances_
            }).sort_values(by='Importância', ascending=False)
            fig = plot_bars(feat_imp_df)
            st.plotly_chart(fig, width='stretch')

    except FileNotFoundError:
        st.error("Erro: Arquivos de modelo ou features não encontrados nos diretórios configurados.")
    except Exception as e:
        st.error(f"Erro ao gerar métricas de performance: {e}")