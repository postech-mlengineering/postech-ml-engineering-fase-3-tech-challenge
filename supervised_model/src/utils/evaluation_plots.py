from typing import Union, List

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff 
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> None:
    '''
    Gera e exibe uma matriz de confusão interativa utilizando Plotly.

    Args:
        y_test (pd.Series): Vetor com os valores reais (target).
        y_pred (np.ndarray): Vetor com os valores preditos pelo modelo.

    Returns:
        None: A função exibe o gráfico diretamente no notebook.
    '''
    cm = confusion_matrix(y_test, y_pred)
    x = ['Previsto Pontual', 'Previsto Atrasado']
    y = ['Real Pontual', 'Real Atrasado']
    
    fig = ff.create_annotated_heatmap(
        z=cm, 
        x=x, 
        y=y, 
        annotation_text=cm, 
        colorscale='Blues'
    )
    fig.update_layout(
        title='Matriz de Confusão',
        xaxis_title='Predição',
        yaxis_title='Realidade',
        template='plotly_white'
    )
    fig.show()


def plot_roc_curve(y_test: pd.Series, y_probs: np.ndarray) -> None:
    '''
    Calcula e plota a curva ROC (Receiver Operating Characteristic) e o valor da AUC.

    Args:
        y_test (pd.Series): Vetor com os valores reais.
        y_probs (np.ndarray): Vetor com as probabilidades da classe positiva 
            (geralmente obtido via model.predict_proba()[:, 1]).

    Returns:
        None: A função exibe o gráfico diretamente no notebook.
    '''
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr,
            mode='lines',
            name=f'AUC = {auc_score:.4f}',
            line=dict(color='darkblue', width=3)
        )
    )
    fig.add_trace(go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            name='Predição Aleatória',
            line=dict(color='red', dash='dash')
        )
    )
    fig.update_layout(
        title='Curva ROC',
        xaxis_title='Taxa de Falso Positivo (1 - Especificidade)',
        yaxis_title='Taxa de Verdadeiro Positivo (Sensibilidade)',
        height=600,
        template='plotly_white'
    )
    fig.show()


def plot_feature_importances(df: pd.DataFrame) -> None:
    '''
    Cria um gráfico de barras horizontal representando a importância de cada variável.

    Args:
        df (pd.DataFrame): DataFrame contendo obrigatoriamente as colunas 
            ['Variável', 'Importância'], ordenado da maior para a menor importância.

    Returns:
        None: A função exibe o gráfico diretamente no notebook.
    '''
    fig = go.Figure(
        data=[
            go.Bar(
                x=df['Importância'],
                y=df['Variável'],
                orientation='h',
                marker=dict(
                    color=df['Importância'],
                    colorscale='Blues',
                    showscale=False
                )
            )
        ]
    )
    fig.update_layout(
        title={
            'text': 'Importância das Variáveis Explanatórias',
            'y': 0.95,
            'x': 0.05,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        xaxis_title='Importância',
        yaxis_title='Variável',
        yaxis=dict(autorange='reversed'),
        margin=dict(l=150, r=30, t=60, b=50),
        template='plotly_white',
        height=500
    )
    fig.show()