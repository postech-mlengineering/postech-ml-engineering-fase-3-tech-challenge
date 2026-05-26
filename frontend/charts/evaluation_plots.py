import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff 
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from . import(
    SYNTH_BG,
    SYNTH_TEXT,
    SYNTH_PINK,
    SYNTH_ORANGE,
    SYNTH_GRID,
    SYNTH_FONT
)

def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    cm = confusion_matrix(y_test, y_pred)
    x = ['Previsto Pontual', 'Previsto Atrasado']
    y = ['Real Pontual', 'Real Atrasado']
    
    z = cm[::-1]
    y_labels = y[::-1]

    colorscale = [[0, "#1a1a2e"], [1, SYNTH_PINK]]
    
    fig = ff.create_annotated_heatmap(
        z=z, 
        x=x, 
        y=y_labels, 
        annotation_text=z, 
        colorscale=colorscale,
        font_colors=[SYNTH_TEXT, "white"]
    )
    
    # Aplicação direta do estilo
    fig.update_layout(
        title=dict(text='Matriz de Confusão', font=dict(family=SYNTH_FONT, color=SYNTH_TEXT)),
        paper_bgcolor=SYNTH_BG,
        plot_bgcolor=SYNTH_BG,
        template=None,
        font=dict(family=SYNTH_FONT, color=SYNTH_TEXT),
        xaxis=dict(linecolor=SYNTH_ORANGE, showgrid=False, tickfont=dict(family=SYNTH_FONT)),
        yaxis=dict(linecolor=SYNTH_ORANGE, showgrid=False, tickfont=dict(family=SYNTH_FONT)),
        height=500,
        margin=dict(l=150, r=120, b=120, t=120)
    )
    
    # Forçar a fonte Synthwave nas anotações internas do heatmap
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.family = SYNTH_FONT
        
    return fig

def plot_roc_curve(y_test: pd.Series, y_probs: np.ndarray) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = roc_auc_score(y_test, y_probs)

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'Modelo (AUC: {auc_score:.4f})',
            line=dict(color=SYNTH_TEXT, width=4),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 255, 0.05)'
        )
    )
    
    fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Aleatório',
            line=dict(color=SYNTH_ORANGE, dash='dash')
        )
    )
    
    # Aplicação direta do estilo
    fig.update_layout(
        title=dict(text='Curva ROC Performance', font=dict(family=SYNTH_FONT, color=SYNTH_TEXT)),
        paper_bgcolor=SYNTH_BG,
        plot_bgcolor=SYNTH_BG,
        template=None,
        font=dict(family=SYNTH_FONT, color=SYNTH_TEXT),
        xaxis=dict(title="Taxa Falso Positivo", gridcolor=SYNTH_GRID, linecolor=SYNTH_ORANGE, showgrid=True),
        yaxis=dict(title="Taxa Verdadeiro Positivo", gridcolor=SYNTH_GRID, linecolor=SYNTH_ORANGE, showgrid=True),
        legend=dict(bgcolor=SYNTH_BG, bordercolor=SYNTH_PINK, font=dict(family=SYNTH_FONT)),
        showlegend=True,
        height=600,
        margin=dict(l=150, r=120, b=120, t=120)
    )
    return fig

def plot_bars(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=df['Importância'],
                y=df['Variável'],
                orientation='h',
                marker=dict(
                    color=df['Importância'],
                    colorscale=[[0, SYNTH_PINK], [1, SYNTH_TEXT]],
                    line=dict(color=SYNTH_BG, width=1)
                )
            )
        ]
    )
    
    # Aplicação direta do estilo
    fig.update_layout(
        title=dict(text='Importância das Variáveis', font=dict(family=SYNTH_FONT, color=SYNTH_TEXT)),
        paper_bgcolor=SYNTH_BG,
        plot_bgcolor=SYNTH_BG,
        template=None,
        font=dict(family=SYNTH_FONT, color=SYNTH_TEXT),
        xaxis=dict(title="Importância Relativa", gridcolor=SYNTH_GRID, linecolor=SYNTH_ORANGE, showgrid=False),
        yaxis=dict(autorange='reversed', linecolor=SYNTH_ORANGE, showgrid=False),
        height=550,
        margin=dict(l=400, r=120, b=120, t=120)
    )
    return fig