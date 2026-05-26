import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from . import(
    SYNTH_BG,
    SYNTH_TEXT,
    SYNTH_PINK,
    SYNTH_ORANGE,
    SYNTH_GRID,
    SYNTH_FONT
)


def plot_scatter(df: pd.DataFrame, x: str, y: str, color: str, hover_name: str) -> go.Figure:
    fig = go.Figure()
    
    # Paleta de cores neon para os clusters
    colors = [SYNTH_PINK, SYNTH_TEXT, SYNTH_ORANGE, "#9d00ff", "#00ff00"]
    
    for i, category in enumerate(df[color].unique()):
        df_subset = df[df[color] == category]
        fig.add_trace(
            go.Scatter(
                x=df_subset[x],
                y=df_subset[y],
                name=str(category),
                mode='markers',
                text=df_subset[hover_name],
                hovertemplate='<b>%{text}</b><br>Volume: %{x}<br>Taxa: %{y:.4f}<extra></extra>',
                marker=dict(
                    size=12, 
                    color=colors[i % len(colors)], 
                    line=dict(width=1, color="white"),
                    opacity=0.8
                )
            )
        )
        
    fig.update_layout(
        title=dict(text=f'Clusters: Volume Vs. Taxa de Atraso', font=dict(family=SYNTH_FONT, color=SYNTH_TEXT)),
        paper_bgcolor=SYNTH_BG,
        plot_bgcolor=SYNTH_BG,
        template=None,
        font=dict(family=SYNTH_FONT, color=SYNTH_TEXT),
        xaxis=dict(title='Volume', gridcolor=SYNTH_GRID, linecolor=SYNTH_ORANGE, showgrid=True),
        yaxis=dict(title='Taxa de Atraso', gridcolor=SYNTH_GRID, linecolor=SYNTH_ORANGE, showgrid=True),
        legend=dict(bgcolor=SYNTH_BG, bordercolor=SYNTH_PINK, font=dict(family=SYNTH_FONT)),
        margin=dict(l=100, r=100, b=100, t=100),
        showlegend=True
    )
    return fig

def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale=[[0, SYNTH_PINK], [0.5, "#1a1a2e"], [1, SYNTH_TEXT]],
        zmin=-1, zmax=1,
        text=df.values,
        texttemplate='%{text:.2f}',
        showscale=True
    ))
    
    fig.update_layout(
        title=dict(text='Matriz de Correlação', font=dict(family=SYNTH_FONT, color=SYNTH_TEXT)),
        paper_bgcolor=SYNTH_BG,
        plot_bgcolor=SYNTH_BG,
        template=None,
        font=dict(family=SYNTH_FONT, color=SYNTH_TEXT),
        xaxis=dict(linecolor=SYNTH_ORANGE, showgrid=False),
        yaxis=dict(linecolor=SYNTH_ORANGE, showgrid=False),
        height=600,
        margin=dict(l=150, r=120, b=120, t=120)
    )
    return fig

def plot_pie(df):
    df_temp = df['IS_DELAYED'].value_counts().reset_index()
    df_temp.columns = ['TARGET_STATUS', 'COUNT']
    df_temp['STATUS_NAME'] = df_temp['TARGET_STATUS'].map({0: 'Pontual', 1: 'Atrasado'})

    colors = [SYNTH_TEXT, SYNTH_PINK]

    fig = go.Figure(data=[go.Pie(
        labels=df_temp['STATUS_NAME'],
        values=df_temp['COUNT'],
        hole=0.5,
        marker=dict(colors=colors, line=dict(color=SYNTH_BG, width=2)),
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        title=dict(text='Distribuição de Voos Atrasados e Pontuais', font=dict(family=SYNTH_FONT, color=SYNTH_TEXT)),
        paper_bgcolor=SYNTH_BG,
        plot_bgcolor=SYNTH_BG,
        template=None,
        font=dict(family=SYNTH_FONT, color=SYNTH_TEXT),
        height=550,
        margin=dict(l=50, r=50, b=50, t=100)
    )
    return fig

def plot_columns(df, columns, labels, title):
    df_metrics = df[columns].describe()
    df_metrics.loc['cv'] = df_metrics.loc['std'] / df_metrics.loc['mean']
    df_metrics.loc['kurtosis'] = df[columns].kurtosis()

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Média de Minutos', 'Coeficiente de Variação', 'Curtose'),
        vertical_spacing=0.12
    )
    
    colorscale_synth = [[0, SYNTH_PINK], [1, SYNTH_TEXT]]

    # Média
    fig.add_trace(go.Bar(
        x=labels, y=df_metrics.loc['mean', columns],
        marker=dict(color=df_metrics.loc['mean', columns], colorscale=colorscale_synth),
        name='Média', textposition='auto'
    ), row=1, col=1)

    # CV
    fig.add_trace(go.Bar(
        x=labels, y=df_metrics.loc['cv', columns],
        marker=dict(color=df_metrics.loc['cv', columns], colorscale=colorscale_synth),
        name='CV'
    ), row=2, col=1)
    fig.add_hline(y=1, line_dash='dash', line_color=SYNTH_ORANGE, row=2, col=1)

    # Curtose
    fig.add_trace(go.Bar(
        x=labels, y=df_metrics.loc['kurtosis', columns],
        marker=dict(color=df_metrics.loc['kurtosis', columns], colorscale=colorscale_synth),
        name='Curtose'
    ), row=3, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(family=SYNTH_FONT, size=20, color=SYNTH_TEXT)),
        paper_bgcolor=SYNTH_BG,
        plot_bgcolor=SYNTH_BG,
        template=None,
        font=dict(family=SYNTH_FONT, color=SYNTH_TEXT),
        height=900,
        showlegend=False,
        margin=dict(l=120, r=120, b=120, t=120)
    )

    # Ajuste manual de eixos e títulos de subplots
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=14, color=SYNTH_TEXT, family=SYNTH_FONT)
    
    for axis in ['xaxis', 'xaxis2', 'xaxis3']:
        fig['layout'][axis].update(linecolor=SYNTH_ORANGE, gridcolor=SYNTH_GRID)
    for axis in ['yaxis', 'yaxis2', 'yaxis3']:
        fig['layout'][axis].update(linecolor=SYNTH_ORANGE, gridcolor=SYNTH_GRID)
        
    return fig

def plot_columns_lines(df):
    df_temp = df.copy()
    if 'SCHEDULED_DEPARTURE_HOUR' not in df_temp.columns:
        df_temp['SCHEDULED_DEPARTURE_HOUR'] = df_temp['SCHEDULED_DEPARTURE'] // 100

    df_month = df_temp.groupby('MONTH')['IS_DELAYED'].mean().reset_index()
    df_dow = df_temp.groupby('DAY_OF_WEEK')['IS_DELAYED'].mean().reset_index()
    df_hour = df_temp.groupby('SCHEDULED_DEPARTURE_HOUR')['IS_DELAYED'].mean().reset_index()

    fig = make_subplots(
        rows=3, cols=1, 
        subplot_titles=('% Atraso por Mês', '% Atraso por Dia', '% Atraso por Hora'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(go.Bar(x=df_month['MONTH'], y=df_month['IS_DELAYED'], marker_color=SYNTH_PINK), row=1, col=1)
    fig.add_trace(go.Bar(x=df_dow['DAY_OF_WEEK'], y=df_dow['IS_DELAYED'], marker_color=SYNTH_ORANGE), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df_hour['SCHEDULED_DEPARTURE_HOUR'], y=df_hour['IS_DELAYED'], 
        mode='lines+markers', line=dict(color=SYNTH_TEXT, width=3),
        marker=dict(size=8, symbol='diamond')
    ), row=3, col=1)

    fig.update_layout(
        title=dict(text='Distribuição Temporal de Atrasos', font=dict(family=SYNTH_FONT, color=SYNTH_TEXT)),
        paper_bgcolor=SYNTH_BG,
        plot_bgcolor=SYNTH_BG,
        template=None,
        font=dict(family=SYNTH_FONT, color=SYNTH_TEXT),
        height=900,
        showlegend=False,
        margin=dict(l=120, r=120, b=120, t=120)
    )

    for i in fig['layout']['annotations']:
        i['font'] = dict(size=14, color=SYNTH_TEXT, family=SYNTH_FONT)
        
    for axis in ['xaxis', 'xaxis2', 'xaxis3']:
        fig['layout'][axis].update(linecolor=SYNTH_ORANGE, gridcolor=SYNTH_GRID)
    for axis in ['yaxis', 'yaxis2', 'yaxis3']:
        fig['layout'][axis].update(linecolor=SYNTH_ORANGE, gridcolor=SYNTH_GRID)

    return fig