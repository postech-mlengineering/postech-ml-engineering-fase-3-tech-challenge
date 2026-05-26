import pandas as pd
import plotly.graph_objects as go


def plot_scatter(
    df: pd.DataFrame, 
    x: str, 
    y: str, 
    color: str, 
    hover_name: str
) -> None:
    '''
    Gera um gráfico de dispersão (scatter plot) interativo segmentado por categorias.

    Args:
        df (pd.DataFrame): DataFrame contendo os dados para plotagem.
        x (str): Nome da coluna para o eixo X (ex: 'Volume').
        y (str): Nome da coluna para o eixo Y (ex: 'Taxa de Atraso').
        color (str): Nome da coluna usada para agrupar e colorir os pontos (ex: 'CLUSTER').
        hover_name (str): Nome da coluna que será exibida no título do tooltip (ex: 'AIRPORT').

    Returns:
        None: A função exibe o gráfico interativo.
    '''
    fig = go.Figure()
    
    for category in df[color].unique():
        df_subset = df[df[color] == category]
        fig.add_trace(
            go.Scatter(
                x=df_subset[x],
                y=df_subset[y],
                name=str(category),
                mode='markers',
                text=df_subset[hover_name],
                hovertemplate=(
                    '<b>%{text}</b><br>' +
                    'Volume: %{x}<br>' +
                    'Taxa de Atraso: %{y:.4f}<br>' +
                    '<extra></extra>'
                ),
                marker=dict(size=10)
            )
        )
        
    fig.update_layout(
        title=f'Perfil: Volume vs. Taxa de Atraso - {hover_name}',
        xaxis_title='Volume',
        yaxis_title='Taxa de Atraso',
        showlegend=False,
        template='plotly_white'
    )
    fig.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    '''
    Gera um mapa de calor (heatmap) interativo para visualizar a matriz de correlação.

    Args:
        df (pd.DataFrame): DataFrame representando a matriz de correlação 
            (geralmente o resultado de df.corr()).

    Returns:
        None: A função exibe o gráfico interativo.
    '''
    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index
    ))
    
    fig.update_traces(
        colorscale='Blues',
        zmin=-1,
        zmax=1,
        text=df.values,
        texttemplate='%{text:.2f}',
        textfont={'size': 10},
        showscale=False
    )
    
    fig.update_layout(
        title='Matriz de Correlação',
        height=600,
        template='plotly_white',
        xaxis_side='bottom'
    )
    fig.show()