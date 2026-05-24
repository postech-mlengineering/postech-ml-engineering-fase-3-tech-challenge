import plotly.graph_objects as go


def plot_scatter(df, x, y, color, hover_name):
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
                hovertemplate = (
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


def plot_correlation_matrix(df):
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