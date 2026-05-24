import plotly.graph_objects as go
import plotly.figure_factory as ff 
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
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
        title=title,
        xaxis_title='Predição',
        yaxis_title='Realidade',
        template='plotly_white'
    )
    fig.show()


def plot_roc_curve(y_true, y_probs, title):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

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
        title=title,
        xaxis_title='Taxa de Falso Positivo (1 - Especificidade)',
        yaxis_title='Taxa de Verdadeiro Positivo (Sensibilidade)',
        height=600,
        template='plotly_white'
    )
    fig.show()