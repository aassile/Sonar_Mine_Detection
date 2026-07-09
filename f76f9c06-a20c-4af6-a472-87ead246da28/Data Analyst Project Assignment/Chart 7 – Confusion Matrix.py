import plotly.graph_objects as go
import numpy as np

# Confusion matrix values from model output
# Rows = Actual, Cols = Predicted
# [[TN, FP], [FN, TP]]  →  Rock=0, Mine=1
z_vals = [[16, 4],
           [3, 19]]

labels = ['Rock', 'Mine']

# Annotations
annotations = [
    [f"<b>True Rock</b><br>16", f"<b>False Mine</b><br>4"],
    [f"<b>False Rock</b><br>3", f"<b>True Mine</b><br>19"]
]

# Colour scale: lighter = lower count, darker = higher count
fig_confusion_matrix = go.Figure(data=go.Heatmap(
    z=z_vals,
    x=[f'Predicted {l}' for l in labels],
    y=[f'Actual {l}' for l in labels],
    colorscale=[
        [0.0, '#1D2D3E'],
        [0.5, '#1F77B4'],
        [1.0, '#A1C9F4']
    ],
    showscale=False,
    hovertemplate='%{y} → %{x}<br>Count: %{z}<extra></extra>'
))

# Add text annotations
for i in range(2):
    for j in range(2):
        fig_confusion_matrix.add_annotation(
            x=j, y=i,
            text=annotations[i][j],
            font=dict(size=18, color='#fbfbff'),
            showarrow=False
        )

fig_confusion_matrix.update_layout(
    title=dict(
        text='Confusion Matrix — Random Forest Classifier<br>'
             '<sup>Mine vs. Rock · Test Set (n=42) · Accuracy 83.3%</sup>',
        font=dict(size=17, color='#fbfbff'),
        x=0.5
    ),
    paper_bgcolor='#1D1D20',
    plot_bgcolor='#1D1D20',
    xaxis=dict(
        title='Predicted Label',
        title_font=dict(color='#909094'),
        tickfont=dict(color='#fbfbff', size=13),
        tickvals=[0, 1],
        ticktext=['Predicted Rock', 'Predicted Mine']
    ),
    yaxis=dict(
        title='Actual Label',
        title_font=dict(color='#909094'),
        tickfont=dict(color='#fbfbff', size=13),
        tickvals=[0, 1],
        ticktext=['Actual Rock', 'Actual Mine'],
        autorange='reversed'
    ),
    margin=dict(l=120, r=40, t=100, b=80),
    width=560,
    height=460
)
