import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

import itertools

pio.renderers.default = 'iframe'


def show_pie(df, along_feature, w=800, h=600, title=""):
    fig = go.Figure()

    vc = df.groupby(along_feature).count().reset_index()
    fig.add_trace(go.Pie(labels=vc.iloc[:,0], values=vc.iloc[:,1]))
    fig.update_layout(title=title, height=h, width=w)
    fig.show()
    
    
def show_lines(x, y_dict, y_log=False, x_log=False, title="", h=None):
    fig = make_subplots(rows=len(y_dict.keys()), cols=1)
    
    if h is None:
        h = 30 + 200 * len(y_dict.keys())
    j = 1
    for line_name in y_dict.keys():
        fig.add_trace(
            go.Scatter(x=x, y=y_dict[line_name], mode='lines', name=line_name),
            row = j, col = 1
        )
        j += 1
    if y_log:
        fig.update_yaxes(type="log")
    if x_log:
        fig.update_xaxes(type="log")
        
    fig.update_layout(
        title=title,
        height = h
    )
    fig.show()

    
def show_pca_variance_graph(pca, w=800, h=800):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(range(1, pca.n_components + 1)),
            y=pca.explained_variance_ratio_ * 100,
            hoverinfo='x+y',
            name="Variance expliquée" 
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, pca.n_components + 1)), 
            y=np.cumsum(pca.explained_variance_ratio_*100),
            mode='lines',
            name='Variance expliquée cumulée'
        )
    )

    fig.update_layout(
        title=f"""Ebouli des valeurs propres avec {pca.n_components} composantes""",
        title_x=0.5,
        autosize=False,
        width=w,
        height=h,
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=0.99
        )
    )
    fig.show()
    
def show_pca_correlation_graph(pca, x_y, features, w=800, h=600): 
    colors_cycle = itertools.cycle(px.colors.qualitative.Dark24)

    x,y=x_y

    fig = go.Figure()
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-1, y0=-1, x1=1, y1=1,
        line_color="Black",
    )
    
    for i in range(0, pca.components_.shape[1]):
        color = next(colors_cycle)
        fig.add_annotation(
            x=pca.components_[x, i],
            y=pca.components_[y, i],
            xref="x", yref="y",
            showarrow=True,
            axref = "x", ayref='y',
            ax=0,
            ay=0,
            arrowhead=4,
            arrowwidth=2,
            arrowcolor=color,
            font=dict(
                color=color,
                size=12
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=[pca.components_[x, i]] + 0.05 * pca.components_[x, i] ,
                y=[pca.components_[y, i]] + 0.05 * pca.components_[y, i] ,
                text=[features[i]],
                mode='text',
                marker=dict(
                    color = color
                ),
                name=features[i]
            ), 
        )
        
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        width=w,
        height=h,
    )
    fig.update_xaxes(range=[-1.1, 1.1], title=dict(text = f"F{x+1}"))
    fig.update_yaxes(range=[-1.1, 1.1], title=dict(text = f"F{y+1}"))
    
    fig.for_each_trace(lambda t: t.update(textfont_color=t.marker.color))
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    fig.show()
    
    
def show_scatter_graph(
    x, y, colors=None, w=800, h=600, title="", dict_color_labels=dict()
):
    fig = go.Figure()
    if colors is not None:
        colors_cycle = itertools.cycle(px.colors.qualitative.Dark24)
        dict_colors = {}
        for c in colors.unique():
            dict_colors[c] = next(colors_cycle)

        for c in colors.unique():
            l = list(colors.index[colors == c])

            fig.add_trace(
                go.Scattergl(
                    x=x[l],
                    y=y[l],
                    mode="markers",
                    name=dict_color_labels.get(c, c),
                    marker=dict(
                        color=dict_colors[c],
                    ),
                )
            )
    else:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
            )
        )
    fig.update_layout(title=title, height=h, width=w)
    fig.show()
    
    
def show_cols_boxplots_by_col(df, cols_to_show, by_col, w=800, h=600, title="", orient="h"):
    
    if isinstance(by_col, str)==False:
        df["score"] = by_col
        by_col = "score"
        
    if orient=="h":
        fig = make_subplots(rows=1, cols=len(cols_to_show), subplot_titles=(cols_to_show))
    else:
        fig = make_subplots(cols=1, rows=len(cols_to_show), subplot_titles=(cols_to_show))
        
    i = 1
    for c in cols_to_show:
        colors_cycle = itertools.cycle(px.colors.qualitative.Dark24)
        if orient=="h":
            row = 1
            col = i
        else:
            row = i
            col = 1
        for label in df[by_col].unique():
            fig.add_trace(
                go.Box(y=df.loc[df[by_col]==label, c], 
                    name=label,
                    legendgroup='group1',
                    showlegend = i==1,
                    line_color=next(colors_cycle),
                ),
                row=row, col=col,
            )
        i += 1
    fig.update_layout(
        width=w,
        height=h,
        title=title
    )
    fig.show()

    
def view_clustered_heatmap(df, clusterer, scaler, num_cols=None, title="", height=1500):
    df2 = df.copy(deep=True)

    if num_cols is None:
        num_cols = df2.select_dtypes(include=[np.number]).columns

    df2 = pd.DataFrame(
        scaler.inverse_transform(df.loc[:, num_cols]),
        columns=num_cols,
    )

    df2 = df2.reset_index(drop=True)
    df2["cluster_score"] = clusterer.labels_.astype(str)

    df2_sorted = df2.sort_values("cluster_score").reset_index(drop=True)

    clust_rect = []
    for b in df2_sorted["cluster_score"].unique():
        clust_rect.append(
            {
                "xmin": df2_sorted[df2_sorted["cluster_score"] == b].index.min(),
                "xmax": df2_sorted[df2_sorted["cluster_score"] == b].index.max(),
                "label": f"Cluster {b}",
            }
        )

    fig = make_subplots(
        rows=len(num_cols), cols=1, vertical_spacing=0, subplot_titles=df2.columns
    )

    for y in range(0, len(num_cols)):
        fig.add_trace(
            go.Heatmap(
                z=df2_sorted.loc[:, [num_cols[y]]].T,
                colorscale="ylorrd",
                colorbar=dict(y=1 - (y / len(num_cols) + 0.15), len=1 / len(num_cols)),
            ),
            row=y + 1,
            col=1,
        )
        if y == 0:
            fig.layout["yaxis"].title.text = num_cols[y]
        else:
            fig.layout[f"yaxis{y+1}"].title.text = num_cols[y]
            fig.update_xaxes(showticklabels=False)  # hide all the xticks

    for cb in clust_rect:
        fig.add_vline(
            x=cb["xmin"],
        )
        fig.add_annotation(
            x=cb["xmin"] + 3000,
            y=-0.15,
            text=cb["label"],
            showarrow=False,
            xref="x",
            yref="paper",
        )
        fig.add_annotation(
            x=cb["xmin"] + 3000,
            y=1.15,
            text=cb["label"],
            showarrow=False,
            xref="x",
            yref="paper",
        )

    fig.update_layout(
        title=title,
        height=height,
    )
    fig.update_xaxes(showticklabels=True, row=len(num_cols), col=1)
    fig.show()