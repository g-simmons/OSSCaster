import dash
from dash.dash_table.DataTable import DataTable
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table
from keras.models import load_model
import plotly.graph_objs as go
import plotly.express as px
import json
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate

import pandas as pd
import numpy as np

from constants import (
    LINEPLOT_STYLE,
    DATATABLE_STYLE,
    INSTRUCTIONS,
    REQUIRED_FEATURES,
    FIGURE_MARGINS,
)
from utils import (
    _clean_df,
    _table_data_to_df,
    _df_to_table_data,
    _parse_contents,
    _get_sample_feature_importances_local,
    _get_sample_feature_importances_global,
)

# external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

CSV_COLUMNS_AS_MONTHS = True


# model = load_model(MODEL_PATH)

instructions = dcc.Markdown(INSTRUCTIONS)

example_data = pd.read_csv("./single_project_100.csv")
sample_data, sample_cols = _df_to_table_data(_clean_df(example_data))
sample_feature_importances_local = _get_sample_feature_importances_local()
sample_feature_importances_global = _get_sample_feature_importances_global()


def get_model_predictions(data: pd.DataFrame):
    return (
        [0.8] * len(data),
        [0.9] * len(data),
        [0.7] * len(data),
    )


def get_explainability_results(data: pd.DataFrame):
    pass


def _update_global_feature_importances(df: pd.DataFrame):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Box(x=df[col], name=col))
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=FIGURE_MARGINS)
    return fig


def _update_local_feature_importances(feature_importances: pd.Series):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=feature_importances.index,
            x=feature_importances.values,
            # name=col,
            orientation="h",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=FIGURE_MARGINS)
    return fig


def _update_figure(data, columns):
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{}], [{}]],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.01,
    )
    fig.update_yaxes(title_text="Success Probability", row=1, col=1)
    fig.update_yaxes(title_text="Project History", row=2, col=1)
    fig.update_xaxes(title_text="Month", row=[1, 2], col=1)

    if data is not None and columns is not None:
        data = _table_data_to_df(data, columns)

        preds, preds_upper, preds_lower = get_model_predictions(data)
        x = data.index.tolist()

        # colors = {"graphBackground": "#F5F5F5", "background": "#ffffff", "text": "#000000"}
        fig.add_traces(
            [
                go.Scatter(x=x, y=data[col], mode="lines+markers", name=col)
                for col in data.columns
            ],
            rows=2,
            cols=1,
        )
        fig.add_traces(
            [
                go.Scatter(
                    x=x, y=preds, mode="lines+markers", name="success_probability"
                )
            ],
            rows=1,
            cols=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],  # x, then x reversed
                y=preds_upper + preds_lower[::-1],  # upper, then lower reversed
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.update_layout(margin=FIGURE_MARGINS)
    return fig


upload_instrs_col = dbc.Col(
    [
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        instructions,
    ],
    width=3,
)
lineplot_title = html.H3("Project History")
lineplot = html.Div(
    dcc.Graph(
        id="lineplot",
        style=LINEPLOT_STYLE,
        figure=_update_figure(sample_data, sample_cols),
    ),
    style={"display": "inline-block", "padding": "0 20"},
)
tablediv = html.Div(
    [
        html.Div(id="filename"),
        html.Div(id="upload-time"),
        dash_table.DataTable(
            data=sample_data,
            columns=sample_cols,
            editable=True,
            id="table",
            style_cell=DATATABLE_STYLE,
        ),
    ],
    style={
        "overflow": "scroll",
    },
)

prediction_bignumber_header = html.H4("Predicted Probability of Graduation")
prediction_bignumber = html.Div(id="prediction-bignumber-div", children=[])
month_features = html.Div(id="month-features-div", children=[])

explain_local_button = html.Button("Explain", id="explain-local-button", n_clicks=0)

month_detail_view = html.Div(
    [
        html.H3("Month Details"),
        html.Div(
            [
                prediction_bignumber_header,
                prediction_bignumber,
                month_features,
                explain_local_button,
            ],
            id="month-detail",
        ),
    ]
)

line_graph_col = dbc.Col(
    [lineplot_title, lineplot, month_detail_view, tablediv],
    width=6,
)

global_explanations = html.Div(
    [
        html.H3("Global Explanations"),
        dcc.Graph(
            id="global-explanations-boxplot",
            figure=_update_global_feature_importances(
                sample_feature_importances_global
            ),
        ),
    ],
)
local_explanations = html.Div(
    [
        html.H3("Local Explanations"),
        dcc.Graph(
            id="local-explanations-barplot",
            figure=_update_local_feature_importances(sample_feature_importances_local),
        ),
    ],
)

explanations_col = dbc.Col(children=[global_explanations, local_explanations], width=3)

app.layout = html.Div(
    dbc.Row(
        [
            upload_instrs_col,
            line_graph_col,
            explanations_col,
        ]
    )
)


@app.callback(
    Output("table", "data"),
    Output("table", "columns"),
    Output("filename", "children"),
    Output("upload-time", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("upload-data", "last_modified"),
)
def update_table(list_of_contents, list_of_names, list_of_dates):
    data = filename = date = columns = None

    if list_of_contents is None:
        raise PreventUpdate

    if list_of_contents is not None:
        filename, date, df = _parse_contents(
            list_of_contents[0], list_of_names[0], list_of_dates[0]
        )
        data, columns = _df_to_table_data(df)

    return data, columns, filename, date


@app.callback(
    Output("local-explanations-barplot", "figure"),
    Input("explain-local-button", "n_clicks"),
)
def update_output(value):
    return _update_local_feature_importances(_get_sample_feature_importances_local())


@app.callback(
    Output("lineplot", "figure"), Input("table", "data"), Input("table", "columns")
)
def update_figure(data, columns):
    if data is None:
        raise PreventUpdate
    return _update_figure(data, columns)


def _get_month_success_prob(figdata, month):
    for trace in figdata:
        if "name" in trace.keys():
            if trace["name"] == "success_probability":
                return trace["y"][month]
    return None


def _get_month_features(figdata, month):
    month_features = {}
    for trace in figdata:
        if "name" in trace.keys():
            if trace["name"] == "success_probability":
                continue
            month_features[trace["name"]] = trace["y"][month]
    return month_features


@app.callback(
    Output("prediction-bignumber-div", "children"),
    Output("month-features-div", "children"),
    Input("lineplot", "clickData"),
    Input("lineplot", "figure"),
)
def update_prediction_bignumber(hover_data, figure):
    month = hover_data["points"][0]["pointNumber"]
    month_for_display = month + 1
    month_success_prob = _get_month_success_prob(figure["data"], month)
    month_features = _get_month_features(figure["data"], month)
    month_features_style = {"font-size": 8, "margin": 0}
    return html.H1(str(month_success_prob)), html.P(
        ", ".join([f"{k}: {v}" for k, v in month_features.items()]),
        style=month_features_style,
    )


if __name__ == "__main__":
    app.run_server(debug=True)
