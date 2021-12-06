from math import exp
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide tensorflow warnings
import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # hide tensorflow warnings
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
from osscaster.explain import SustainabilityExplainer
from osscaster.constants import MODELS_DIR, RANDOM_STATE, DATA_COLUMNS, N_TIMESTEPS

import pandas as pd
import numpy as np

from constants import (
    LINEPLOT_STYLE,
    DATATABLE_STYLE,
    INSTRUCTIONS,
    FIGURE_MARGINS,
    MONTH_FEATURES_STYLE,
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

instructions = dcc.Markdown(INSTRUCTIONS)

example_data = pd.read_csv("./single_project_100.csv")
example_data = example_data[example_data.iloc[:, 0].isin(DATA_COLUMNS)]
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
    data = data.head(N_TIMESTEPS)
    model = load_model(MODELS_DIR / ("model_" + str(N_TIMESTEPS) + ".h5"))
    explainer = SustainabilityExplainer(
        feature_names=DATA_COLUMNS,
        class_names=["Graduated", "Retired"],
        n_timesteps=N_TIMESTEPS,
        random_state=RANDOM_STATE,
    )

    return explainer.explain_by_shap(data.values, model)
    # return explainer.explain_by_lime(data.values, model)


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

explain_local_button = html.Button(
    "Update Predictions", id="update-predictions-button", n_clicks=0
)

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
all_feature_importances = html.P(id="all-feature-importances")

explanations_col = dbc.Col(
    children=[global_explanations, local_explanations, all_feature_importances], width=3
)

app.layout = html.Div(
    dbc.Row(
        [
            upload_instrs_col,
            line_graph_col,
            explanations_col,
        ]
    )
)


def _get_global_feature_importances_from_explainability_results(exp_results):
    feature_importances = [pd.Series(v) for k, v in exp_results.items()]
    feature_importances = pd.concat(feature_importances, axis=1)
    feature_importances = feature_importances.transpose()
    return feature_importances


@app.callback(
    Output("table", "data"),
    Output("table", "columns"),
    Output("filename", "children"),
    Output("upload-time", "children"),
    Output("global-explanations-boxplot", "figure"),
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

    print("Calculating explainability results")
    df = df.set_index("Unnamed: 0")
    df = df.transpose()
    df = df[DATA_COLUMNS]
    exp_results = get_explainability_results(df)
    global_feature_importances_fig = _update_global_feature_importances(
        _get_global_feature_importances_from_explainability_results(exp_results)
    )

    return data, columns, filename, date, global_feature_importances_fig


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


def _get_month_feature_importances(figdata, month):
    feature_importances = {}
    for trace in figdata:
        if "name" in trace.keys():
            if month >= len(trace["x"]):
                raise ValueError(
                    "Selected month is greater than the number of months used in prediction"
                )
            if trace["name"] == "success_probability":
                continue
            feature_importances[trace["name"]] = trace["x"][month]
    return feature_importances


@app.callback(
    Output("prediction-bignumber-div", "children"),
    Output("month-features-div", "children"),
    Output("local-explanations-barplot", "figure"),
    Input("lineplot", "clickData"),
    Input("lineplot", "figure"),
    Input("global-explanations-boxplot", "figure"),
    Input("update-predictions-button", "n_clicks"),
)
def update_prediction_bignumber(hover_data, lineplot_fig, global_explanations_fig, _):
    if hover_data is None:
        raise PreventUpdate
    month = hover_data["points"][0]["pointNumber"]
    month_for_display = month + 1
    month_success_prob = _get_month_success_prob(lineplot_fig["data"], month)
    month_features = _get_month_features(lineplot_fig["data"], month)

    # if user clicks a month that is higher index than the number of months used in prediction,
    # feature importances will not be available, so we populate the graph with zeros.
    # TODO: inform user that feature importances are not available for this month/hide the graph etc.
    if month >= len(global_explanations_fig["data"][0]["x"]):
        local_explanations_fig = _update_local_feature_importances(
            pd.Series(index=DATA_COLUMNS, data=np.zeros(len(DATA_COLUMNS)))
        )
    else:
        month_feature_importances = _get_month_feature_importances(
            global_explanations_fig["data"], month
        )
        local_explanations_fig = _update_local_feature_importances(
            pd.Series(month_feature_importances)
        )

    return (
        html.H1(str(month_success_prob)),
        html.P(
            ", ".join([f"{k}: {v}" for k, v in month_features.items()]),
            style=MONTH_FEATURES_STYLE,
        ),
        local_explanations_fig,
    )


if __name__ == "__main__":
    app.run_server(debug=True)
