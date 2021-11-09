import dash
from dash.dash_table.DataTable import DataTable
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table
from keras.models import load_model
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate

import pandas as pd

from constants import LINEPLOT_STYLE
from utils import _clean_df, _table_data_to_df, _df_to_table_data, _parse_contents

# external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

CSV_COLUMNS_AS_MONTHS = True


# model = load_model(MODEL_PATH)


example_data = pd.read_csv("./single_project_100.csv")
sample_data, sample_cols = _df_to_table_data(_clean_df(example_data))


def get_model_predictions(data: pd.DataFrame):
    return (
        [0.8] * len(data),
        [0.9] * len(data),
        [0.7] * len(data),
    )


def get_explainability_results(data: pd.DataFrame):
    pass


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
    ],
    width=3,
)
lineplot_title = html.P("Project History")
lineplot = html.Div(
    dcc.Graph(id="lineplot", style=LINEPLOT_STYLE),
    style={"display": "inline-block", "padding": "0 20"},
)
tablediv = html.Div(
    [
        html.Div(id="filename"),
        html.Div(id="upload-time"),
        dash_table.DataTable(
            data=sample_data, columns=sample_cols, editable=True, id="table"
        ),
    ],
    style={
        "overflow": "scroll",
    },
)

line_graph_col = dbc.Col(
    [lineplot_title, lineplot, tablediv],
    width=6,
)

explanations_col = dbc.Col(children=None, width=3)

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
    Output("lineplot", "figure"), Input("table", "data"), Input("table", "columns")
)
def update_figure(data, columns):
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
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
