import base64
import datetime
import io
from json import load

import dash
from dash.dash_table.DataTable import DataTable
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash import dash_table
from keras.models import load_model
import plotly.graph_objs as go

import pandas as pd

# external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

CSV_COLUMNS_AS_MONTHS = True

# MODEL_PATH = "./model.h5"

# model = load_model(MODEL_PATH)

app.layout = html.Div(
    dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            ["Drag and Drop or ", html.A("Select Files")]
                        ),
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
                    html.Div(
                        id="output-data-upload",
                        style={
                            "overflow": "scroll",
                        },
                    ),
                ],
                width=4,
            ),
            dbc.Col([
                html.P(["Project History"]),
                html.Div([
                    dcc.Graph(
                        id='crossfilter-indicator-scatter',
                    )],
                style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),],
            width=8),
        ]
    ),
)


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")),index_col=0)
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded),index_col=0)
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return (df, html.Div(
        [
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),
            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
            ),
        ]
    ))


@app.callback(
    Output("output-data-upload", "children"),
    Output('crossfilter-indicator-scatter', "figure"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("upload-data", "last_modified"),
)
def update_output(list_of_contents, list_of_names, list_of_dates):
    children = None
    data = None
    fig = go.Figure(
    )
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d)
            for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
        ]
        data, children = children[0]

        if CSV_COLUMNS_AS_MONTHS:
            print("transposing")
            data = data.transpose()

        data = data.dropna(how='all')

        colors = {"graphBackground": "#F5F5F5", "background": "#ffffff", "text": "#000000"}
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=data.index,
                    y=data[col],
                    mode='lines+markers',name=col) for col in data.columns

                ],
            layout=go.Layout(
                plot_bgcolor=colors["graphBackground"],
                paper_bgcolor=colors["graphBackground"]
            )
        )


    return children, fig


if __name__ == "__main__":
    app.run_server(debug=True)
