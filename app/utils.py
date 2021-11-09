import pandas as pd
import base64
import datetime
import io

from dash import html


def _table_data_to_df(data, columns):
    df = pd.DataFrame(data, columns=[c["name"] for c in columns])
    df = df.set_index(df.columns[0])
    df = df.transpose()
    return df


def _df_to_table_data(df):
    data = df.to_dict("records")
    columns = [{"name": i, "id": i} for i in df.columns]
    return data, columns


def _clean_df(df):
    df = df.dropna(how="all", axis=1)
    df = df.round(2)
    return df


def _parse_contents(contents, filename, date):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

        df = _clean_df(df)
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return html.H5(filename), html.H6(datetime.datetime.fromtimestamp(date), id=""), df
