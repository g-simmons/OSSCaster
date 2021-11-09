import pandas as pd
import numpy as np
import base64
import datetime
import io

from dash import html
from constants import REQUIRED_FEATURES


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


def _get_sample_feature_importances_local():
    sample_feature_importances_local = np.random.rand(len(REQUIRED_FEATURES))
    return pd.Series(sample_feature_importances_local, index=REQUIRED_FEATURES)


def _get_sample_feature_importances_global():
    sample_feature_importances_global = pd.DataFrame(
        np.array(
            [
                np.random.normal(loc=x, scale=0.1, size=(10,))
                for x in _get_sample_feature_importances_local()
            ]
        ).transpose(),
        columns=REQUIRED_FEATURES,
    )
    return sample_feature_importances_global
