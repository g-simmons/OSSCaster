import pandas as pd
import numpy as np
import base64
import datetime
from typing import List, Dict
import io
import logging

from dash import html
from osscaster.constants import DATA_COLUMNS


def _table_data_to_df(data, columns):
    df = pd.DataFrame(data, columns=[c["name"] for c in columns])
    # df = df.transpose()
    # df = df.set_index(df.columns[0])
    return df


def _uploaded_df_to_table_data(df) -> List[Dict[str, str]]:
    """
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with first column containing feature names and remaining columns containing feature values
    """
    if not all([x in df.iloc[:, 0]] for x in DATA_COLUMNS):
        raise ValueError("Dataframe does not contain all required columns")
    df = df.set_index(df.columns[0])
    df = df.transpose()
    if not set(df.columns) == set(DATA_COLUMNS):
        raise ValueError
    if not len(df) > 0:
        raise ValueError
    data = df.to_dict("records")
    columns = [{"name": str(i), "id": str(i)} for i in df.columns]
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
        logging.error(e)
        return html.Div(["There was an error processing this file."])

    return html.H5(filename), html.H6(datetime.datetime.fromtimestamp(date), id=""), df


def _get_sample_feature_importances_local():
    sample_feature_importances_local = np.random.rand(len(DATA_COLUMNS))
    return pd.Series(sample_feature_importances_local, index=DATA_COLUMNS)


def _get_sample_feature_importances_global():
    sample_feature_importances_global = pd.DataFrame(
        np.array(
            [
                np.random.normal(loc=x, scale=0.1, size=(10,))
                for x in _get_sample_feature_importances_local()
            ]
        ).transpose(),
        columns=DATA_COLUMNS,
    )
    return sample_feature_importances_global
