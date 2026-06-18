import numpy as np

from sklearn.preprocessing import StandardScaler


def preprocessar(df):

    df["Amount_log"] = np.log1p(df["Amount"])

    scaler = StandardScaler()

    df["Amount_scaled"] = scaler.fit_transform(
        df[["Amount"]]
    )

    return df
