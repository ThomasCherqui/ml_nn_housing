import pandas as pd
import numpy as np

def impute_missing_data(data, method='mean'):
    if method == "mean":
        data =  data.fillna(data.mean)
    if method == "median":
        data =  data.fillna(data.mean)

    return data

def normalize_numerical(data:pd.DataFrame):
    """Normalize numerical features to range [0,1]."""
    numerical_features = data.select_dtypes(include=[np.number]).columns.tolist()
    normalized_data = data.copy()
    for col in numerical_features:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        if max_val != min_val:  # avoid division by zero
            normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        else:
            normalized_data[col] = 0.0
    return normalized_data

