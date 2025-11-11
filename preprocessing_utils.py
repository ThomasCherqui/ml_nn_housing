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

# Determines which features are categorical and re
def one_hot_encode(data:pd.DataFrame):
    categorical_features = data.select_dtypes(exclude=[np.number]).columns.tolist()
    returnArray = data
    for ftr in categorical_features:
        uniqueList = np.unique(data[ftr])
        for val in uniqueList:
            colName = ftr + str(val)
            def row_lambda(row):
                if row[ftr] == val:
                    return 1
                else:  
                    return 0
            returnArray[colName] = returnArray.apply(row_lambda, axis=1)
        returnArray = returnArray.drop(columns=[ftr])
    return returnArray