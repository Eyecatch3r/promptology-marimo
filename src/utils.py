import polars as pl
import numpy as np

def symmetric_to_condensed(matrix: np.ndarray) -> np.ndarray:
    """Convert a symmetric matrix (2D array) to its condensed form (1D
    array of upper triangle, excluding diagonal)."""
    # Extract upper triangle without diagonal
    return matrix[np.triu_indices_from(matrix, k=1)]

def get_nan_feats(df, feats):
    res = {}
    idx = df.select(feats).select(pl.all().is_nan().any())
    for f in feats:
        if idx[f].sum() > 0:
            res[f] = df[f].is_nan().sum()
    return res

def get_null_feats(df, feats):
    res = {}
    idx = df.select(feats).select(pl.all().is_null().any())
    for f in feats:
        if idx[f].sum() > 0:
            res[f] = df[f].is_null().sum()
    return res

def get_inf_feats(df, feats):
    res = {}
    idx = df.select(feats).select(pl.all().is_infinite().any())
    for f in feats:
        if idx[f].sum() > 0:
            res[f] = df[f].is_infinite().sum()
    return res

