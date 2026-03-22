import pandas as pd
import numpy as np
from typing import Union
from collections.abc import Iterable
from itertools import combinations


def resampler(df, resample, **kwargs) -> pd.DataFrame:

    if resample is None:
        return df
    elif isinstance(resample, Iterable) and not isinstance(resample, str):
        return df.reindex(resample, method='ffill')

    return df.resample(resample, closed='right', label='right', **kwargs).last()



def cpcv(df, num_splits=6, test_size=2):
    """
    Performs Combinatorial Purged Cross-Validation (CPCV) on a pandas DataFrame.

    Parameters:
        df: pandas DataFrame
        num_splits: Number of CPCV splits to perform (default=6)
        test_size: Number of splits to use for testing (default=2)

    Returns:
        Generator that yields tuples of indices for the training and testing sets.
    """
    # Define variables
    n = num_splits
    s = test_size
    block_size = int(len(df) / n)  # Calculate block size based on number of splits
    combs = set(range(n))  # Set of all possible splits
    
    # Loop over all combinations of splits for testing and training
    for prod in combinations(combs, r=s):
        # Calculate the indices for the training and testing sets
        train_idx = sum([np.arange(i*block_size, (i+1)*block_size, 1).tolist() for i in combs - set(prod)], [])
        test_idx = sum([np.arange(i*block_size, (i+1)*block_size).tolist() for i in prod], [])
        
        # Yield a tuple of the training and testing indices
        yield (train_idx, test_idx)


def combine_cpcv(splits: Union[pd.DataFrame, pd.Series]):
    """
    Combines a list of time series datasets into a list of combined datasets.

    Parameters:
    ys: List of pandas DataFrames, each representing a time series dataset.

    Returns:
    List of pandas DataFrames, each representing a combined time series dataset.
    """
    # Initialize dictionaries to store combined time series datasets
    creturns = {0:[]}
    init_keys = {0:[]}

    # Loop over each time series dataset
    for s in splits:
        # Get the initial index value of the dataset
        init_key = s.index[0]

        # Search for an existing dataset with the same initial index value
        found = False
        for k in init_keys:
            if init_key not in init_keys[k]:
                # If no dataset is found, append the current dataset to a new entry
                init_keys[k].append(init_key)
                creturns[k].append(s)
                found = True
                break

        # If a dataset is found, append the current dataset to the existing entry
        if not found:
            new_return_id = len(creturns)
            init_keys[new_return_id] = [init_key]
            creturns[new_return_id] = [s]

    # Combine each set of datasets into a single time series dataset
    ret = []
    for s in creturns.values():
        comb_s = pd.concat(s).sort_index()
        ret.append(comb_s)

    # Return the list of combined time series datasets
    return ret
