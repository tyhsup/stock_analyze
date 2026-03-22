import datetime
import pandas as pd
import numpy as np
from typing import Union, List, Optional
from itertools import combinations


class CPCV(object):

    def __init__(self, df:Union[pd.DataFrame, pd.Series], 
                 num_splits:int=6, 
                 test_size:int=2, 
                 perge_period:datetime.timedelta=datetime.timedelta(days=3)):
        
        self.ref_df = df
        self.num_splits = num_splits
        self.test_size = test_size
        self.perge_period = perge_period

    def train_test_split(self, df:Optional[Union[pd.DataFrame, pd.Series]]=None):
        return cpcv_train_test_split(df or self.ref_df, 
                                     num_splits=self.num_splits, 
                                     test_size=self.test_size, 
                                     perge_period=self.perge_period)
    
    def combine_splits(self, splits:List[Union[pd.DataFrame, pd.Series]]):
        return cpcv_combine_splits(splits=splits, 
                                   df=self.ref_df, 
                                   num_splits=self.num_splits, 
                                   test_size=self.test_size)
    
    
def get_bin_start_end(df, num_splits):

    ret = []
    block_size = int(len(df) / num_splits)
    dt = df.index.get_level_values('datetime')
    for i in range(num_splits):
        t_start = dt[i * block_size]
        t_end = dt[min((i + 1) * block_size, len(dt)-1)]
        ret.append((t_start, t_end))
    return ret

def cpcv_train_test_split(df, num_splits=6, test_size=2, perge_period=datetime.timedelta(days=3)):
    """
    Performs Combinatorial Purged Cross-Validation (CPCV) on a pandas DataFrame.

    Parameters:
        df: pandas DataFrame
        num_splits: Number of CPCV splits to perform (default=6)
        test_size: Number of splits to use for testing (default=2)

    Returns:
        Generator that yields tuples of indices for the training and testing sets.
    """
    
    # Get the start and end indices of the bins
    bin_se = get_bin_start_end(df, num_splits)

    # Create a set of bin indices
    combs = set(range(num_splits))
    # Get the datetime index from the DataFrame
    dt = df.index.get_level_values('datetime')

    # Iterate over combinations of test bin indices
    for c in combinations(combs, r=test_size):
        test_bin_ids = set(c)
        # Calculate the train bin indices by subtracting the test bin indices from the full set
        train_bin_ids = (combs - test_bin_ids)

        # Initialize lists to store train and test indices
        train_ids = []
        test_ids = []

        # Iterate over train bin indices
        for i in train_bin_ids:
            # Get the start and end indices for the current train bin
            s, e = bin_se[i]

            # If the next bin is in the test set, subtract the perged value from the end index
            if i + 1 in test_bin_ids:
                e -= perge_period

            # Find the indices of the DataFrame that are within the start and end of the current train bin
            train_ids.append(np.where((dt >= s) & (dt < e))[0])

        # Iterate over test bin indices
        for i in test_bin_ids:
            # Get the start and end indices for the current test bin
            s, e = bin_se[i]
            # Find the indices of the DataFrame that are within the start and end of the current test bin
            test_ids.append(np.where((dt >= s) & (dt < e))[0])

        # Concatenate the train and test indices
        is_train = np.concatenate(train_ids, axis=0)
        is_test = np.concatenate(test_ids, axis=0)

        # Yield a tuple of train and test indices
        yield (is_train, is_test)

def cpcv_combine_splits(splits:List[Union[pd.DataFrame, pd.Series]], df:Union[pd.DataFrame, pd.Series], num_splits=6, test_size=2):

    """
    Combines a list of time series datasets into a list of combined datasets.

    Parameters:
    ys: List of pandas DataFrames, each representing a time series dataset.

    Returns:
    List of pandas DataFrames, each representing a combined time series dataset.
    """
    
    bin_se = get_bin_start_end(df, num_splits)

    bins = {i:[] for i in range(num_splits)}
    
    # split dataframe into bins
    for sp in splits:
        
        for i, (s, e) in enumerate(bin_se):
            if s in sp.index:
                dt = sp.index.get_level_values('datetime')
                bins[i].append(sp[(dt >= s) & (dt < e)])

    # check the number of bins
    path_lengths = set([len(v) for v in bins.values()])
    if len(path_lengths) != 1:
        raise Exception("There is a potential bug in combine_cpcv since the lengths of path are not the same")
    
    # get path length
    path_length = list(path_lengths)[0]

    # combine pathes
    paths = []
    for p in range(path_length):
        selected_bins = []
        for i in range(num_splits):
            selected_bins.append(bins[i][p])
        
        paths.append(pd.concat(selected_bins))
    
    return paths
