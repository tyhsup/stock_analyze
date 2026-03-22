import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List
from alphalens.tears import GridFigure

from alphalens import plotting
from alphalens import performance as perf
from alphalens import utils
import alphalens
from finlab.ml.feature import combine

# fix error orginal code: http://quantopian.github.io/
@plotting.customize
def create_turnover_tear_sheet(factor_data, turnover_periods=None):

    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True,
        ).to_numpy()
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(
            turnover_periods,
        )

    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    autocorrelation = pd.concat(
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    fr_cols = len(turnover_periods)
    columns_wide = 1
    rows_when_wide = ((fr_cols - 1) // 1) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    for period in turnover_periods:
        if quantile_turnover[period].isnull().all().all():
            continue
        plotting.plot_top_bottom_quantile_turnover(
            quantile_turnover[period], period=period, ax=gf.next_row()
        )

    for period in autocorrelation:
        if autocorrelation[period].isnull().all():
            continue
        plotting.plot_factor_rank_auto_correlation(
            autocorrelation[period], period=period, ax=gf.next_row()
        )

    plt.show()
    gf.close()

alphalens.tears.create_turnover_tear_sheet = create_turnover_tear_sheet


def factor_weights(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False):
    """
    Computes asset weights by factor values and dividing by the sum of their
    absolute value (achieving gross leverage of 1). Positive factor values will
    results in positive weights and negative values in negative weights.

    Args:
        factor_data (pd.DataFrame - MultiIndex):
            A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
            containing the values for a single alpha factor, forward returns for
            each period, the factor quantile/bin that factor value belongs to, and
            (optionally) the group the asset belongs to.
            - See full explanation in utils.get_clean_factor_and_forward_returns
        demeaned (bool):
            Should this computation happen on a long short portfolio? if True,
            weights are computed by demeaning factor values and dividing by the sum
            of their absolute value (achieving gross leverage of 1). The sum of
            positive weights will be the same as the negative weights (absolute
            value), suitable for a dollar neutral long-short portfolio
        group_adjust (bool):
            Should this computation happen on a group neutral portfolio? If True,
            compute group neutral weights: each group will weight the same and
            if 'demeaned' is enabled the factor values demeaning will occur on the
            group level.
        equal_weight (bool, optional):
            if True the assets will be equal-weighted instead of factor-weighted
            If demeaned is True then the factor universe will be split in two
            equal sized groups, top assets with positive weights and bottom assets
            with negative weights

    Returns:
        returns : pd.Series
            Assets weighted by factor value.
    """

    def to_weights(group, _demeaned, _equal_weight):

        if _equal_weight:
            group = group.copy()

            if _demeaned:
                # top assets positive weights, bottom ones negative
                group = group - group.median()

            negative_mask = group < 0
            group[negative_mask] = -1.0
            positive_mask = group > 0
            group[positive_mask] = 1.0

            if _demeaned:
                # positive weights must equal negative weights
                if negative_mask.any():
                    group[negative_mask] /= negative_mask.sum()
                if positive_mask.any():
                    group[positive_mask] /= positive_mask.sum()

        elif _demeaned:
            group = group - group.mean()

        return group / group.abs().sum()

    grouper = ['date']
    if group_adjust:
        grouper.append('group')

    weights = factor_data.groupby(grouper, group_keys=False)['factor'] \
        .apply(to_weights, demeaned, equal_weight)

    if group_adjust:
        weights = weights.groupby(level='date', group_keys=False).apply(to_weights, False, False)

    return weights

from alphalens import performance
performance.factor_weights = factor_weights


def create_factor_data(factor:pd.DataFrame, adj_close:pd.DataFrame, 
                       days:List[int]=[5,10,20, 60]):

    '''create factor data, which contains future return

    Args:
        factor (pd.DataFrame): factor data where index is datetime and columns is asset id
        adj_close (pd.DataFrame): adj close where index is datetime and columns is asset id
        days (List[int]): future return considered
        
    Return:
        Analytic plots and tables

    Examples:
        ``` py title="股價淨值比分析"
        import alphalens
        from finlab import data
        from finlab.ml.alphalens import create_factor_data

        factor = data.get('price_earning_ratio:股價淨值比')
        adj_close = data.get('etl:adj_close')

        factor_data = create_factor_data(factor, adj_close)

        alphalens.tears.create_full_tear_sheet(factor_data.dropna(), long_short=False,
                                               group_neutral=False, by_group=False)

        ```

    '''

    adj_close = adj_close.loc[factor.index[0]:factor.index[-1]]
    factor = factor.reindex(adj_close.index, method='ffill').loc[factor.index[0]:factor.index[-1]]

    sids = adj_close.columns.intersection(factor.columns)
    adj_close = adj_close[sids]
    factor = factor[sids]

    ret = {}
    ret['factor'] = factor.unstack().values
    ret['factor_quantile'] = (factor.rank(axis=1, pct=True) // 0.2).unstack().values

    total_index = None

    for d in days:
        temp = (adj_close.shift(-d-1) / adj_close.shift(-1) - 1).unstack()
        ret[f"{d}D"] = temp.values
        total_index = temp.index
    ret = pd.DataFrame(ret, index=total_index.swaplevel(0, 1))\
        .replace([-np.inf, np.inf], np.nan)\
        .dropna()
    ret.index.names = ['date', 'asset']
    return ret
