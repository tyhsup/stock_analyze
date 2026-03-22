from finlab import data
from finlab.tools.event_study import create_factor_data


def icir(factor, adj_close, days=[5, 10, 20, 60]):
    """
    Calculate the information coefficient of a factor.

    Args:
        df (pd.DataFrame): A pd.DataFrame containing the factor values.
        adj_close (pd.DataFrame): A pd.DataFrame containing the adjusted close prices.
        period (int): The number of periods to calculate the IC over. Defaults to 1.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the IC values for each period.
    """

    close = data.get('price:收盤價')
    adj_close = data.get('etl:adj_close')
    fdata = create_factor_data({'factor':close / close.average(60)}, adj_close, event=None, days=[5, 10, 20, 60])

    ict = fdata.groupby(level=0).corrwith(fdata['factor_factor']).iloc[:,-len(days):]

    return {
        'ic': fdata.corrwith(fdata['factor_factor']).iloc[-len(days):],
        'ic_t': ict,
        'ir': ict.mean() / ict.std()
    }