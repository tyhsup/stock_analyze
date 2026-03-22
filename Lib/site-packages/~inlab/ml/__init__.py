import finlab.market_info


_market = finlab.market_info.TWMarketInfo()

def set_market(market:finlab.market_info.MarketInfo):

    """
    Set the stock market for FinLab machine learning model to generate features and labels.

    Args:
        market (MarketInfo): A MarketInfo object representing the market.
    """

    global _market

    if isinstance(market, type):
        _market = market()
    else:
        _market = market

def get_market():
    global _market
    return _market

def reset_market():

    """
    Reset the stock market for FinLab machine learning model to the default market, TWMarketInfo.
    """

    global _market
    _market = TWMarketInfo()
