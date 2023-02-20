import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm

def calculate_drawdown(input_series: pd.Series):
    """
    It takes a time series of returns 
    Computes and returns a DataFrame that contains:
        - Wealth indedx
        - Previous peaks
        - Percent drawdowns
    """
    wealth_index = 1_000 * (1 + input_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    output_data = pd.DataFrame(
        {
            "wealth": wealth_index, 
            "peaks": previous_peaks, 
            "drawdown": drawdowns
        }
    )
    return output_data


def get_ffme_returns():
    """
    Load the Famma-French dataset for the returns of the
    top and Bottom Deciles by MarketCap. 
    """
    me_m = pd.read_csv(
        "data/Portfolios_Formed_on_ME_monthly_EW.csv",
        header=0, 
        index_col=0, 
        na_values=-99.99,  
    )
    
    rets = me_m[["Lo 10", "Hi 10"]]
    rets.columns = ["SmallCap", "LargeCap"]
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M")
    return rets
    
    
def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Return
    """
    hfi = pd.read_csv(
        "data/edhec-hedgefundindices.csv",
        header = 0, 
        index_col = 0, 
        parse_dates = True,
        infer_datetime_format=True
    )
    
    hfi = hfi / 100
    hfi.index = hfi.index.to_period("M")
    return hfi
    
    
def calculate_skewness(input_r):
    """
    Alternative to scipy.stats.skew()
    Compute the skeweness of the supplied Series or DataFrame
    Returns a float or a Series.
    """
    demeaned_r = input_r - input_r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = input_r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp / sigma_r ** 3
    
    
def calculate_kurtosis(input_r):
    """
    Alternative to scipy.stats.kurtosis()
    Compute the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series.
    """
    demeaned_r = input_r - input_r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = input_r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp / sigma_r ** 4
    
    
def is_normal(input_r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a series is 
    normal or not. 
    Test is applied at the 1% level by default.
    Rerturns true if they hypothesis of normality is accepted, 
    false otherwise.
    """
    statistic, p_value = scipy.stats.jarque_bera(input_r)
    return p_value > level


def calculate_semideviation(input_r):
    """
    Returns semideviation aka negative semideviation of r
    r must me a series or a DataFrame
    """
    is_negative = input_r < 0
    return input_r[is_negative].std(ddof=0)

def var_historic(input_r, level=5):
    """
    Returns the historic VaR Value at Risk at a specified
    level i.e. returns the number such that "level" percent 
    of the returns fall below that number, and the (100 level)
    percent are above
    """
    if isinstance(input_r, pd.DataFrame):
        return input_r.aggregate(var_historic, level=level)
    elif isinstance(input_r, pd.Series):
        return -np.percentile(input_r, level)
    else:
        raise TypeError("Expected r input to be a pandas Series or Dataframe")
       
    
def var_gaussian(input_r, level=5, modified=False):
    """
    Returns the parametric Gaussian VaR of a Series or a Dataframe
    """
    # compute the z score assuming it was gaussian
    z = norm.ppf(level / 100)
    
    if modified:
        # modify the z score based on observed skewness and kurtosis
        s = calculate_skewness(input_r)
        k = calculate_kurtosis(input_r)
        z = (
            z +
            (z ** 2 - 1) * s / 6 +
            (z ** 3 - 3 * z) * (k - 3) / 24 -
            (2 * z ** 3 - 5 * z) * (s ** 2) / 36
        )

    return -(input_r.mean() + z * input_r.std(ddof=0))


def cvar_historic(input_r, level=5):
    """
    Returns the historic VaR Value at Risk at a specified
    level i.e. returns the number such that "level" percent 
    of the returns fall below that number, and the (100 level)
    percent are above
    """
    if isinstance(input_r, pd.Series):
        is_beyond = input_r <= - var_historic(input_r, level=level)
        return - input_r[is_beyond].mean()
    elif isinstance(input_r, pd.DataFrame):
        return input_r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r input to be a pandas Series or Dataframe")
       