# quant finance metrics library Arturo Polanco
import pandas as pd
import numpy as np
import scipy.stats

def drawdown(input_series:pd.Series):
    """
    Takes a time series of returns and 
    computes and returns:
    - Wealth index
    - Previous peaks
    - Percent Drawdown
    """

    output_wealth_index  = input_series.add(1).cumprod().multiply(1_000)
    output_previous_peaks = output_wealth_index.cummax()
    output_drawdowns = output_wealth_index.subtract(
        output_previous_peaks).divide(
            output_previous_peaks)
    
    output_data = pd.DataFrame(
        {
            "wealth": output_wealth_index,
            "peaks": output_previous_peaks,
            "drawdown": output_drawdowns
        }
    )
    return output_data


def skewness(input_returns):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied series or dataframe
    Returns a float or a series.
    """
    demeaned_returns = input_returns.subtract(input_returns.mean())
    # use the population standard deviation, so set dof=0
    sigma_r = input_returns.std(ddof=0)
    sigma_r_cubed = sigma_r ** 3
    exp = demeaned_returns.pow(3).mean()

    return exp / sigma_r_cubed


def kurtosis(input_returns):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied series or dataframe
    Returns a float or a series.
    """
    demeaned_returns = input_returns.subtract(input_returns.mean())
    # use the population standard deviation, so set dof=0
    sigma_r = input_returns.std(ddof=0)
    sigma_r_the_fourth = sigma_r ** 4
    exp = demeaned_returns.pow(4).mean()

    return exp / sigma_r_the_fourth


def is_normal(input_returns, level=0.01):
    """
    Apply Jarque-Bera test to check
    for normal distribution returns. 
    Test is applied at the 1% level by default.
    Returns True if the hypothesis of normality is accepted, 
    False otherwise. 
    """

    statistics, p_value = scipy.stats.jarque_bera(input_returns)

    return p_value > level


def semi_deviation(input_returns):
    """
    Returns the semi deviation aka as negative semi deviation of returns
    returns most be a series or a dataframe
    """
    negative_mask = input_returns < 0
    negative_returns = input_returns[negative_mask]
    return negative_returns.std(ddof=0)



def var_historic(input_returns, level=5):
    """
    Returns the historic value at risk at a specified level.
    """
    if isinstance(input_returns, pd.DataFrame):
        return input_returns.aggregate(var_historic, level=level)

    elif isinstance(input_returns, pd.Series):
        return -np.percentile(a = input_returns, q=level)
    else:
        raise TypeError("Expected input to be pandas series or dataframe.")
    

def var_gaussian(input_returns, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or Dataframe
    If "modified" is true, then returns the Cornish-Fisher modification.
    """
    # compute z score assuming gaussian distribution
    z = scipy.stats.norm.ppf(level/100)

    if modified:
        # modify the z score based on observed skewness and kurtosis
        s = skewness(input_returns)
        k = kurtosis(input_returns)
        z = (z +
             (z**2 - 1) * s/6 +
             (z**3 - 3*z) * (k-3)/24 - 
             (2* z**3 - 5*z) * (s**2) / 36 
             )
        

    return -input_returns.std(ddof=0).multiply(z).add(input_returns.mean())



def cvar_historic(input_returns, level=5):
    """
    Returns the historic value at risk at a specified level.
    """
    if isinstance(input_returns, pd.Series):
        is_beyond = input_returns <= -var_historic(input_returns, level=level)
        return -input_returns[is_beyond].mean()

    elif isinstance(input_returns, pd.DataFrame):
        return input_returns.aggregate(cvar_historic, level=level)

    else:
        raise TypeError("Expected input to be pandas series or dataframe.")
    