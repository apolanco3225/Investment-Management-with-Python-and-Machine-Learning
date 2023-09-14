# quant finance metrics library Arturo Polanco
import pandas as pd
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