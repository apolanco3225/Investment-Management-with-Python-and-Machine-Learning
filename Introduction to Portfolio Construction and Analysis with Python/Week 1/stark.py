import pandas as pd
import scipy.stats

# data convinience functions
def get_ffme_returns():
    """
    Load the Fama-French dataset for the returns of the
    top and botton deciles by market cap
    """
    # load data
    returns = pd.read_csv(
        "../data/Portfolios_Formed_on_ME_monthly_EW.csv",
        header = 0, 
        index_col = 0, 
        na_values = -99.99, 
    )
    # select columns
    columns = ["Lo 10", "Hi 10"]
    portfolio_returns = returns[columns]
    # transform percentage values into float
    portfolio_returns = portfolio_returns / 100
    # rename columns
    new_columns_names = ["SmallCap", "LargeCap"]
    portfolio_returns.columns = new_columns_names
    # set time column to datetime type
    portfolio_returns.index = pd.to_datetime(
        portfolio_returns.index, format="%Y%m"
    ).to_period("M")
    return portfolio_returns


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi_returns = pd.read_csv(
        "../data/edhec-hedgefundindices.csv", 
        header = 0, 
        index_col = 0,
        parse_dates = True, 
        infer_datetime_format=True

    )
    hfi_returns = hfi_returns / 100
    hfi_returns.index = hfi_returns.index.to_period("M")
    return hfi_returns


# risk functions
def calculate_drawdown(input_series: pd.Series):
    """
    Takes a time series of returns
    Computes and returns a DataFrame:
        - Wealth index
        - Previous peaks
        - Percentage drawdown
    """
    # calculate components
    wealth_index = 1_000 * (1 + input_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks) / previous_peaks
    # create output dataframe with previous calculations
    drawdown_data = pd.DataFrame(
        {
            "wealth": wealth_index,
            "peaks": previous_peaks, 
            "drawdown": drawdown
        }
    )
    return drawdown_data


def calculate_skewness(returns_data):
    """
    Alternative to scipy.stats.skew()
    Compute the skewness of the supplied Series or DataFrame
    Returns a float or a series
    """
    demeaned_returns = returns_data - returns_data.mean()
    # use the population standard deviation, so set dof=0
    sigma_returns = returns_data.std(ddof=0)
    exponent = (demeaned_returns ** 3).mean()
    return exponent / sigma_returns ** 3


def calculate_kurtosis(returns_data):
    """
    Alternative to scipy.stats.kurtosis()
    Compute the kurtosis of the supplied Series or DataFrame
    Returns a float or a series
    """
    demeaned_returns = returns_data - returns_data.mean()
    # use the population standard deviation, so set dof=0
    sigma_returns = returns_data.std(ddof=0)
    exponent = (demeaned_returns ** 4).mean()
    return exponent / sigma_returns ** 4


def is_normal(returns_data, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a series is normal or not
    Test is applied at the 1% level by default
    Returns true if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(returns_data)
    return p_value > level
    