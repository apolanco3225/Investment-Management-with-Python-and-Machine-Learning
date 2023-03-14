import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm


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

def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolio
    Value Weighted Monthly Returns
    """
    industry_returns = pd.read_csv(
        "../data/ind30_m_vw_rets.csv", 
        index_col=0, 
        parse_dates = True
    ) 
    
    industry_returns = industry_returns / 100
    industry_returns.index = pd.to_datetime(
        industry_returns.index, 
        format="%Y%m"
    ).to_period("M")
    
    industry_returns.columns = industry_returns.columns.str.strip()
    industry_returns.columns = industry_returns.columns.str.lower()
    return industry_returns
    
    

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
    
    
def calculate_semideviation(returns_data):
    """
    Returns the semideviation aka negative semideviation
    of the input returns. The returns can be either a Series
    or a DataFrame
    """
    negative_values_mask = returns_data < 0
    return returns_data[negative_values_mask].std(ddof=0)

def calculate_var_historic(returns_data, level=5):
    """
    Returns the historic value at risk at a specific level.
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above.
    """
    if isinstance(returns_data, pd.DataFrame):
        return returns_data.aggregate(
            calculate_var_historic, 
            level = level
        )
    elif isinstance(returns_data, pd.Series):
        return - np.percentile(
            a = returns_data, 
            q = level
        )
    else:
        raise TypeError ("Expected returns to be a pandas Series or DataFrame")
        

def calculate_var_gaussian(returns_data, level = 5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a pandas Series
    or a DataFrame. 
    If "modified" is True, then the modified VaR is returned
    using the Cornish-Fisher modification. 
    """
    z_score = norm.ppf(level / 100)
    
    if modified:
        # modify the z score based on observed skewness and kurtosis
        s = calculate_skewness(returns_data)
        k = calculate_kurtosis(returns_data)
        z_score = (
            z_score + 
            (z_score ** 2 - 1) * s / 6 +
            (z_score ** 3 - 3 * z_score) * (k - 3) / 24 -
            (2 * z_score ** 3 - 5 * z_score) * (s ** 2) / 36
        )
    return -(returns_data.mean() + z_score * returns_data.std(ddof=0))


def calculate_cvar_historic(returns_data, level=5):
    """
    Computes the conditional var of a Series or a DataFrame
    """
    if isinstance(returns_data, pd.Series):
        is_beyond = r <= calculate_var_historic(returns_data, level=level)
        return -returns_data[is_beyond].mean()
    
    elif isinstance(returns_data, pd.DataFrame):
        return returns_data.aggregate(calculate_cvar_historic, level=level)
    
    else:
        raise TypeError ("Expected returns to be a pandas Series or DataFrame")

        
        
    
def calculate_annualize_vol(input_returns, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    """
    return input_returns.std()*(periods_per_year**0.5)


def calculate_annualize_rets(input_returns, periods_per_year):
    """
    Annualize a set of returns
    We should infer the periods per year
    """
    compounded_growth = (1 + input_returns).prod()
    n_periods = len(input_returns)
    return compounded_growth ** (periods_per_year / n_periods) - 1


def calculate_sharpe_ratio(
    input_returns, 
    risk_free_rate, 
    periods_per_year
    ):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual risk free rate to per period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_ret = input_returns - rf_per_period
    ann_ex_ret = calculate_annualize_rets(excess_ret, periods_per_year)
    ann_vol = calculate_annualize_vol(input_returns, periods_per_year)
    return ann_ex_ret / ann_vol
    