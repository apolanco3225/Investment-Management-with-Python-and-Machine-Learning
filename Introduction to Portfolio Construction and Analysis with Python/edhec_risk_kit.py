import pandas as pd
import numpy as np
import scipy
from scipy.stats import norm
from scipy.optimize import minimize

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
    
    
def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolio
    Value Weighted Monthly Returns
    """
    ind = pd.read_csv(
        "data/ind30_m_vw_rets.csv", 
        index_col=0, 
        parse_dates = True
    ) 
    
    ind = ind / 100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    
    ind.columns = ind.columns.str.strip()
    ind.columns = ind.columns.str.lower()
    return ind
    
    
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
       
    
def annualize_vol(input_r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    """
    return input_r.std()*(periods_per_year**0.5)


def annualize_rets(input_r, periods_per_year):
    """
    Annualize a set of returns
    We should infer the periods per year
    """
    compounded_growth = (1 + input_r).prod()
    n_periods = input_r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def sharpe_ratio(
    input_r, 
    risk_free_rate, 
    periods_per_year
    ):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual risk free rate to per period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_ret = input_r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(input_r, periods_per_year)
    return ann_ex_ret / ann_vol
    
    
def portfolio_return(weight, returns):
    """
    Weights -> Returns
    """
    return weight.T @ returns


def portfolio_vol(weight, covmat):
    """
    Weights -> Vol
    """
    return (weight.T @ covmat @ weight) ** 0.5



def plot_ef2(n_points, er, cov, style=".-"):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2 asset frontiers")
        
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    
    ef = pd.DataFrame(
        {
            "Returns": rets, 
            "Volatility": vols
        }
    )
    return ef.plot.line(x="Volatility", y="Returns", style=style)



def minimize_vol(target_return, er, cov):
    """
    target_return -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    
    bounds = ((0.0, 1.0),) * n
    
    return_is_target = {
        "type": "eq",
        "args": (er,),
        "fun": lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_one = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }
    
    results = minimize(
        portfolio_vol, 
        init_guess, 
        args = (cov,),
        method = "SLSQP",
        options = {"disp": False},
        constraints = (return_is_target, weights_sum_to_one), 
        bounds = bounds
    )
    
    return results.x



def optimal_weights(n_points, er, cov):
    """
    -> List of weights to run the optimizer on
    to minimize the vol.
    """
    target_rs = np.linspace(er.mean(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


def plot_ef(n_points, er, cov, style=".-"):
    """
    Plots the N-asset efficient frontier
    """
  
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    
    ef = pd.DataFrame(
        {
            "Returns": rets, 
            "Volatility": vols
        }
    )
    return ef.plot.line(x="Volatility", y="Returns", style=style) 