import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize

from tqdm import tqdm

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
    
def get_ind_size():
    """
    Load and format the Ken French 30 Industry Portfolio Size
    """
    industry_returns = pd.read_csv(
        "../data/ind30_m_size.csv", 
        index_col=0, 
        parse_dates = True
    ) 
    
    industry_returns.index = pd.to_datetime(
        industry_returns.index, 
        format="%Y%m"
    ).to_period("M")
    
    industry_returns.columns = industry_returns.columns.str.strip()
    industry_returns.columns = industry_returns.columns.str.lower()
    return industry_returns
   
    
def get_ind_nfirms():
    """
    Load and format the Ken French 30 Industry Portfolio 
    number of firms.
    """
    industry_returns = pd.read_csv(
        "../data/ind30_m_nfirms.csv", 
        index_col=0, 
        parse_dates = True
    ) 
    
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
        is_beyond = returns_data <= calculate_var_historic(returns_data, level=level)
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
    
    
def calculate_portfolio_return(weights, returns_data):
    """
    Weights -> Returns
    """
    return weights.T @ returns_data


def calculate_portfolio_vol(weights, cov_mat):
    """
    Weights -> Volatility
    """
    return (weights.T @ cov_mat @ weights) ** 0.5


def plot_ef2(n_points, returns_data, cov_matrix):
    """
    Plots the 2-assets efficient frontier
    """
    if returns_data.shape[0] != 2 or returns_data.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2 asset frontiers")
        
    # iterate to obtain weights depending on n_points input variable    
    weights = [np.array([w, 1 - w]) for w in np.linspace(0, 1, n_points)]

    # calculate returns for every weight
    returns = [
        calculate_portfolio_return(
            w, 
            returns_data
        ) for w in weights
    ]
    
    # calculate volatility for every weight
    volatility = [
            calculate_portfolio_vol(
                w, 
                cov_matrix
            ) for w in weights
    ]

    # construct dataframe for efficient frontier
    efficient_frontier_df = pd.DataFrame(
        {
            "returns": returns,
            "volatility": volatility
        }
    )

    return efficient_frontier_df.plot.line(
        x = "volatility",
        y = "returns", 
        style = ".-"
    )


def minimize_vol(target_return, returns_data, cov_matrix):
    """
    target_return -> W
    """
    num_assets = len(returns_data)
    unit_weight = 1 / num_assets
    init_guess = np.repeat(unit_weight, num_assets)
    
    bounds = ((0.0, 1.0),) * num_assets
    
    return_is_target = {
        "type": "eq",
        "args": (returns_data,),
        "fun": lambda weights, returns_data: 
            target_return - calculate_portfolio_return(
                weights, 
                returns_data
            )
    }
    weights_sum_to_one = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }
    
    results = minimize(
        calculate_portfolio_vol, 
        init_guess, 
        args = (cov_matrix,),
        method = "SLSQP",
        options = {"disp": False},
        constraints = (return_is_target, weights_sum_to_one), 
        bounds = bounds
    )
    
    return results.x

def optimal_weights(n_points, returns_data, cov_matrix):
    """
    -> Generates a list of weights to run the optimizer
    on to minimize the vol.
    """
    target_returns = np.linspace(
        returns_data.min(), 
        returns_data.max(), 
        n_points
    )
    
    weights = [
        minimize_vol(target_return, returns_data, cov_matrix)
        for target_return in target_returns
    ]
    return weights
    
    
    
def calculate_gmv(cov_matrix):
    """
    Returns weights of the global minimal vol portfolio
    given covariance matrix.
    """
    num_assets = len(cov_matrix)
    return calculate_maximum_sharpe_ratio(0, np.repeat(1, num_assets), cov_matrix)
    
    
def plot_ef(n_points, returns_data, cov_matrix, show_cml=False, risk_free_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the N-assets efficient frontier
    """
    # iterate to obtain weights depending on n_points input variable    
    weights = optimal_weights(n_points, returns_data, cov_matrix)

    # calculate returns for every weight
    returns = [
        calculate_portfolio_return(
            w, 
            returns_data
        ) for w in weights
    ]
    
    # calculate volatility for every weight
    volatility = [
            calculate_portfolio_vol(
                w, 
                cov_matrix
            ) for w in weights
    ]

    # construct dataframe for efficient frontier
    efficient_frontier_df = pd.DataFrame(
        {
            "returns": returns,
            "volatility": volatility
        }
    )

    ax = efficient_frontier_df.plot.line(
        x = "volatility",
        y = "returns", 
        style = ".-"
    )
    
    if show_ew:
        # equally weighted portfolio
        num_assets = len(returns_data)
        weights_ew = np.repeat(1 / num_assets, num_assets)
        returns_ew = calculate_portfolio_return(weights_ew, returns_data)
        volatility_ew = calculate_portfolio_vol(weights_ew, cov_matrix)
        # display EW porfolio
        ax.plot(
            [volatility_ew], 
            [returns_ew], 
            color="goldenrod", 
            marker="o", 
            markersize=12
        )
        
    if show_gmv:
        # global minimum variance portfolio
        weights_gmv = calculate_gmv(cov_matrix)
        returns_gmv = calculate_portfolio_return(weights_gmv, returns_data)
        volatility_gmv = calculate_portfolio_vol(weights_gmv, cov_matrix)
        # display GMV porfolio
        ax.plot(
            [volatility_gmv], 
            [returns_gmv], 
            color="midnightblue", 
            marker="o", 
            markersize=12
        )

    if show_cml:
        
        ax.set_xlim(left=0)

        weights_msr = calculate_maximum_sharpe_ratio(
            risk_free_rate, 
            returns_data, 
            cov_matrix
        )

        returns_msr = calculate_portfolio_return(
            weights_msr, 
            returns_data
        )

        volatility_msr = calculate_portfolio_vol(
            weights_msr, 
            cov_matrix
        )

        # add capital market line
        cml_x = [0, volatility_msr]
        cml_y = [risk_free_rate, returns_msr]
        ax.plot(
            cml_x, 
            cml_y, 
            color ="green", 
            marker ="o", 
            linestyle ="dashed", 
            markersize =12,
            linewidth = 2
        )


    return ax



def calculate_maximum_sharpe_ratio(risk_free_rate, returns_data, cov_matrix):
    """
    Maximum Sharpe Ratio
    Risk Free Rate + ER + COV -> W
    """
    num_assets = len(returns_data)
    unit_weight = 1 / num_assets
    init_guess = np.repeat(unit_weight, num_assets)
    
    bounds = ((0.0, 1.0),) * num_assets
    
    weights_sum_to_one = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, risk_free_rate, returns_data, cov_matrix):
        """
        Returns the negative of the sharpe ratio, given the weights
        """
        returns = calculate_portfolio_return(weights, returns_data)
        volatility = calculate_portfolio_vol(weights, cov_matrix)
        
        return -(returns - risk_free_rate) / volatility
    
    results = minimize(
        neg_sharpe_ratio, 
        init_guess, 
        args = (
            risk_free_rate, 
            returns_data,
            cov_matrix,),
        method = "SLSQP",
        options = {"disp": False},
        constraints = (weights_sum_to_one), 
        bounds = bounds
    )
    
    return results.x

# execute
def run_cppi(risky_returns, safe_returns=None, m=3, start=1_000, floor=0.8, risk_free_rate=0.03, drawdown_constraint=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for 
    the history asset. 
    Returns a dictionary containing: Asset value history, risk budget history,
    risky weight history.
    """
    dates = risky_returns.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start
    
    if isinstance(risky_returns, pd.Series):
        risky_returns = pd.DataFrame(risky_returns, columns=["R"])
        
    if safe_returns == None:
        safe_returns = pd.DataFrame().reindex_like(risky_returns)
        safe_returns.values[:] = risk_free_rate / 12
        
        
    account_history = pd.DataFrame().reindex_like(risky_returns)
    cushion_history = pd.DataFrame().reindex_like(risky_returns)
    risky_w_history = pd.DataFrame().reindex_like(risky_returns)

    for step in tqdm(range(n_steps)):
        if drawdown_constraint is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown_constraint)
        cushion = (account_value - floor_value) / account_value
        risky_weight = m * cushion
        risky_weight = np.minimum(risky_weight, 1)
        risky_weight = np.maximum(risky_weight, 0)

        safe_weight = 1 - risky_weight

        risky_allocation = account_value * risky_weight
        safe_allocation = account_value * safe_weight

        # update account value for this time step
        account_value = (risky_allocation * (risky_returns.iloc[step] + 1)) + \
            (safe_allocation * (safe_returns.iloc[step] + 1))

        # save values  
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_weight
        account_history.iloc[step] = account_value
        
    risky_wealth = start * (1 + risky_returns).cumprod()
    
    backtest_dict = {
        "wealth": account_history, 
        "risky_wealth": risky_wealth, 
        "risky_budget": cushion_history, 
        "risk_allocation": risky_w_history, 
        "m": m, 
        "start": start, 
        "floor": floor, 
        "risky_return": risky_returns, 
        "safe_returns": safe_returns
    }
    return backtest_dict


def summary_stats(returns_data, risk_free_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats
    for the returns in the columns of returns_data
    """
    annual_returns = returns_data.aggregate(calculate_annualize_rets, periods_per_year = 12)
    annual_volatility = returns_data.aggregate(calculate_annualize_vol, periods_per_year = 12)
    annual_sharp_ratio = returns_data.aggregate(
        calculate_sharpe_ratio, 
        risk_free_rate = risk_free_rate, 
        periods_per_year = 12
    )
    drawdown = returns_data.aggregate(
        lambda returns_data: 
            calculate_drawdown(returns_data).drawdown.min()
    )
    skewness = returns_data.aggregate(calculate_skewness)
    kurtosis = returns_data.aggregate(calculate_kurtosis)
    cf_var5 = returns_data.aggregate(calculate_var_gaussian, modified=True)
    hist_cvar5 = returns_data.aggregate(calculate_cvar_historic)
    summary_dict = {
        "annualized_return": annual_returns, 
        "annualized_volatility":annual_volatility, 
        "skewness": skewness,
        "kurtosis": kurtosis, 
        "cornish_fisher_var": cf_var5, 
        "historic_var": hist_cvar5, 
        "sharpe_ratio": annual_sharp_ratio, 
        "drawdown": drawdown
    }
    
    summary_df = pd.DataFrame(summary_dict)
    return summary_df


def geometric_brownian_motion(
    n_years = 10, 
    n_scenarios = 1_000, 
    mu=0.07, 
    sigma = 0.15,
    steps_per_year = 12, 
    s_0 = 100.0
):
    """
    Evolution of a stock price using geometric brownian motion model
    """
    delta_time = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year)
    rets_plus_one = np.random.normal(
        loc = (1 + mu * delta_time), 
        scale = (sigma * np.sqrt(delta_time)), 
        size=(n_steps, n_scenarios)
    )
    prices = s_0 * pd.DataFrame(rets_plus_one).cumprod()
    return prices