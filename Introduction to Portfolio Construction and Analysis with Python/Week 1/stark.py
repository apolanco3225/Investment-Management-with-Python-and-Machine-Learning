import pandas as pd

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