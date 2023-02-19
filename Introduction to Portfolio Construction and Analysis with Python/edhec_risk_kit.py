import pandas as pd

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
    