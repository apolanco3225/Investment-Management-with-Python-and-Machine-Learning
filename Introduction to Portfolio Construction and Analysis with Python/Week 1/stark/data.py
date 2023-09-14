# quant finance data library Arturo Polanco
import pandas as pd

def fama_french_market_equity():
    """
    Load the Fama-French Dataset for the returns
    of the top and bottom deciles by market cap.
    """
    fama_french_data  = pd.read_csv(
        "../data/Portfolios_Formed_on_ME_monthly_EW.csv",
        header = 0, 
        index_col = 0, 
        na_values = -99.99
    )

    fama_french_data = fama_french_data[["Lo 10", "Hi 10"]]
    fama_french_data.columns  = ["SmallCap", "LargeCap"]
    fama_french_data = fama_french_data.divide(100)
    fama_french_data.index = pd.to_datetime(
        fama_french_data.index, 
        format="%Y%m"
    ).to_period("M")
    return fama_french_data