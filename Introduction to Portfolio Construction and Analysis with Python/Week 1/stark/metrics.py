# quant finance metrics library Arturo Polanco
import pandas as pd

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