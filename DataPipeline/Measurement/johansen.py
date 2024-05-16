import numpy    as np
import pylab    as pl

from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(data):
    """
    Perform the Johansen test of cointegration on a given dataset.

    Parameters:
    - data: A numpy array or pandas DataFrame containing the time series data.

    Returns:
    - result: A dictionary containing the test statistics and critical values.

    """
    result = {}

    # Perform the Johansen test
    #result['test_statistic'], result['eigenvectors'], result['critical_values'] = coint_johansen(data, det_order=1, k_ar_diff=1)

    result  = coint_johansen(data, det_order=-1, k_ar_diff=1)
    print("CVM")
    print(result.cvm)
    print("CVT")
    print(result.cvt)
    print("Eig")
    print(result.eig)
    print("Evec")
    print(result.evec)
    print("Ind")
    print(result.ind)
    print("LR1")
    print(result.lr1)
    print("LR2")
    print(result.lr2)
    print("Max Eig")
    print(result.max_eig_stat)
    print("Max Eig Crit")
    print(result.max_eig_stat_crit_vals)
    print("Meth")
    print(result.meth)
    print("R0t")
    print(result.r0t)
    print("Rkt")
    print(result.rkt)
    print("Trace")
    print(result.trace_stat)
    print("Trace Crit")
    print(result.trace_stat_crit_vals)

    return result

# Example usage
if __name__ == "__main__":
    import pandas   as pd
    import pylab    as pl

    # Step 1: Prepare data
    # Example: Assume `data` is a DataFrame with time series data for multiple assets
    # Replace with actual asset data
    np.random.seed(42)
    
    dates   = pd.date_range('2020-01-01', periods=100, freq='D')
    asset1  = np.random.normal(size=100).cumsum()
    asset2  = asset1 + np.random.normal(size=100)
    asset3  = 2 * asset1 + np.random.normal(size=100)
    
    data = pd.DataFrame({'asset1': asset1, 'asset2': asset2, 'asset3': asset3}, index=dates)

    # Step 2: Apply Johansen test
    result = coint_johansen(data, det_order=0, k_ar_diff=1)

    # Step 3: Calculate the spread
    # Using the first eigenvector for the cointegrating relationship
    eigenvector = result.evec[:, 0]
    spread      = np.dot(data.values, eigenvector)

    # Normalize the spread
    spread = (spread - np.mean(spread)) / np.std(spread)

    # Step 4: Develop the trading strategy
    # Define trading signals based on spread z-scores
    z_score_threshold   = 1.5  # Threshold for entering trades
    
    positions   = np.zeros_like(spread)
    positions[spread > z_score_threshold]   = -1  # Short when spread is above the threshold
    positions[spread < -z_score_threshold]  = 1  # Long when spread is below the threshold

    # Step 5: Backtest the strategy
    # Calculate returns
    data['spread']      = spread
    data['positions']   = positions
    data['returns']     = data['positions'].shift(1) * data['asset1'].pct_change()
    
    data['strategy_returns']    = data['returns'].cumsum()

    # Plot the results
    pl.figure(figsize=(14, 7))
    pl.plot(data.index, data['strategy_returns'], label='Strategy Returns')
    pl.plot(data.index, data['spread'], label='Spread')
    pl.axhline(z_score_threshold, color='red', linestyle='--', label='Upper Threshold')
    pl.axhline(-z_score_threshold, color='green', linestyle='--', label='Lower Threshold')
    pl.title('Mean-Reverting Strategy Based on Johansen Test')
    pl.legend()
    pl.show()