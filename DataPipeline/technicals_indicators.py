import pandas   as pd
import numpy    as np

from typing             import Union, Iterable, Optional
from functools          import partial
from scipy.stats        import linregress
from collections.abc    import Iterable
from concurrent.futures import ProcessPoolExecutor

"""
Technical Indicators documentation:

### Volume Based Indicators:

1. **volume_adi (Accumulation/Distribution Index)**
   - **Calculation:** The ADI is computed by calculating the Money Flow Multiplier, which is then multiplied by the period's volume. This product is then added to the previous period's ADI value.
     - Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
     - ADI = Previous ADI + (Money Flow Multiplier * Volume)

2. **volume_obv (On-Balance Volume)**
   - **Calculation:** OBV is a cumulative total of volume. Volume is added to the indicator if closing price moves up and subtracted if closing price moves down.
     - If today's close > yesterday's close, OBV = previous OBV + today's volume.
     - If today's close < yesterday's close, OBV = previous OBV - today's volume.

3. **volume_cmf (Chaikin Money Flow)**
   - **Calculation:** CMF combines price and volume to measure the buying and selling pressure over a set period. It is the sum of Accumulation/Distribution for 20 periods divided by the sum of volume for 20 periods.
     - CMF = ∑[(Close - Low) - (High - Close)] / (High - Low) * Volume / ∑ Volume

4. **volume_fi (Force Index)**
   - **Calculation:** The Force Index uses price and volume to measure the power behind a price move. The shorter the period, the more volatile the indicator.
     - FI = Volume * (Current Close - Previous Close)

5. **volume_em (Ease of Movement)**
   - **Calculation:** EoM indicates how easily prices move. A large positive value indicates the price moved up on low volume. 
     - EoM = [(High + Low)/2 - (Previous High + Previous Low)/2] / (Volume / 10^6 / (High - Low))

6. **volume_sma_em (Simple Moving Average of Ease of Movement)**
   - **Calculation:** This is simply the SMA of the EoM values over a specified period, often 14 days.
     - SMA of EoM = Sum of EoM over N periods / N

7. **volume_vpt (Volume-Price Trend)**
   - **Calculation:** VPT is a cumulative volume-based indicator used to show the direction of the trend. 
     - VPT = Previous VPT + (Volume * (Today’s Close - Previous Close) / Previous Close)

8. **volume_vwap (Volume Weighted Average Price)**
   - **Calculation:** VWAP gives the average price a security has traded at throughout the day, based on volume and price. It is often used as a trading benchmark.
     - VWAP = ∑ (Typical Price * Volume) / ∑ Volume
     - Where Typical Price = (High + Low + Close) / 3

9. **volume_mfi (Money Flow Index)**
   - **Calculation:** MFI uses both price and volume to measure buying and selling pressure. Often called the volume-weighted RSI.
     - Raw Money Flow = Typical Price * Volume
     - Money Ratio = Positive Money Flow / Negative Money Flow
     - MFI = 100 - (100 / (1 + Money Ratio))

10. **volume_nvi (Negative Volume Index)**
    - **Calculation:** NVI focuses on days where the volume decreases from the previous day. If volume decreases, NVI is adjusted by the percentage change in price.
      - If today's volume < yesterday's volume, NVI = yesterday's NVI + (today's close - yesterday's close) / yesterday's close * yesterday's NVI.
      - If today's volume ≥ yesterday's volume, NVI is unchanged.

### Momentum Indicators:

1. **momentum_rsi (Relative Strength Index)**
   - **Calculation:** RSI measures the speed and change of price movements. It is calculated using average gains and losses over a specified period, typically 14 days.
     - RSI = 100 - (100 / (1 + (Average Gain / Average Loss)))

2. **momentum_stoch_rsi (Stochastic RSI)**
   - **Calculation:** The Stoch RSI applies the Stochastic formula to RSI values, instead of price values, to measure RSI's momentum.
     - Stoch RSI = (RSI - Lowest Low RSI) / (Highest High RSI - Lowest Low RSI)

3. **momentum_stoch_rsi_k (Stochastic RSI K)**
   - **Calculation:** Represents the current value of the Stochastic RSI. It's essentially the same as the Stoch RSI value and can sometimes refer to a smoothed version using a moving average.

4. **momentum_stoch_rsi_d (Stochastic RSI D)**
   - **Calculation:** The D line is the simple moving average of the Stochastic RSI K line. It's used to smooth out the K line for better signal interpretation.
     - Stoch RSI D = 3-day SMA of Stoch RSI K

5. **momentum_tsi (True Strength Index)**
   - **Calculation:** TSI is a momentum oscillator that blends the price momentum with its moving average. It is calculated by taking a double smoothed moving average of the price momentum.
     - TSI = (Double Smoothed Price Change / Double Smoothed Absolute Price Change) * 100

6. **momentum_uo (Ultimate Oscillator)**
   - **Calculation:** The UO combines short, medium, and long-term market trends in one indicator. It uses three different time frames and weights them into a final oscillating value.
     - UO = [(4 x SMA1) + (2 x SMA2) + SMA3] / (4 + 2 + 1)
     - Where SMA1, SMA2, and SMA3 are the average ratios of buying pressure to true range over their respective periods.

7. **momentum_stoch (Stochastic Oscillator)**
   - **Calculation:** This indicator measures the current price relative to its price range over a given time period.
     - %K = [(Current Close - Lowest Low) / (Highest High - Lowest Low)] * 100

8. **momentum_stoch_signal (Stochastic Oscillator Signal)**
   - **Calculation:** The signal line is typically a 3-day simple moving average of the Stochastic Oscillator (%K).
     - %D = 3-day SMA of %K

9. **momentum_wr (Williams %R)**
   - **Calculation:** Williams %R is similar to the Stochastic Oscillator but inverted. It measures overbought and oversold levels.
     - %R = [(Highest High - Close) / (Highest High - Lowest Low)] * -100

10. **momentum_ao (Awesome Oscillator)**
    - **Calculation:** AO is a 34-period simple moving average, subtracted from a 5-period simple moving average. These moving averages are based on the midpoint of the bars (not the closing prices).
      - AO = SMA(Midpoint, 5 periods) - SMA(Midpoint, 34 periods)
      - Where Midpoint = (High + Low) / 2

11. **momentum_roc (Rate of Change)**
   - **Calculation:** The ROC measures the percentage change between the current price and the price a certain number of periods ago.
     - ROC = [(Current Close - Close N periods ago) / Close N periods ago] * 100

12. **momentum_ppo (Percentage Price Oscillator)**
   - **Calculation:** The PPO calculates the percentage difference between two exponential moving averages (EMAs), typically the 26-period EMA and the 12-period EMA.
     - PPO = [(12-period EMA - 26-period EMA) / 26-period EMA] * 100

13. **momentum_ppo_signal (Percentage Price Oscillator Signal Line)**
   - **Calculation:** The PPO signal line is usually a 9-period EMA of the PPO itself. It is used to generate trading signals.
     - PPO Signal = 9-period EMA of PPO

14. **momentum_ppo_hist (Percentage Price Oscillator Histogram)**
   - **Calculation:** The PPO histogram represents the difference between the PPO and its signal line. It's used to identify bullish or bearish momentum.
     - PPO Histogram = PPO - PPO Signal

15. **momentum_pvo (Percentage Volume Oscillator)**
   - **Calculation:** Similar to the PPO but for volume. The PVO measures the difference between two volume EMAs as a percentage of the longer EMA, typically the 26-period EMA and the 12-period EMA.
     - PVO = [(12-period Volume EMA - 26-period Volume EMA) / 26-period Volume EMA] * 100

16. **momentum_pvo_signal (Percentage Volume Oscillator Signal Line)**
   - **Calculation:** The PVO signal line is a 9-period EMA of the PVO, used to generate trading signals based on volume.
     - PVO Signal = 9-period EMA of PVO

17. **momentum_pvo_hist (Percentage Volume Oscillator Histogram)**
   - **Calculation:** The PVO histogram is the difference between the PVO and its signal line, indicating changes in volume momentum.
     - PVO Histogram = PVO - PVO Signal

18. **momentum_kama (Kaufman's Adaptive Moving Average)**
   - **Calculation:** KAMA adjusts its sensitivity based on market volatility. It starts with an Efficiency Ratio (ER) and then applies smoothing constants to an EMA based on the ER, to calculate the moving average.
     - ER = Absolute value of [(Current Close - Close N periods ago) / Sum of N period Absolute Price Changes]
     - KAMA = Previous KAMA + SC * (Current Price - Previous KAMA)
     - Where SC (Smoothing Constant) is calculated based on the ER and specified fast/slow EMA constants.

"""
# To Implement!!
"""
### Volatility Indicators:

#### Bollinger Bands Indicators:

1. **volatility_bbm (Bollinger Bands Middle Band)**
   - **Calculation:** The middle band is typically a 20-period simple moving average (SMA) of the closing prices.
     - BBM = 20-period SMA of Close

2. **volatility_bbh (Bollinger Bands Upper Band)**
   - **Calculation:** The upper band is calculated by adding (a specified number of standard deviations, usually 2) to the middle band (SMA).
     - BBH = BBM + (2 * 20-period Standard Deviation of Close)

3. **volatility_bbl (Bollinger Bands Lower Band)**
   - **Calculation:** The lower band is calculated by subtracting (a specified number of standard deviations, usually 2) from the middle band (SMA).
     - BBL = BBM - (2 * 20-period Standard Deviation of Close)

4. **volatility_bbw (Bollinger Bands Width)**
   - **Calculation:** The bandwidth quantifies the width of the Bollinger Bands. It is calculated by subtracting the lower band from the upper band, and often expressed as a percentage of the middle band.
     - BBW = (BBH - BBL) / BBM

5. **volatility_bbp (Bollinger Bands Percentage)**
   - **Calculation:** BBP measures where the last price is relative to the bands. It is calculated by taking the difference between the close and the lower band and dividing by the difference between the upper and lower bands.
     - BBP = (Close - BBL) / (BBH - BBL)

6. **volatility_bbhi (Bollinger Bands High Indicator)**
   - **Calculation:** BBHI is a binary indicator that shows if the closing price is above the upper Bollinger Band.
     - BBHI = 1 if Close > BBH else 0

7. **volatility_bbli (Bollinger Bands Low Indicator)**
   - **Calculation:** BBLI is a binary indicator that shows if the closing price is below the lower Bollinger Band.
     - BBLI = 1 if Close < BBL else 0
      
#### Keltner Channel-Related Indicators:

8. **volatility_kcc (Keltner Channel Middle Line)**
   - **Calculation:** The middle line of Keltner Channel is a 20-period Exponential Moving Average (EMA) of the closing prices.
     - KCC = 20-period EMA of Close

9. **volatility_kch (Keltner Channel Upper Line)**
   - **Calculation:** The upper line is determined by adding the value of the Average True Range (ATR) of the past 10 periods, multiplied by a factor (commonly 2), to the middle line.
     - KCH = KCC + (2 * 10-period ATR)

10. **volatility_kcl (Keltner Channel Lower Line)**
    - **Calculation:** The lower line is calculated by subtracting the value of the 10-period ATR, multiplied by a factor (commonly 2), from the middle line.
      - KCL = KCC - (2 * 10-period ATR)

11. **volatility_kcw (Keltner Channel Width)**
   - **Calculation:** The width of the Keltner Channel, calculated as the difference between the upper and lower bands, often expressed as a percentage of the middle line.
     - KCW = (KCH - KCL) / KCC

12. **volatility_kcp (Keltner Channel Percentage)**
   - **Calculation:** Measures the closing price's position within the Keltner Channel, expressed as a percentage.
     - KCP = (Close - KCL) / (KCH - KCL)

13. **volatility_kchi (Keltner Channel High Indicator)**
   - **Calculation:** A binary indicator that signals if the close is above the upper Keltner Channel.
     - KCHI = 1 if Close > KCH else 0

14. **volatility_kcli (Keltner Channel Low Indicator)**
   - **Calculation:** A binary indicator that signals if the close is below the lower Keltner Channel.
     - KCLI = 1 if Close < KCL else 0

#### Donchian Channel-Related Indicators:

15. **volatility_dcl (Donchian Channel Lower)**
   - **Calculation:** The lowest low of the past N periods.
     - DCL = Lowest Low over last N periods

16. **volatility_dch (Donchian Channel High)**
   - **Calculation:** The highest high of the past N periods.
     - DCH = Highest High over last N periods

17. **volatility_dcm (Donchian Channel Middle)**
   - **Calculation:** The average of the Donchian Channel High and Low.
     - DCM = (DCH + DCL) / 2

18. **volatility_dcw (Donchian Channel Width)**
   - **Calculation:** The difference between the upper and lower Donchian Channel lines.
     - DCW = DCH - DCL

19. **volatility_dcp (Donchian Channel Percentage)**
   - **Calculation:** Measures where the current closing price is within the Donchian Channel, as a percentage.
     - DCP = (Close - DCL) / (DCH - DCL)

#### Other Volatility Indicators:

20. **volatility_atr (Average True Range)**
    - **Calculation:** ATR measures market volatility by decomposing the entire range of an asset for the period. It takes the maximum of the following three values: current high minus current low, absolute value of current high minus previous close, and absolute value of current low minus previous close. The ATR is typically a moving average of these true ranges.
      - ATR = Moving average (typically 14-period) of the True Ranges

21. **volatility_ui (Ulcer Index)**
    - **Calculation:** The Ulcer Index measures downside risk over a given period. It is calculated by squaring the percentage drawdowns (from the highest point) and then computing the square root of the average of these squares over the period.
      - UI = sqrt(Sum of squared percentage drawdowns / N)

Exploring the calculation details for trend indicators using OHLC (Open, High, Low, Close) data provides insights into market direction and momentum. Here’s how each listed trend indicator is typically calculated:

### Trend Indicators:

1. **trend_macd (Moving Average Convergence Divergence)**
   - **Calculation:** MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA.
     - MACD = 12-period EMA - 26-period EMA

2. **trend_macd_signal**
   - **Calculation:** The MACD Signal line is a 9-period EMA of the MACD line itself, serving as a trigger for buy and sell signals.
     - MACD Signal = 9-period EMA of MACD

3. **trend_macd_diff (MACD Histogram)**
   - **Calculation:** The MACD Histogram represents the difference between the MACD and its signal line. It's used to anticipate MACD crossovers.
     - MACD Histogram = MACD - MACD Signal

4. **trend_sma_fast (Fast Simple Moving Average)**
   - **Calculation:** A simple moving average (SMA) calculated over a shorter period, often used to identify quick, recent trends.
     - Fast SMA = Sum of Close / N (where N is the period, e.g., 10 days)

5. **trend_sma_slow (Slow Simple Moving Average)**
   - **Calculation:** A simple moving average (SMA) calculated over a longer period than the Fast SMA, often used to identify longer-term market trends.
     - Slow SMA = Sum of Close / N (where N is the period, e.g., 50 days)

6. **trend_ema_fast (Fast Exponential Moving Average)**
   - **Calculation:** Similar to the Fast SMA but gives more weight to recent prices, making it more responsive to new information.
     - Fast EMA = (Close - Previous EMA) * (2 / (N + 1)) + Previous EMA

7. **trend_ema_slow (Slow Exponential Moving Average)**
   - **Calculation:** Similar to the Slow SMA but gives more weight to recent prices, suitable for identifying long-term trends with a greater reaction to recent price changes.
     - Slow EMA = (Close - Previous EMA) * (2 / (N + 1)) + Previous EMA

8. **trend_vortex_ind_pos (Positive Vortex Indicator)**
   - **Calculation:** Measures upward trend movement by comparing the current high to the previous low, often smoothed over a period.
     - VI+ = (Sum of positive VMIs over N periods / Sum of TRs over N periods)

9. **trend_vortex_ind_neg (Negative Vortex Indicator)**
   - **Calculation:** Measures downward trend movement by comparing the current low to the previous high, often smoothed over a period.
     - VI- = (Sum of negative VMIs over N periods / Sum of TRs over N periods)

10. **trend_vortex_ind_diff**
    - **Calculation:** The difference between the Positive and Negative Vortex Indicators, indicating the dominance of a trend direction.
      - Vortex Indicator Difference = VI+ - VI-

11. **trend_trix (Triple Exponential Average)**
   - **Calculation:** TRIX smooths the minor fluctuations in the market and highlights the underlying trend. It's the rate of change of a triple exponentially smoothed moving average.
     - TRIX = 1-period percent change of a 15-period Triple EMA of Close

12. **trend_mass_index**
   - **Calculation:** The Mass Index detects trend reversals by measuring the narrowing and widening between the high and low prices. It involves a 9-period EMA of the high-low range divided by a 9-period EMA of the 9-period EMA of the high-low range.
     - MI = Sum of 9-period EMA of (High - Low) / 9-period EMA of the EMA over 25 periods

13. **trend_dpo (Detrended Price Oscillator)**
   - **Calculation:** DPO removes long-term trends from price data, allowing a focus on short-term patterns and cycles by comparing past prices to a shifted moving average.
     - DPO = Price N/2 + 1 periods ago - N-period SMA of Price

14. **trend_kst (Know Sure Thing)**
   - **Calculation:** KST is a momentum oscillator that combines multiple smoothed rates of change of the closing price. It sums four different time frames of the ROC, each weighted.
     - KST = ROC1 * 1 + ROC2 * 2 + ROC3 * 3 + ROC4 * 4

15. **trend_kst_sig (KST Signal Line)**
   - **Calculation:** The KST signal line is typically a 9-period SMA of the KST indicator, used to generate buy and sell signals.
     - KST Signal = 9-period SMA of KST

16. **trend_kst_diff**
   - **Calculation:** The difference between the KST and its signal line. This can indicate momentum shifts.
     - KST Difference = KST - KST Signal

17. **trend_ichimoku_conv (Ichimoku Conversion Line)**
   - **Calculation:** The conversion line is calculated as the midpoint between the 9-period high and the 9-period low.
     - Ichimoku Conversion Line = (9-period High + 9-period Low) / 2

18. **trend_ichimoku_base (Ichimoku Base Line)**
   - **Calculation:** The base line is the midpoint between the 26-period high and the 26-period low.
     - Ichimoku Base Line = (26-period High + 26-period Low) / 2

19. **trend_ichimoku_a (Ichimoku Leading Span A)**
   - **Calculation:** Leading Span A is the average of the conversion line and the base line. This line is plotted 26 periods into the future.
     - Ichimoku A = (Conversion Line + Base Line) / 2

20. **trend_ichimoku_b (Ichimoku Leading Span B)**
    - **Calculation:** Leading Span B is calculated as the midpoint between the 52-period high and the 52-period low, plotted 26 periods into the future.
      - Ichimoku B = (52-period High + 52-period Low) / 2

31. **trend_stc (Schaff Trend Cycle)**
   - **Calculation:** The STC is a combination of the MACD and stochastic oscillators, designed to improve accuracy in identifying trend changes. It involves calculating the MACD for price, then applying a stochastic oscillator formula to the MACD.
     - First, calculate the MACD line as you normally would.
     - Next, apply the Stochastic formula to the MACD values: STC = %K stochastic of MACD.

32. **trend_adx (Average Directional Index)**
   - **Calculation:** The ADX quantifies trend strength regardless of direction. It is derived from the directional movement indicators (DMI) (+DI and -DI) and smoothed by the average true range.
     - ADX = 14-period exponential moving average of the absolute value of (+DI - -DI) / (+DI + -DI).

33. **trend_adx_pos (+DI, Positive Directional Indicator)**
   - **Calculation:** +DI measures the upward trend strength. It's calculated as the 14-period exponential moving average of +DM (positive directional movement) divided by the average true range.
     - +DI = 14-period EMA of +DM / ATR.

34. **trend_adx_neg (-DI, Negative Directional Indicator)**
   - **Calculation:** -DI measures the downward trend strength. It's similar to +DI but uses -DM (negative directional movement).
     - -DI = 14-period EMA of -DM / ATR.

35. **trend_cci (Commodity Channel Index)**
   - **Calculation:** The CCI compares the current price to the average price over a specific time period. It's designed to identify cyclical trends.
     - CCI = (Typical Price - 20-period SMA of TP) / (0.015 * Mean Deviation),
     - where TP (Typical Price) = (High + Low + Close) / 3.

36. **trend_visual_ichimoku_a**
   - **Explanation:** This typically refers to a visualization aspect of the Ichimoku Cloud (Ichimoku Kinko Hyo), representing the Leading Span A (Senkou Span A), which is the midpoint of the Conversion Line and the Base Line plotted 26 periods ahead.

37. **trend_visual_ichimoku_b**
   - **Explanation:** Similar to `trend_visual_ichimoku_a`, this refers to the visualization of the Leading Span B (Senkou Span B) of the Ichimoku Cloud, plotted 26 periods into the future. It's the midpoint of the 52-period high and low.

38. **trend_aroon_up**
   - **Calculation:** The Aroon Up indicator measures the time since the last 25-period high, expressed as a percentage of the total period.
     - Aroon Up = [(25 - Periods Since 25-period High) / 25] * 100

39. **trend_aroon_down**
   - **Calculation:** The Aroon Down indicator measures the time since the last 25-period low, expressed as a percentage.
     - Aroon Down = [(25 - Periods Since 25-period Low) / 25] * 100

40. **trend_aroon_ind (Aroon Indicator)**
    - **Calculation:** The Aroon Indicator is often represented by the Aroon Up and Aroon Down lines. However, some calculations might represent it as the difference between Aroon Up and Aroon Down to provide a consolidated view of the trend.
      - Aroon Indicator = Aroon Up - Aroon Down

#### Parabolic SAR Components:

41. **trend_psar_up**
   - **Calculation:** This value is calculated during a downtrend when the trend reverses to uptrend. The Parabolic SAR for the next period is calculated using the previous period's SAR, the extreme point (EP, which is the highest high during the current uptrend), and the acceleration factor (AF). For an uptrend, the SAR is calculated as:
     - `PSAR = Previous PSAR + AF * (EP - Previous PSAR)`
   - The initial PSAR when a trend turns up is the lowest low of the previous downtrend.

42. **trend_psar_down**
   - **Calculation:** Similar to `trend_psar_up`, but for when the trend is downward. The calculation involves the previous period's SAR, the lowest low (EP) during the current downtrend, and the acceleration factor (AF). The formula for a downtrend is:
     - `PSAR = Previous PSAR - AF * (Previous PSAR - EP)`
   - The initial PSAR when a trend turns down is the highest high of the previous uptrend.

43. **trend_psar_up_indicator**
   - **Calculation:** This is a binary indicator that signals when the Parabolic SAR switches to below the price, indicating a potential uptrend. If the current PSAR value is below the current price, it signals an uptrend.
     - `PSAR Up Indicator = 1 if Current PSAR < Current Price else 0`

44. **trend_psar_down_indicator**
   - **Calculation:** This is a binary indicator that signals when the Parabolic SAR switches to above the price, indicating a potential downtrend. If the current PSAR value is above the current price, it signals a downtrend.
     - `PSAR Down Indicator = 1 if Current PSAR > Current Price else 0`

### 1. others_dr (Daily Return)
    - **Definition:** Daily Return measures the percentage change in the price of a financial instrument from one day to the next. It is a simple way to gauge the day-to-day fluctuation in the price.
    - **Calculation:** The formula for calculating the Daily Return (`others_dr`) is:
    - `Daily Return = [(Current Day's Close - Previous Day's Close) / Previous Day's Close] * 100`
    - This calculation provides the return for a specific day as a percentage change from the previous day's closing price.

### 2. others_dlr (Daily Log Return)
    - **Definition:** Daily Log Return is another way to measure the price change of a financial instrument, using the natural logarithm. Log returns are useful for mathematical analysis, particularly because they are time additive.
    - **Calculation:** The formula for calculating the Daily Log Return (`others_dlr`) is:
    - `Daily Log Return = ln(Current Day's Close / Previous Day's Close) * 100`
    - Where `ln` denotes the natural logarithm. Log returns are especially handy in portfolio analysis and are time additive, which means you can sum log returns over time to get a cumulative return.

### 3. others_cr (Cumulative Return)
    - **Definition:** Cumulative Return measures the total change in the price of a financial instrument over a specified period. It represents the aggregate effect of price movements on the initial investment over time.
    - **Calculation:** The formula for calculating the Cumulative Return (`others_cr`) when starting from 0 and expressed as a percentage might look like this:
    - `Cumulative Return = [(Current Price / Initial Price) - 1] * 100`
    - This calculation gives the total percentage return on the investment from the beginning of the period to the current date.
"""

class TechnicalIndicators:

  @staticmethod
  def assert_timeseries(df: pd.DataFrame, add_col: list = []):
    """
    Assert the pandas DataFrame input structure

    Parameters:
    - df (pd.DataFrame): DataFrame with a datetime index and columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume'] + add_col
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

  @staticmethod
  def assert_price(price: Union[float, int]):
    if not ((isinstance(price, (int, float)) and price > 0) or (price is None)):
        raise ValueError("price must be strictly positive float or int, or it should be None")

  @staticmethod
  def assert_span(span: Union[float, int, list, np.ndarray]) -> Iterable:
    """
    Assert the span which has to be an Iterable composed of float or int, or it should be a float or int

    Parameters:
    - span (float, int, List[float,int], numpy.ndarray): window size for moving averages (each span >= 1)

    Returns:
    - Itearable[float,int >= 1]: span
    """
    # Check if span is already an iterable (but not a string, as strings are also iterable)
    if isinstance(span, Iterable) and not isinstance(span, str):
        # Convert numpy arrays to lists for consistent handling
        if isinstance(span, np.ndarray):
            span = span.tolist()
        # Check all elements are int or float and greater than 1
        if not all(isinstance(item, (int, float)) and item >= 1 for item in span):
            raise ValueError("All elements in span must be int or float and greater than 1")
    # If span is a single int or float, check it's greater than 1 and make it a list
    elif isinstance(span, (int, float)):
        if span <= 1:
            raise ValueError("span must be greater than 1")
        span = [span]  # Convert to list for consistency
    else:
        raise ValueError("span must be an int, float, or an iterable of int/float")

    return span

  @staticmethod
  def assert_float(para: float, para_name: str):
    if not isinstance(para, float):
        raise ValueError(f"{para_name} must be float")

  @staticmethod
  def append_from_parallelize(df: pd.DataFrame, processes: list, n_jobs: int = 5):
    with ProcessPoolExecutor(max_workers=np.minimum(n_jobs, len(processes))) as executor:
      futures = [executor.submit(p, df) for p in processes]
      results = [f.result() for f in futures]

    # Append results to df
    for res in results:
        df.loc[res.index, res.columns] = res
    
    return df

  @staticmethod
  def get_sub_series(x: pd.Series, q1: Union[float, int], q2: Union[float, int]):
    return x[(x > q1) & (x < q2)]
  
  def get_sub_series_up(self, x: pd.Series, q1: Union[float, int], q2: Union[float, int]):
    x_  = self.get_sub_series(x, q1, q2)

    return x_[x_ > 0]
  
  def get_sub_series_low(self, x: pd.Series, q1: Union[float, int], q2: Union[float, int]):
    x_  = self.get_sub_series(x, q1, q2)

    return x_[x_ < 0]
  
  def get_slope_up(self, x: pd.Series, q1: Union[float, int], q2: Union[float, int]):
    x_  = self.get_sub_series(x, q1, q2)

    mean    = x_.mean()
    std     = x_.std()

    x_      = np.sort(x_) - mean
    ccdf    = (np.cumsum(np.ones(len(x_))).astype(float)/len(x_))[::-1]

    mask        = np.where(x_ > 3*std)[0]
    slope_pos   = 0.
    if len(mask) > 2:
        slope_pos, _, _, _, _ = linregress(np.log(x_[mask]), np.log(ccdf[mask]))

    return slope_pos

  def get_slope_low(self, x: pd.Series, q1: Union[float, int], q2: Union[float, int]):
    x_  = self.get_sub_series(x, q1, q2)

    mean    = x_.mean()
    std     = x_.std()

    x_      = np.sort(x_) - mean
    ccdf    = (np.cumsum(np.ones(len(x_))).astype(float)/len(x_))[::-1]

    mask        = np.where(x_ < -3*std)[0]
    slope_neg   = 0.
    if len(mask) > 2:
        slope_neg, _, _, _, _ = linregress(np.log(-1.*x_[mask]), np.log(ccdf[mask]))

    return slope_neg

  def get_previous_data(self, df: pd.DataFrame, df_last: Optional[pd.DataFrame], cols: list):
     # If we have previous data, append them
    if not df_last is None:
      self.assert_timeseries(df_last, cols)
      df  = pd.concat([df_last.iloc[-1:][cols], df[cols]])
    
    return df

  def get_prices(self, df: pd.DataFrame, last_price: Optional[Union[float, int]]) -> pd.DataFrame:
    """
    Calculate Price ...
    
    Parameters:
    - df (pd.DataFrame): DataFrame with a datetime index and columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    
    Returns:
    - pd.DataFrame: DataFrame indexed similarly to the input with columns for each technical indicator.
    
    Raises:
    - ValueError: If the input DataFrame does not meet the requirements.
    """

    # Validate input DataFrame
    self.assert_timeseries(df)
    # Validate last_price
    self.assert_price(last_price)

    prices            = pd.DataFrame(columns=["Typical", "Diff", "Midd", "Money", "PC"], index = df.index)
    prices["Typical"] = df[["Close", "High", "Low"]].sum(axis=1)/3.   # Typical price = (Close + High + Low)/3
    prices["Diff"]    = df["High"] - df["Low"]                        # Max difference = High - Low
    prices["Midd"]    = (df["High"] + df["Low"])/2.                   # Middle Price = (High + Low)/2
    prices["Money"]   = prices["Typical"]*df["Volume"]                    # Money Traded = Typical Price x Volume
    prices["PC"]      = df["Close"].shift(1)                          # Previous Close Price
    # Append last price if any
    if not last_price is None:
        prices.loc[prices.index[0], "PC"] = last_price

    return prices

  def get_price_flow(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Price Flow ...
    
    Parameters:
    - df (pd.DataFrame): DataFrame with a datetime index and columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    
    Returns:
    - pd.DataFrame: DataFrame indexed similarly to the input with columns for each technical indicator.
    
    Raises:
    - ValueError: If the input DataFrame does not meet the requirements.
    """

    # Validate input DataFrame
    self.assert_timeseries(df, add_col=["PC", "Money"])

    evolves                     = pd.DataFrame(columns=["Evolve", "Return", "Positive Flow", "Negative Flow"], index = df.index)
    evolves["Evolve"]           = df["Close"] - df["PC"]                          # Close Price Evolution t = Close_t - Close_{t-1}
    evolves["Return"]           = df["Close"]/df["PC"] - 1.                       # Period return t = Close_t/Close_{t-1} - 1
    evolves['Positive Flow']    = np.where(evolves["Return"] > 0, df['Money'], 0.)     # Positive Money Flow t = Money Flow t where Close t > Close t-1
    evolves['Negative Flow']    = np.where(evolves["Return"] < 0, df['Money'], 0.)     # Negative Money Flow t =  Money Flow t where Close t < Close t-1
    evolves['Positive Evolve']  = np.where(evolves["Return"] > 0, evolves['Evolve'], 0.)    # Positive Evolve Flow t = Money Flow t where Close t > Close t-1
    evolves['Negative Evolve']  = np.where(evolves["Return"] < 0, -1*evolves['Evolve'], 0.) # Negative Evolve Flow t =  Money Flow t where Close t < Close t-1

    return evolves
  
  def get_price_pressure(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Price Pressure ...
    
    Parameters:
    - df (pd.DataFrame): DataFrame with a datetime index and columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    
    Returns:
    - pd.DataFrame: DataFrame indexed similarly to the input with columns for each technical indicator.
    
    Raises:
    - ValueError: If the input DataFrame does not meet the requirements.
    """

    # Validate input DataFrame
    self.assert_timeseries(df, add_col=["PC"])

    pressure            = pd.DataFrame(columns=["minLPC", "maxHPC", "BP", "TR"], index = df.index)
    pressure["minLPC"]  = df[["Low", "PC"]].min(axis=1)   # Minimum between Previous Price and Current Low
    pressure["maxHPC"]  = df[["High", "PC"]].min(axis=1)  # Maximum between Previous Price and Current High
    pressure["BP"]      = df["Close"] - pressure["minLPC"]      # Buying Pressure
    pressure["TR"]      = pressure["maxHPC"] - pressure["minLPC"]     # True Range

    return pressure[["BP", "TR"]]

  def get_diffAdi(self, df: pd.DataFrame) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=["Money", "Diff"])

    # Difference of ADI = ((Close - Low) - (High - Close)) x Money / (High - Low)
    diff_adi            = pd.DataFrame(columns=["DiffADI"], index = df.index)
    diff_adi["DiffADI"] = (2.*df["Close"] - df["High"] - df["Low"])*df["Money"]/df["Diff"]                  
    
    return diff_adi

  def get_force_index(self, df: pd.DataFrame) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=["Money", "Return"])

    # Regularized Force Index = Money x Period Return
    force_index       = pd.DataFrame(columns=["FI"], index = df.index)
    force_index["FI"] = df["Money"]*df["Return"]

    return force_index

  def get_ease_of_movement(self, df: pd.DataFrame, df_last: Optional[pd.DataFrame]) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=["Money", "Return"])

    # Measure Previous Midd = (High + Low) / 2
    midd_shift = df["Midd"].shift(1)
    # If we have previous data, append the last Midd point
    if not df_last is None:
        self.assert_timeseries(df_last)
        midd_shift.loc[midd_shift.index[0]]  = (df_last["High"].iloc[-1] + df_last["Low"].iloc[-1])/2.

    # Ease of Movement t = ((High_t + Low_t) - (High_{t-1} + Low_{t-1})) / Money_t / (High_t - Low_t) / 2
    ease_of_movement        = pd.DataFrame(columns=["EOM"], index = df.index)
    ease_of_movement["EOM"] = (df["Midd"] - midd_shift)/df["Money"]/df["Diff"]

    return ease_of_movement

  def get_chaikin_money_flow(self, df: pd.DataFrame, df_last: Optional[pd.DataFrame], span: int = 10) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=["DiffADI", "Money"])
    
    # Initialize EWM
    df_ = self.get_previous_data(df[["DiffADI", "Money"]], df_last, cols = [f"DiffADI-{span}", f"Money-{span}"])

    # Chaikin Money Flow = Average over Past Difference of ADI / Average over Past Volume in Money
    cmf = pd.DataFrame(columns=[f"{c}-{span}" for c in ["CMF", "DiffADI", "Money"]], index=df.index)
    
    cmf[[f"DiffADI-{span}", f"Money-{span}"]] = df_.ewm(span=span, axis=0).mean().iloc[1:]
    
    cmf[f"CMF-{span}"]  = cmf[f"DiffADI-{span}"]/cmf[f"Money-{span}"]

    return cmf

  def get_ewm_ease_of_movement(self, df: pd.DataFrame, df_last: Optional[pd.DataFrame], span: int = 10) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=["EOM"])
    
    # Initialize EWM
    df_ = self.get_previous_data(df[["EOM"]], df_last, cols = [f"EOM-{span}"])

    # EWM Average of Ease of Movement
    eom = pd.DataFrame(columns=[f"EOM-{span}"], index=df.index)
    eom[f"EOM-{span}"]  = df_["EOM"].ewm(span=span).mean().iloc[1:]

    return eom

  def get_money_flow(self, df: pd.DataFrame, df_last: Optional[pd.DataFrame], span: int = 10) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=['Positive Flow', 'Negative Flow'])
    
    # Initialize EWM
    df_ = self.get_previous_data(df[['Positive Flow', 'Negative Flow']], df_last, cols = [f'Positive Flow-{span}', f'Negative Flow-{span}'])

    money_flow  = pd.DataFrame(columns=[f"MFI-{span}", f'Positive Flow-{span}', f'Negative Flow-{span}'])
    money_flow[[f'Positive Flow-{span}', f'Negative Flow-{span}']]  = df_.ewm(span=span, axis=0).mean().iloc[1:]
    # Money Flow Ratio = Average Positive Money Flow / Average Negative Money Flow
    mfr = money_flow[f'Positive Flow-{span}']/money_flow[f'Negative Flow-{span}']
    # Money Flow Index = 1 - 1/(1 + Money Flow Ratio)
    money_flow[f"MFI-{span}"] = 1. - 1./(1.+mfr)
    
    return money_flow

  def get_buying_pressure_strength(self, df: pd.DataFrame, df_last: Optional[pd.DataFrame], span: int = 10) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=['BP', 'TR'])
    
    # Initialize EWM
    df_ = self.get_previous_data(df[['BP', 'TR']], df_last, cols = [f'BP-{span}', f'TR-{span}'])

    # Strength between Buying Pressure and True Range
    strength  = pd.DataFrame(columns=[f"BP/TR-{span}", f'BP-{span}', f'TR-{span}'])
    strength[f"BP-{span}"]     = df_["BP"].ewm(span=span).mean().iloc[1:]                                     
    strength[f"TR-{span}"]     = df_["TR"].ewm(span=span).mean().iloc[1:]
    strength[f"BP/TR-{span}"]  = strength[f"BP-{span}"] / strength[f"TR-{span}"]

    return strength

  def get_ultimate_oscillator(self, df: pd.DataFrame, span: int = 10) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=[f"BP/TR-{i*span}" for i in [1, 2, 4]])

    # Ultimate Oscillator = Average over different time scale of Buying Pressure / True Range
    uo  = pd.DataFrame(columns=[f"UO-{span}"])
    uo[f"UO-{span}"]  = (df[[f"BP/TR-{i*span}" for i in [1, 2, 4]]]*np.array([1, 2, 4])).sum(axis=1) / 7

    return uo

  def get_stochastic_oscillator(self, df: pd.DataFrame, df_last: Optional[pd.DataFrame], span: int = 10) -> pd.DataFrame:
     # Validate input DataFrame
    self.assert_timeseries(df)
    
    # Initialize EWM
    df_ = self.get_previous_data(df[['High', 'Low']], df_last, cols = [f"maxHigh-{span}", f"minLow-{span}"])

    # Stochastic Oscillator = (Close - Lowest Low) / (Highest High - Lowes Low)
    so  = pd.DataFrame(columns=[f"SO-{span}", f"maxHigh-{span}", f"minLow-{span}"])
    so[f"minLow-{span}"]  = df_["Low"].rolling(window=span, min_periods=1).min().iloc[1:]   # Lowest Low
    so[f"maxHigh-{span}"] = df["High"].rolling(window=span, min_periods=1).max().iloc[1:]                  # Highest High
    so[f"SO-{span}"]      = (df["Close"] - so[f"minLow-{span}"])/(so[f"maxHigh-{span}"] - so[f"minLow-{span}"])

    return so

  def get_relative_strength(self, df: pd.DataFrame, df_last: Optional[pd.DataFrame], span: int = 10) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=['Positive Evolve', 'Negative Evolve'])
    
    # Initialize EWM
    df_ = self.get_previous_data(df[['Positive Evolve', 'Negative Evolve']], df_last, cols = [f"Positive Evolve-{span}", f"Negative Evolve-{span}"])

    rsi = pd.DataFrame(columns=[f"RSI-{span}", f"Positive Evolve-{span}", f"Negative Evolve-{span}"])
    rsi[[f"Positive Evolve-{span}", f"Negative Evolve-{span}"]] = df_[["Positive Evolve", "Negative Evolve"]].ewm(span=span).mean()
    rs  = rsi[f"Positive Evolve-{span}"]/rsi[f"Negative Evolve-{span}"] # Relative Strenght = Average Positive Evolution / Average Negative Evolution
    rsi[f"RSI-{span}"]  = 1. - 1./(1. + rs)                             # Relative Strength Index = 1 - 1/(1 + RS)

    return rsi
  
  def get_true_strength(self, df: pd.DataFrame, df_last: Optional[pd.DataFrame], span: int = 10) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=['Evolve'])
    
    # Get min & max
    ts  = df[["Evolve"]]
    ts["EvolveAbs"] = np.abs(ts["Evolve"])
    if not df_last is None:
        self.assert_timeseries(df_last)
        ts  = pd.concat([df_last.iloc[-1:][[f"Evolve-{span}", f"EvolveAbs-{span}"]], ts], axis=0)

    tsi = pd.DataFrame(columns=[f"TSI-{span}", f"Evolve-{span}", f"EvolveAbs-{span}"])
    tsi[[f"Evolve-{span}", f"EvolveAbs-{span}"]]  = ts[["Evolve", "EvolveAbs"]].ewm(span=span).mean().loc[df.index[0]:]
    
    tsi[f"TSI-{span}"]  = tsi[f"Evolve-{span}"]/tsi[f"EvolveAbs-{span}"]

    return tsi
  
  def get_stochastic_index(self, df: pd.DataFrame, index: str, df_last: Optional[pd.DataFrame], span: int = 10) -> pd.DataFrame:
    # Validate input DataFrame
    self.assert_timeseries(df, add_col=[f"{index}-{span}"])
    
    # Get min & max
    max_min = df[[f"{index}-{span}", f"{index}-{span}"]]
    max_min.columns = [f"max{index}-{span}", f"min{index}-{span}"]
    if not df_last is None:
        self.assert_timeseries(df_last)
        max_min = pd.concat([df_last.iloc[-1:][[f"max{index}-{span}", f"min{index}-{span}"]], max_min], axis=0)
    
    # Stochastic Measure of an Index
    si  = pd.DataFrame(columns=[f"Stoch{index}-{span}", f"max{index}-{span}", f"min{index}-{span}"])
    si[f"max{index}-{span}"]    = max_min[f"max{index}-{span}"].rolling(window=span, min_periods=1).max().loc[df.index[0]:]
    si[f"min{index}-{span}"]    = max_min[f"min{index}-{span}"].rolling(window=span, min_periods=1).min().loc[df.index[0]:]
    si[f"Stoch{index}-{span}"]  = (df[f"{index}-{span}"] - si[f"min{index}-{span}"])/(si[f"max{index}-{span}"] - si[f"min{index}-{span}"])
  
    return si

  def get_quantile(self, df: pd.Series, q: float, span: int = 10) -> pd.DataFrame:
    data  = pd.DataFrame(columns=[f"{int(q*100)}%-Q-{span}"])
    data[f"{int(q*100)}%-Q-{span}"] = df["Return"].rolling(window=span).quantile(q)

    return data
  
  def get_skweness(self, df: pd.Series, span: int = 10) -> pd.DataFrame:
    data = pd.DataFrame(columns=[f"skw-{span}"])
    data[f"skw-{span}"]  = df["Return"].rolling(window=span).skew()

    return data
  
  def get_kurtosis(self, df: pd.Series, span: int = 10) -> pd.DataFrame:
    data = pd.DataFrame(columns=[f"kts-{span}"])
    data[f"kts-{span}"]  = df["Return"].rolling(window=span).kurt()

    return data

  def get_mean(self, df: pd.Series, span: int = 10) -> pd.DataFrame:
    data = pd.DataFrame(columns=[f"mean-{span}"])
    data[f"mean-{span}"] = df["Return"].rolling(window=span).apply(lambda x: self.get_sub_series(x, df[f"5%-Q-{span}"][x.index[-1]], df[f"95%-Q-{span}"][x.index[-1]]).mean())

    return data
  
  def get_std(self, df: pd.Series, span: int = 10) -> pd.DataFrame:
    data = pd.DataFrame(columns=[f"std-{span}"])
    data[f"std-{span}"] = df["Return"].rolling(window=span).apply(lambda x: self.get_sub_series(x, df[f"5%-Q-{span}"][x.index[-1]], df[f"95%-Q-{span}"][x.index[-1]]).std())

    return data
  
  def get_std_up(self, df: pd.Series, span: int = 10) -> pd.DataFrame:
    data = pd.DataFrame(columns=[f"std(+)-{span}"])
    data[f"std(+)-{span}"] = df["Return"].rolling(window=span).apply(lambda x: self.get_sub_series_up(x, df[f"5%-Q-{span}"][x.index[-1]], df[f"95%-Q-{span}"][x.index[-1]]).std())

    return data

  def get_std_low(self, df: pd.Series, span: int = 10) -> pd.DataFrame:
    data = pd.DataFrame(columns=[f"std(-)-{span}"])
    data[f"std(-)-{span}"] = df["Return"].rolling(window=span).apply(lambda x: self.get_sub_series_low(x, df[f"5%-Q-{span}"][x.index[-1]], df[f"95%-Q-{span}"][x.index[-1]]).std())
    
    return data

  def get_all_slope_up(self, df: pd.Series, span: int = 10) -> pd.DataFrame:
    data = pd.DataFrame(columns=[f"slope(+)-{span}"])
    data[f"slope(+)-{span}"] = df["Return"].rolling(window=span).apply(lambda x: self.get_slope_up(x, df[f"5%-Q-{span}"][x.index[-1]], df[f"95%-Q-{span}"][x.index[-1]]))

    return data

  def get_all_slope_low(self, df: pd.Series, span: int = 10) -> pd.DataFrame:
    data = pd.DataFrame(columns=[f"slope(-)-{span}"])
    data[f"slope(-)-{span}"]  = df["Return"].rolling(window=span).apply(lambda x: self.get_slope_low(x, df[f"5%-Q-{span}"][x.index[-1]], df[f"95%-Q-{span}"][x.index[-1]]))

    return data

  def measure_draw(self, price: pd.Series) -> pd.DataFrame:
    
    if price.iloc[0] > price.iloc[1]:
      i   = price[price > price.iloc[0]]
      if len(i) >0:
        i   = i.index[0]
      else:
        i   = -1
        i   = price[:i].argmin()
    elif price.iloc[0] < price.iloc[1]:
      i   = price[price < price.iloc[0]]
      if len(i) >0:
        i   = i.index[0]
      else:
        i   = -1
        i   = price[:i].argmax()
    else:
      i   = price.index[0]

    return pd.DataFrame({"cum log": [price[i]/price.iloc[0] - 1.], "time start": [price.index[0]], "time end": [i]})

  def get_draw(self, price: pd.Series) -> pd.DataFrame:
    price_last  = price.shift(1)
    price_next  = price.shift(-1)

    I       = price.index[(((price > price_last) & (price < price_next)) | ((price < price_last) & (price > price_next)))]
    bg_wl   = [self.measure_draw(price[i:]) for i in I]
    if len(bg_wl) > 0:
      bg_wl   = pd.concat(bg_wl, axis=0, ignore_index=True)
    else:
      bg_wl   = pd.DataFrame({"cum log": [0], "time start": [np.nan], "time end": [np.nan]})

    return bg_wl

  def get(
          self,
          df: pd.DataFrame,
          df_last: Optional[pd.DataFrame] = None,
          span: Union[float, int, list, np.ndarray] = 10,
          stat_span: Union[float, int, list, np.ndarray] = 20,
          n_jobs: int = 5,
          ) -> pd.DataFrame:
    """
    Calculate various technical indicators on a pandas DataFrame containing OHLCV data.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with a datetime index and columns ['Open', 'High', 'Low', 'Close', 'Volume'].
    
    Returns:
    - pd.DataFrame: DataFrame indexed similarly to the input with columns for each technical indicator.
    
    Raises:
    - ValueError: If the input DataFrame does not meet the requirements.
    """

    self.assert_timeseries(df)
    if not df_last is None:
        self.assert_timeseries(df_last)

    span        = self.assert_span(span)
    stat_span   = self.assert_span(stat_span)

    # Initialize 

    # Get:  Typical price = (Close + High + Low)/3
    #       Max difference = High - Low, Middle Price = (High + Low)/2
    #       Middle Price = (High + Low)/2
    #       Money Traded = Typical Price x Volume
    #       Previous Close Price
    df  = pd.concat([df, self.get_prices(df, last_price= None if df_last is None else df_last["Close"].iloc[-1])], axis = 1)

    # With get_price_flow, Get:
    #       Close Price Evolution t = Close_t - Close_{t-1}
    #       Period return t = Close_t/Close_{t-1} - 1
    #       Positive Money Flow t = Money Flow t where Close t > Close t-1
    #       Negative Money Flow t =  Money Flow t where Close t < Close t-1
    # With get_price_pressure, Get:
    #       Buying Pressure = Close_t - min(Close_{t-1}, Low_t)
    #       True Range = max(Close_{t-1}, High_t) - Close_t
    df  = self.append_from_parallelize(df, processes=[self.get_price_flow, self.get_price_pressure], n_jobs = np.minimum(n_jobs, 2))

    # Money Based Point Indicators
    df  = self.append_from_parallelize(df,
                                        processes=[self.get_diffAdi,
                                                  self.get_force_index,
                                                  partial(self.get_ease_of_movement, df_last=df_last)],
                                        n_jobs=np.minimum(n_jobs, 3))

    df  = self.append_from_parallelize(df,
                                        processes=[self.get_diffAdi,
                                                  self.get_force_index,
                                                  partial(self.get_ease_of_movement, df_last=df_last)],
                                        n_jobs=np.minimum(n_jobs, 3))
    
    # Money Based Average Indicators
    money_based_processes = [partial(self.get_chaikin_money_flow, df_last=df_last, span=s) for s in span]
    money_based_processes += [partial(self.get_ewm_ease_of_movement, df_last=df_last, span=s) for s in span]
    money_based_processes += [partial(self.get_money_flow, df_last=df_last, span=s) for s in span]
    df  = self.append_from_parallelize(df, processes=money_based_processes, n_jobs=n_jobs)

    # Money Based Indicators (3 + 3*len(span))
    money_cols  = ["DiffADI", "FI", "EOM"] + [f"{e}-{s}"for e in ["CMF", "EOM", "MFI"] for s in span]

    # Momentum Based Average Indicators
    strength_processes  = [partial(self.get_buying_pressure_strength, df_last=df_last, span=s) for s in list(set([i*s for i in [1, 2, 4] for s in span]))]
    df  = self.append_from_parallelize(df, processes=strength_processes, n_jobs=n_jobs)
    
    momentum_based_processes  = [partial(self.get_ultimate_oscillator, span=s) for s in span]
    momentum_based_processes  += [partial(self.get_stochastic_oscillator, df_last=df_last, span=s) for s in span]
    momentum_based_processes  += [partial(self.get_relative_strength, df_last=df_last, span=s) for s in span]
    momentum_based_processes  += [partial(self.get_true_strength, df_last=df_last, span=s) for s in span]
    df  = self.append_from_parallelize(df, processes=momentum_based_processes, n_jobs=n_jobs)

    stochastic_processes  = [partial(self.get_stochastic_index, index="RSI", df_last=df_last, span=s) for s in span]
    stochastic_processes  += [partial(self.get_stochastic_index, index="TSI", df_last=df_last, span=s) for s in span]
    df  = self.append_from_parallelize(df, processes=stochastic_processes, n_jobs=n_jobs)

    # Momentum Based Indicators ()
    momentum_cols   = [f"{e}-{s}"for e in ["UO", "SO", "RSI", "StochRSI", "TSI", "StochTSI"] for s in span]

    # Statistics
    ret, t0 = df[["Return"]], df.index[0]
    if not df_last is None:
       self.assert_timeseries(df_last, add_col=["Return"])
       ret  = pd.concat([df_last[["Return"]].iloc[-np.max(stat_span)], ret], axis=0)
    
    stat_process  = [partial(self.get_quantile, q=q, span=s) for q in [0.05, 0.25, 0.5, 0.75, 0.95] for s in stat_span]
    stat_process  += [partial(self.get_skweness, span=s) for s in stat_span]
    stat_process  += [partial(self.get_kurtosis, span=s) for s in stat_span]
    ret = self.append_from_parallelize(ret, processes=stat_process, n_jobs=n_jobs)

    stat_process  = [partial(self.get_mean, span=s) for s in stat_span]
    stat_process  += [partial(self.get_std, span=s) for s in stat_span]
    stat_process  += [partial(self.get_std_up, span=s) for s in stat_span]
    stat_process  += [partial(self.get_std_low, span=s) for s in stat_span]
    stat_process  += [partial(self.get_all_slope_up, span=s) for s in stat_span]
    stat_process  += [partial(self.get_all_slope_low, span=s) for s in stat_span]
    ret = self.append_from_parallelize(ret, processes=stat_process, n_jobs=n_jobs)
    
    statistics_cols = [f"{e}-{s}"for e in ["mean", "std", "std(+)", "std(-)", "slope(+)", "slope(-)", "skw", "kts"] + [f"{i}%-Q" for i in [5, 25, 50, 75, 95]] for s in stat_span]

    df[statistics_cols] = ret[statistics_cols].loc[t0:]

    return df[money_cols + momentum_cols + statistics_cols], df.iloc[-np.max(stat_span):]