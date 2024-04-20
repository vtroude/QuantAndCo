# User Manual of DataPipeline

## Fetch Data from External Database

There exists different source to gather data:

* Binance to get **crypto** pairs Data (__binance_data.py__)
* OANDA to get **forex** pairs Data (__forex_oanda_data.py__)

To get from any of the sources you have to use the function __get_time_series__ in __get_fetch.py__

## Get & Save Data in CSV files + Technical Indicators

The function to get & save data in the directory __Data__ are all in __make_data.py__, where there is two main functions:

* __get_and_save_timeseries__: fetch the data from an external database and save it in __Data__
* __get_and_save_indicators__: get the OHLC + Volume data saved in __Data__, compute the technical indicators and save them

## Load Local Data

You can access local data by using __pandas.read_csv__ combined to the function __make_filename__ from __make_data.py__, by doing the following

```
python
ohlc        = pd.read_csv(make_filename(market, symbol, interval, start_time, end_time, "OHLC"), index_col=0)
ohlc.index  = pd.to_datetime(ohlc.index)
```