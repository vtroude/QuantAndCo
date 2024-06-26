import pandas as pd
  
#######################################################################################################################

def prepare_data(price_df, indicators_df):
    if price_df.index.name != 'timestamp':
        if 'timestamp' not in price_df.columns:
            raise ValueError('"timestamp" column missing in price_df')
        else:
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            price_df.set_index('timestamp', inplace=True)

    if indicators_df.index.name != 'timestamp':
        if 'timestamp' not in indicators_df.columns:
            raise ValueError('"timestamp" column missing in indicators_df')
        else:
            indicators_df['timestamp'] = pd.to_datetime(indicators_df['timestamp'])
            indicators_df.set_index('timestamp', inplace=True)
    
    if len(price_df) != len(indicators_df):
        print(f'Datasets have different lengths. price_df has {len(price_df)} rows while indicators_df has {len(indicators_df)} rows.')

    return pd.merge(price_df, indicators_df, right_index=True, left_index=True)