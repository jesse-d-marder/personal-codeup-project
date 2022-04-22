import pandas as pd
import acquire
import numpy as np

def prepare_crypto_data(results):
    """ Takes in a dictionary with keys as the symbols of different cryptocurrencies and values as a dataframe of open, high, low, close, and volume prices. Returns dictionary with the data prepared:
        -Sets time to datetime index
        -Truncates dataframes so all start at same date
        """
    
    first_dates = []
    for key in results.keys():
        results[key] = results[key].set_index(pd.to_datetime(results[key]['time']))
        first_dates.append(results[key].index.min())
        
    starting_date = max(first_dates)

    print(f'Max first date is {starting_date}, starting all dataframes at this day')

    for key in results.keys():
        df = results[key]
        df = df.loc[starting_date:]

        results[key] = df
        
        # correct a low value for april 15 2017 that was due to an exchange error
        if key == "BTC_USD":
            # minute_data = acquire.acquire_crypto_data(acquire.get_full_product_info(['BTC-USD']),datetime(2017, 4, 15, 0,0,0), datetime(2017, 4, 15, 23, 59, 0), 60)
        
            # minute_data['BTC-USD']=minute_data['BTC-USD'].loc[(minute_data['BTC-USD'].index<'2017-04-15 23:00:00' )|(minute_data['BTC-USD'].index>'2017-04-15 23:50:00')]
            
            # To save time on subsequent lows this value is set based on prior exploration
            # results[key].loc['2017-04-15','low'] = minute_data['BTC-USD'].low.min()
            print("Corrected btc low data for 2017-04-15")
            results[key].loc['2017-04-15','low'] = 568.120000

    return results

def add_features(df):
    """ Adds target and additional features to dataframe. Returns dataframe with additional features """
    
    ###### TARGETS ######
    # forward 1 day log returns
    df["log_ret_fwd"] = np.log(df.close) - np.log(df.close.shift(-1))
    # forward standard returns
    df["ret_fwd"] = df.close.shift(-1) - df.close
    # forward pct change
    df["fwd_pct_chg"] = df.close.pct_change(-1)
    # binary positive vs negative next day return
    df["next_close_positive"] = df.ret_fwd>0
    
    ###### FEATURES ######
    # Pct change from yesterday
    df["pct_chg"] = df.close.pct_change()

    # Calculate lagged log returns 
    for i in range(1,8):
        df[f'log_ret_lag_{i}'] = np.log(df.close) - np.log(df.close.shift(i))
        
    # Volatility:
    # relative price range: RR
        
    df["RR"] = 2*(df.high.shift(1)-df.low.shift(1))/(df.high.shift(1)+df.low.shift(1))
    
    # range volatility estimator of Parkinson: sigma  - lags 1-7
    for i in range(1,8):
        df[f'sigma_lag_{i}'] = ((np.log(df.high.shift(i)/df.low.shift(i))**2)/(4*np.log(2)))**0.5
    
    # Day of the week shown to be significant from literature
    df["day_name"] = df.index.day_name()
    
    # Dummy variable for day name
    df = pd.get_dummies(df, columns=['day_name'])
    
    # Drop any remaining nulls (created due to lagged values)
    df = df.dropna()
    
    return df