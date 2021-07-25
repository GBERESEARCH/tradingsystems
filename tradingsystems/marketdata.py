import norgatedata
import numpy as np
import os
import pandas as pd
import requests
from yahoofinancials import YahooFinancials

class Markets():
    
    @classmethod
    def create_base_data(
            cls, ticker=None, ccy_1=None, ccy_2=None, start_date=None, 
            end_date=None, source=None, asset_type=None, api_key=None):
        """
        Create DataFrame of OHLC prices from NorgateData or Yahoo Finance

        Parameters
        ----------
        ticker : Str, optional
            Underlying to return. The default '$SPX'.
        ccy_1 : Str, optional
            Primary currency of pair to return. The default 'GBP'.
        ccy_2 : Str, optional
            Secondary currency of pair to return. The default 'USD'.
        start_date : Str, optional
            Date to begin backtest. Format is 'YYYY-MM-DD'. 
        end_date : Str, optional
            Date to end backtest. Format is 'YYYY-MM-DD'. 
        source : Str, optional
            The data source to use, either 'norgate' or 'yahoo'. The default 
            is 'norgate'.
        api_key : Str    
            AlphaVantage API key. If not provided will look for 
            'ALPHAVANTAGE_API_KEY' in the environment variables.    

        Returns
        -------
        df : DataFrame
            Returns OHLC DataFrame.

        """
        
        # Extract data from Norgate
        if source == 'norgate': 
            timeseriesformat = 'pandas-dataframe' 
            df = norgatedata.price_timeseries(
                symbol=ticker, start_date=start_date, end_date=end_date, 
                format=timeseriesformat)
            
            return df
        
        # Extract data from Yahoo Finance
        elif source == 'yahoo':
            df = cls._return_yahoo_data(
                ticker=ticker, start_date=start_date, end_date=end_date)
        
            return df
        
        # Extract data from AlphaVantage
        elif source == 'alpha':
            df = cls._return_alphavantage_data(
                ccy_1=ccy_1, ccy_2=ccy_2, ticker=ticker, asset_type=asset_type, 
                start_date=start_date, end_date=end_date)
            
            return df
                
        # Otherwise return error message
        else:
            print('Select a data source from yahoo, norgate or alpha')
    
    
    @staticmethod
    def _return_yahoo_data(ticker=None, start_date=None, end_date=None):
        """
        Create DataFrame of historic prices for specified ticker using Yahoo 
        Finance as the source.

        Parameters
        ----------
        ticker : Int
            Stock to be returned in the form of Reuters RIC code as a 
            string. 
        start_date : Str, optional
            Date to begin backtest. Format is 'YYYY-MM-DD'. 
        end_date : Str, optional
            Date to end backtest. Format is 'YYYY-MM-DD'.
        freq : Int
            Frequency of data - set to 'daily'.

        Returns
        -------
        df : DataFrame
            DataFrame of historic prices for given ticker.

        """
        
        # Initialise data class
        yahoo_financials = YahooFinancials(ticker)
        freq='daily'
        
        # Extract historic prices
        df = yahoo_financials.get_historical_price_data(
            start_date, end_date, freq)
        
        # Reformat columns
        df = pd.DataFrame(df[ticker]['prices']).drop(['date'], axis=1) \
                .rename(columns={'formatted_date':'Date',
                                 'open': 'Open',
                                 'high': 'High',
                                 'low': 'Low',
                                 'close': 'Close',
                                 'volume': 'Volume'}) \
                .loc[:, ['Date','Open','High','Low','Close','Volume']] \
                .set_index('Date')
        
        # Set Index to Datetime
        df.index = pd.to_datetime(df.index)
        
        return df


    @staticmethod
    def _return_alphavantage_data(
            ccy_1=None, ccy_2=None, ticker=None, asset_type=None, 
            start_date=None, end_date=None, api_key=None):
        """
        Create DataFrame of historic prices for specified ticker using 
        AlphaVantage as the source.  

        Parameters
        ----------
        ticker : Str
            Underlying to return. The default '$SPX'.
        ccy_1 : Str
            Primary currency of pair to return. The default 'GBP'.
        ccy_2 : Str
            Secondary currency of pair to return. The default 'USD'.
        asset_type : Str
            The alphavantage asset class type. The default is 'fx'.
        start_date : Str, optional
            Date to begin backtest. Format is 'YYYY-MM-DD'. 
        end_date : Str, optional
            Date to end backtest. Format is 'YYYY-MM-DD'.
        api_key : Str    
            AlphaVantage API key. If not provided will look for 
            'ALPHAVANTAGE_API_KEY' in the environment variables.

        Returns
        -------
        df : DataFrame
            DataFrame of historic prices for given ticker.

        """
        
        # Set API key
        if api_key is None:
            api_key = os.getenv('ALPHAVANTAGE_API_KEY')

        # Set url to extract prices from
        base_url = 'https://www.alphavantage.co/query?'
        
        # FX pair
        if asset_type == 'fx':
            params = {'function': 'FX_DAILY',
                      'from_symbol': ccy_1,
                      'to_symbol': ccy_2, 
                      'outputsize':'full',
                      'apikey': api_key}
        
            response = requests.get(base_url, params=params)
            response_dict = response.json()
            
            _, header = response.json()
            
            #Convert to pandas dataframe
            df = pd.DataFrame.from_dict(response_dict[header], orient='index')
            
            #Clean up column names
            df_cols = [i.split(' ')[1].title() for i in df.columns]
            df.columns = df_cols
            
            # Set datatype to float
            df = df.astype(float)
            
            
        # Cryptocurrency        
        elif asset_type == 'crypto':
            params = {'function': 'DIGITAL_CURRENCY_DAILY',
                      'symbol': ccy_1,
                      'market': ccy_2,
                      'apikey': api_key}
            
            response = requests.get(base_url, params=params)
            response_dict = response.json()
            
            _, header = response.json()
            
            #Convert to pandas dataframe
            df = pd.DataFrame.from_dict(response_dict[header], orient='index')
            
            # Select the USD OHLC columns
            df = df[
                [df.columns[1], df.columns[3], df.columns[5], df.columns[7]]]
            
            # Set column names
            df.columns = ['Open', 'High', 'Low', 'Close']
                        
            # Set datatype to float
            df = df.astype(float)
            
            
        # Equity Single stock or Index
        elif asset_type == 'equity':
            params = {'function': 'TIME_SERIES_DAILY_ADJUSTED',
                      'symbol': ticker,
                      'outputsize':'full',
                      'apikey': api_key}

            response = requests.get(base_url, params=params)
            response_dict = response.json()
            
            _, header = response.json()
            
            #Convert to pandas dataframe
            df = pd.DataFrame.from_dict(response_dict[header], orient='index')
           
            #Clean up column names
            df_cols = [i.split(' ')[1].title() for i in df.columns]
            df.columns = df_cols

            # Set datatype to float
            df = df.astype(float)
            
            # Calculate stock split multiplier
            df['split_mult'] = np.array([1.0]*len(df))
            for row in range(1, len(df)):
                if df['Split'][row] == 1:
                    df['split_mult'][row] = df['split_mult'][row-1]
                else:
                    df['split_mult'][row] = (df['split_mult'][row-1] 
                                             * df['Split'][row])

            # Adjust OHLC prices for splits
            df['O'] = np.round(df['Open'] / df['split_mult'], 2) 
            df['H'] = np.round(df['High'] / df['split_mult'], 2)
            df['L'] = np.round(df['Low'] / df['split_mult'], 2)
            df['C'] = np.round(df['Close'] / df['split_mult'], 2)
            
            # Select only OHLC columns
            df = df[['O', 'H', 'L', 'C']]
                        
            # Set column names
            df.columns = ['Open', 'High', 'Low', 'Close']


        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid asset type")
        
        
        # Set Index to Datetime
        df.index = pd.to_datetime(df.index)

        # Sort data in ascending order
        df = df[::-1]
        
        # If a start date has been provided
        if start_date is not None:
            
            # Set the start variable to this, converting to datetime format 
            start = pd.to_datetime(start_date)
        
        # If no start date is provided
        else:
            # Set the start variable to the first row in the DataFrame
            start = df.index[0]
        
        # If an end date has been provided    
        if end_date is not None:
            
            # Set the end variable to this, converting to datetime format
            end = pd.to_datetime(end_date)
        
        # If no end date is provided    
        else:
            
            # Set the end variable to the last row in the DataFrame
            end = df.index[-1]
            
        # Trim data to specified dates            
        df = df.loc[start:end]
        
        return df
    
        
    @staticmethod
    def _norgate_name_dict():
        """
        Create a dictionary of the long names of the Norgate tickers.

        Returns
        -------
        norgate_name_dict : Dict
            Dictionary lookup of Norgate tickers to long names.

        """
        
        # Get list of the available databases
        alldatabasenames = norgatedata.databases()
        
        # Create empty dictionary to store names
        norgate_name_dict = {}
        
        # For each of the available databases
        for database in alldatabasenames:
            
            # Extract dictionary of dictionaries, one for each ticker in the 
            # database 
            databasecontents = norgatedata.database(database)
            
            # For each ticker in the database
            for dicto in databasecontents:
                
                # Set the key-value pair of the new dictionary to the ticker 
                # and long name respectively
                key = dicto['symbol']
                value = dicto['securityname']
                
                # Whether to include backadjusted / regular continuous futures
                if database == 'Continuous Futures':
                    
                    #if '_CCB' in key:
                    norgate_name_dict[key] = value
                
                # Don't include the individual futures contracts
                elif database == 'Futures':
                    pass
                
                # Store the values in the dictionary
                else:
                    norgate_name_dict[key] = value
        
        return norgate_name_dict    

   
