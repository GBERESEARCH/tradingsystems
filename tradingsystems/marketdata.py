"""
Market Data functions
"""
import os
import norgatedata
import numpy as np
import pandas as pd
import requests
from yahoofinancials import YahooFinancials

class Markets():
    """
    Methods for collecting data from Norgate Data, Yahoo Finance and
    AlphaVantage and extracting the long names of the Norgate Data tickers.

    """
    @classmethod
    def create_base_data(
            cls, ticker=None, source=None, params=None, bench_flag=None):
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
        prices : DataFrame
            Returns OHLC DataFrame.

        """

        # If a valid source has been provided
        #try:
        # Extract data from Norgate
        if source == 'norgate':
            prices, params = cls._return_norgate_data(
                ticker=ticker, params=params)

            if bench_flag is False:
                params = cls._contract_data(
                    ticker=ticker, prices=prices, params=params)

        # Extract data from Yahoo Finance
        elif source == 'yahoo':
            prices = cls._return_yahoo_data(
                ticker=ticker,
                start_date=params['start_date'],
                end_date=params['end_date'])

            params['contract_point_value'] = 1

        # Extract data from AlphaVantage
        elif source == 'alpha':
            prices = cls._return_alphavantage_data(
                ticker=ticker, params=params)

            params['contract_point_value'] = 1

        else:
            raise ValueError(
                'Select a data source from yahoo, norgate or alpha')

        return prices, params

        # Otherwise return error message
        #except ValueError:
        #    return 'Select a data source from yahoo, norgate or alpha'


    @staticmethod
    def _return_norgate_data(ticker, params):

        timeseriesformat = 'pandas-dataframe'
        prices = norgatedata.price_timeseries(
            symbol=ticker,
            start_date=params['start_date'],
            end_date=params['end_date'],
            format=timeseriesformat)

        return prices, params


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
        prices : DataFrame
            DataFrame of historic prices for given ticker.

        """

        # Initialise data class
        yahoo_financials = YahooFinancials(ticker)
        freq='daily'

        # Extract historic prices
        prices = yahoo_financials.get_historical_price_data(
            start_date, end_date, freq)

        # Reformat columns
        prices = pd.DataFrame(
            prices[ticker]['prices']).drop(['date'], axis=1) \
                .rename(columns={'formatted_date':'Date',
                                 'open': 'Open',
                                 'high': 'High',
                                 'low': 'Low',
                                 'close': 'Close',
                                 'volume': 'Volume'}) \
                .loc[:, ['Date','Open','High','Low','Close','Volume']] \
                .set_index('Date')

        # Set Index to Datetime
        prices.index = pd.to_datetime(prices.index)

        return prices


    @classmethod
    def _return_alphavantage_data(
            cls, ticker=None, params=None):
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
        prices : DataFrame
            DataFrame of historic prices for given ticker.

        """

        # Set API key
        if params['api_key'] == '':
            params['api_key'] = os.getenv('ALPHAVANTAGE_API_KEY')

        # FX pair
        if params['asset_type'] == 'fx':
            prices = cls._alphavantage_fx(
                ccy_1=params['ccy_1'],
                ccy_2=params['ccy_2'],
                api_key=params['api_key'])

        # Cryptocurrency
        elif params['asset_type'] == 'crypto':
            prices = cls._alphavantage_crypto(
                ccy_1=params['ccy_1'],
                ccy_2=params['ccy_2'],
                api_key=params['api_key'])

        # Equity Single stock or Index
        elif params['asset_type'] == 'equity':
            prices = cls._alphavantage_equity(
                ticker=ticker, api_key=params['api_key'])

        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid asset type")

        # Set Index to Datetime
        prices.index = pd.to_datetime(prices.index)

        # Sort data in ascending order
        prices = prices[::-1]

        # If a start date has been provided
        if params['start_date'] is not None:

            # Set the start variable to this, converting to datetime format
            start = pd.to_datetime(params['start_date'])

        # If no start date is provided
        else:
            # Set the start variable to the first row in the DataFrame
            start = prices.index[0]

        # If an end date has been provided
        if params['end_date'] is not None:

            # Set the end variable to this, converting to datetime format
            end = pd.to_datetime(params['end_date'])

        # If no end date is provided
        else:

            # Set the end variable to the last row in the DataFrame
            end = prices.index[-1]

        # Trim data to specified dates
        prices = prices.loc[start:end]

        return prices


    @staticmethod
    def _alphavantage_fx(ccy_1, ccy_2, api_key):
        """
        Create DataFrame of historic prices for an fx pair using
        AlphaVantage as the source.

        Parameters
        ----------
        ccy_1 : Str
            Primary currency of pair to return. The default 'GBP'.
        ccy_2 : Str
            Secondary currency of pair to return. The default 'USD'.
        api_key : Str
            AlphaVantage API key. If not provided will look for
            'ALPHAVANTAGE_API_KEY' in the environment variables.

        Returns
        -------
        prices :  DataFrame
            DataFrame of historic prices for given ticker.

        """

        # Set url to extract prices from
        base_url = 'https://www.alphavantage.co/query?'

        # Set fx params
        params = {'function': 'FX_DAILY',
                  'from_symbol': ccy_1,
                  'to_symbol': ccy_2,
                  'outputsize':'full',
                  'apikey': api_key}

        response = requests.get(base_url, params=params)
        response_dict = response.json()

        _, header = response.json()

        #Convert to pandas dataframe
        prices = pd.DataFrame.from_dict(
            response_dict[header], orient='index')

        #Clean up column names
        df_cols = [i.split(' ')[1].title() for i in prices.columns]
        prices.columns = df_cols

        # Set datatype to float
        prices = prices.astype(float)

        return prices


    @staticmethod
    def _alphavantage_crypto(ccy_1, ccy_2, api_key):
        """
        Create DataFrame of historic prices for a cryptocurrency pair using
        AlphaVantage as the source.

        Parameters
        ----------
        ccy_1 : Str
            Primary currency of pair to return.
        ccy_2 : Str
            Secondary currency of pair to return.
        api_key : Str
            AlphaVantage API key. If not provided will look for
            'ALPHAVANTAGE_API_KEY' in the environment variables.

        Returns
        -------
        prices :  DataFrame
            DataFrame of historic prices for given ticker.

        """
        # Set url to extract prices from
        base_url = 'https://www.alphavantage.co/query?'

        # Set crypto params
        params = {'function': 'DIGITAL_CURRENCY_DAILY',
                  'symbol': ccy_1,
                  'market': ccy_2,
                  'apikey': api_key}

        response = requests.get(base_url, params=params)
        response_dict = response.json()

        _, header = response.json()

        #Convert to pandas dataframe
        prices = pd.DataFrame.from_dict(
            response_dict[header], orient='index')

        # Select the USD OHLC columns
        prices = prices[
            [prices.columns[1], prices.columns[3], prices.columns[5],
             prices.columns[7]]]

        # Set column names
        prices.columns = ['Open', 'High', 'Low', 'Close']

        # Set datatype to float
        prices = prices.astype(float)

        return prices


    @staticmethod
    def _alphavantage_equity(ticker, api_key):
        """
        Create DataFrame of historic prices for an equity ticker using
        AlphaVantage as the source.

        Parameters
        ----------
        ticker : Str
            Underlying to return. The default '$SPX'.
        api_key : Str
            AlphaVantage API key. If not provided will look for
            'ALPHAVANTAGE_API_KEY' in the environment variables.

        Returns
        -------
        prices :  DataFrame
            DataFrame of historic prices for given ticker.

        """
        # Set url to extract prices from
        base_url = 'https://www.alphavantage.co/query?'

        # Set equity params
        params = {'function': 'TIME_SERIES_DAILY_ADJUSTED',
                  'symbol': ticker,
                  'outputsize':'full',
                  'apikey': api_key}

        response = requests.get(base_url, params=params)
        response_dict = response.json()

        _, header = response.json()

        #Convert to pandas dataframe
        prices = pd.DataFrame.from_dict(
            response_dict[header], orient='index')

        #Clean up column names
        df_cols = [i.split(' ')[1].title() for i in prices.columns]
        prices.columns = df_cols

        # Set datatype to float
        prices = prices.astype(float)

        # Calculate stock split multiplier
        prices['split_mult'] = np.array([1.0]*len(prices))
        for row in range(1, len(prices)):
            if prices['Split'][row] == 1:
                prices['split_mult'][row] = prices['split_mult'][row-1]
            else:
                prices['split_mult'][row] = (prices['split_mult'][row-1]
                                         * prices['Split'][row])

        # Adjust OHLC prices for splits
        prices['O'] = np.round(prices['Open'] / prices['split_mult'], 2)
        prices['H'] = np.round(prices['High'] / prices['split_mult'], 2)
        prices['L'] = np.round(prices['Low'] / prices['split_mult'], 2)
        prices['C'] = np.round(prices['Close'] / prices['split_mult'], 2)

        # Select only OHLC columns
        prices = prices[['O', 'H', 'L', 'C']]

        # Set column names
        prices.columns = ['Open', 'High', 'Low', 'Close']

        return prices


    @staticmethod
    def _contract_data(ticker, prices, params):

        params['front_ticker'] = (
            ticker[1:]
            +'-'
            +str(prices['Delivery Month'][-1])[:4]
            +params['contract_months'][
                str(prices['Delivery Month'][-1])[4:6]])

        params['per_contract_margin'] = norgatedata.margin(
            params['front_ticker'])
        params['contract_point_value'] = norgatedata.point_value(
            params['front_ticker'])

        return params


    @staticmethod
    def norgate_name_dict():
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
