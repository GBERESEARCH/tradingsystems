# Imports
import norgatedata
import pandas as pd
import numpy as np
import math
import systems_params as sp
import datetime as dt
from technicalmethods.methods import Indicators
from operator import itemgetter
from pandas.tseries.offsets import BDay
from yahoofinancials import YahooFinancials
from decimal import Decimal


class Data():
    
    def __init__(self):
        
        # Import dictionary of default parameters 
        self.df_dict = sp.system_params_dict
   
    
    def _refresh_params_default(self, **kwargs):
        """
        Set parameters in various functions to the default values.
        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.
        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if 
            no data provided
        """
        
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Set it to the default value and assign to the object 
                # and to input dictionary
                v = self.df_dict['df_'+str(k)]
                self.__dict__[k] = v
                kwargs[k] = v 
            
            # If the value has been provided as an input, assign this 
            # to the object
            else:
                self.__dict__[k] = v
                      
        return kwargs    
   
    
    def test_strategy(
            self, ticker=None, start_date=None, end_date=None, lookback=None, 
            short_ma=None, medium_ma=None, long_ma=None, ma_1=None, ma_2=None,
            ma_3=None, ma_4=None,position_size=None, source=None, 
            slippage=None, commission=None, strategy=None):
        """
        Run a backtest over the chosen strategy

        Parameters
        ----------
        ticker : Str, optional
            Underlying to test. The default '$SPX'.
        start_date : Str, optional
            Date to begin backtest. Format is YYYY-MM-DD. The default is 500 
            business days prior (circa 2 years).
        end_date : Str, optional
            Date to end backtest. Format is YYYY-MM-DD. The default is the 
            last business day.
        lookback : Int, optional
            Number of business days to use for the backtest. The default is 500 
            business days (circa 2 years).
        short_ma : Int, optional
            The fastest of the 3 moving averages. The default is 4 periods.
        medium_ma : Int, optional
            The middle of the 3 moving averages. The default is 9 periods.
        long_ma : Int, optional
            The slowest of the 3 moving averages. The default is 18 periods.
        position_size : Int, optional
            The number of units to trade. The default is 100.
        source : Str, optional
            The data source to use, either 'norgate' or 'yahoo'. The default 
            is 'norgate'.
        slippage : Float, optional
            The amount of slippage to apply to traded prices. The default is 
            $0.05 per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.

        Returns
        -------
        Results
            Prints out performance data for the strategy.

        """
                
        # If data is not supplied as an input, take default values 
        (ticker, lookback, short_ma, medium_ma, long_ma, ma_1, ma_2, ma_3, 
         ma_4, position_size, source, slippage, commission,
         strategy) = itemgetter(
             'ticker', 'lookback', 'short_ma', 'medium_ma', 'long_ma', 'ma_1', 
             'ma_2', 'ma_3', 'ma_4', 'position_size', 'source', 'slippage', 
             'commission', 'strategy')(self._refresh_params_default(
                 ticker=ticker, lookback=lookback, short_ma=short_ma, 
                 medium_ma=medium_ma, long_ma=long_ma, ma_1=ma_1, ma_2=ma_2, 
                 ma_3=ma_3, ma_4=ma_4, position_size=position_size, 
                 source=source, slippage=slippage, commission=commission, 
                 strategy=strategy))
        
        # Set the start and end dates if not provided
        self._date_set(
            start_date=start_date, end_date=end_date, lookback=lookback)
        
        # Create DataFrame of OHLC prices from NorgateData or Yahoo Finance
        df = self.create_base_data(
            ticker=ticker, start_date=self.start_date, end_date=self.end_date, 
            source=source)

        df, strategy_label = self._strategy_selection(
            short_ma=short_ma, medium_ma=medium_ma, long_ma=long_ma, ma_1=ma_1, 
            ma_2=ma_2, ma_3=ma_3, ma_4=ma_4, df=df, 
            position_size=position_size, strategy=strategy)
        
        # Calculate the trades and pnl for the strategy
        self.df = self._profit_data(df=df, position_size=position_size, 
                                   slippage=slippage, commission=commission)
        
        # Create dictionary of performance data and print out results
        self._output_results(df=self.df, ticker=ticker, 
                             strategy_label=strategy_label)

        return self
        

    def _strategy_selection(self, **kwargs):
        """
        Create label and price signal for chosen strategy

        Parameters
        ----------
        **kwargs : Various
            The input parameters necessary for the chosen strategy.

        Raises
        ------
        ValueError
            If no correct strategy is chosen.

        Returns
        -------
        df : DataFrame
            Price data and trading signals.
        strategy_label : Str
            The longname of the strategy.

        """
        
        if kwargs['strategy'] == '3MA':
            """
            Triple moving average strategy which is long if 
            short_ma > medium_ma > long_ma, short if 
            short_ma < medium_ma < long_ma and flat otherwise
            """
            
            # Set the strategy label
            strategy_label = str('Triple MA : '
                                 +str(kwargs['short_ma'])
                                 +'-'
                                 +str(kwargs['medium_ma'])
                                 +'-'
                                 +str(kwargs['long_ma']))
            
            # Update the DataFrame with the Triple moving average signal
            df = self._triple_ma_signal(
                df=kwargs['df'], short_ma=kwargs['short_ma'], 
                medium_ma=kwargs['medium_ma'], long_ma=kwargs['long_ma'], 
                position_size=kwargs['position_size'])
            
        elif kwargs['strategy'] == '4MA':
            """
            Quad moving average strategy which is long if ma_1 > ma_2 
            and ma_3 > ma_4, short if ma_1 < ma_2 and ma_3 < ma_4
            and flat otherwise
            """    
            
            # Set the strategy label based on the 4 moving averages         
            strategy_label = str('Double MA Cross : '
                                 +str(kwargs['ma_1'])
                                 +'-'
                                 +str(kwargs['ma_2'])
                                 +' '
                                 +str(kwargs['ma_3'])
                                 +'-'
                                 +str(kwargs['ma_4']))
            
            # Update the DataFrame with the Quad moving average signal
            df = self._quad_ma_signal(
                df=kwargs['df'], ma_1=kwargs['ma_1'], ma_2=kwargs['ma_2'], 
                ma_3=kwargs['ma_3'], ma_4=kwargs['ma_4'], 
                position_size=kwargs['position_size']) 
            
        else:
            raise ValueError("Please enter a vilid strategy")
                
        
        return df, strategy_label
        

    def test_strategy_3MA(
            self, ticker=None, start_date=None, end_date=None, lookback=None, 
            short_ma=None, medium_ma=None, long_ma=None, position_size=None, 
            source=None, slippage=None, commission=None):
        """
        Run a backtest over the triple moving average strategy which is long 
        if short_ma > medium_ma > long_ma, short if short_ma < medium_ma <
        long_ma and flat otherwise

        Parameters
        ----------
        ticker : Str, optional
            Underlying to test. The default '$SPX'.
        start_date : Str, optional
            Date to begin backtest. Format is YYYY-MM-DD. The default is 500 
            business days prior (circa 2 years).
        end_date : Str, optional
            Date to end backtest. Format is YYYY-MM-DD. The default is the 
            last business day.
        lookback : Int, optional
            Number of business days to use for the backtest. The default is 500 
            business days (circa 2 years).
        short_ma : Int, optional
            The fastest of the 3 moving averages. The default is 4 periods.
        medium_ma : Int, optional
            The middle of the 3 moving averages. The default is 9 periods.
        long_ma : Int, optional
            The slowest of the 3 moving averages. The default is 18 periods.
        position_size : Int, optional
            The number of units to trade. The default is 100.
        source : Str, optional
            The data source to use, either 'norgate' or 'yahoo'. The default 
            is 'norgate'.
        slippage : Float, optional
            The amount of slippage to apply to traded prices. The default is 
            $0.05 per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.

        Returns
        -------
        Results
            Prints out performance data for the strategy.

        """
        
        # If data is not supplied as an input, take default values 
        (ticker, lookback, short_ma, medium_ma, long_ma, 
         position_size, source, slippage, commission) = itemgetter(
             'ticker', 'lookback', 'short_ma', 'medium_ma', 'long_ma', 
             'position_size', 'source', 'slippage', 'commission')(
                 self._refresh_params_default(
                     ticker=ticker, lookback=lookback, short_ma=short_ma, 
                     medium_ma=medium_ma, long_ma=long_ma, 
                     position_size=position_size, source=source, 
                     slippage=slippage, commission=commission))
        
        # Set the strategy label based on the 3 moving averages         
        strategy_label = str('Triple MA : '+str(short_ma)+'-'+str(medium_ma)+
                       '-'+str(long_ma))
        
        # Set the start and end dates if not provided
        self._date_set(
            start_date=start_date, end_date=end_date, lookback=lookback)
        
        # Create DataFrame of OHLC prices from NorgateData or Yahoo Finance
        self.df = self.create_base_data(
            ticker=ticker, start_date=self.start_date, end_date=self.end_date, 
            source=source)
        
        # Update the DataFrame with the Triple moving average signal
        self.df = self._triple_ma_signal(
            self.df, short_ma=short_ma, medium_ma=medium_ma, long_ma=long_ma, 
            position_size=position_size)
        
        # Calculate the trades and pnl for the strategy
        self.df = self._profit_data(self.df, position_size=position_size, 
                                   slippage=slippage, commission=commission)
        
        # Create dictionary of performance data and print out results
        self._output_results(df=self.df, ticker=ticker, 
                             strategy_label=strategy_label)

        return self
    
    
    def test_strategy_4MA(
            self, ticker=None, start_date=None, end_date=None, lookback=None, 
            ma_1=None, ma_2=None, ma_3=None, ma_4=None, position_size=None, 
            source=None, slippage=None, commission=None):
        """
        Run a backtest over the quad moving average strategy which is long 
        if ma_1 > ma_2 and ma_3 > ma_4, short if ma_1 < ma_2 and ma_3 < ma_4
        and flat otherwise

        Parameters
        ----------
        ticker : Str, optional
            Underlying to test. The default '$SPX'.
        start_date : Str, optional
            Date to begin backtest. Format is YYYY-MM-DD. The default is 500 
            business days prior (circa 2 years).
        end_date : Str, optional
            Date to end backtest. Format is YYYY-MM-DD. The default is the 
            last business day.
        lookback : Int, optional
            Number of business days to use for the backtest. The default is 500 
            business days (circa 2 years).
        ma_1 : Int, optional
            The fastest of the 4 moving averages. The default is 5 periods.
        ma_2 : Int, optional
            The 2nd fastest of the 4 moving averages. The default is 12 
            periods.
        ma_3 : Int, optional
            The second slowest of the 4 moving averages. The default is 20 
            periods.
        ma_4 : Int, optional
            The slowest of the 4 moving averages. The default is 40 periods.
        position_size : Int, optional
            The number of units to trade. The default is 100.
        source : Str, optional
            The data source to use, either 'norgate' or 'yahoo'. The default 
            is 'norgate'.
        slippage : Float, optional
            The amount of slippage to apply to traded prices. The default is 
            $0.05 per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.

        Returns
        -------
        Results
            Prints out performance data for the strategy.

        """
        
        # If data is not supplied as an input, take default values 
        (ticker, lookback, ma_1, ma_2, ma_3, ma_4, position_size, 
         source, slippage, commission) = itemgetter(
             'ticker', 'lookback', 'ma_1', 'ma_2', 'ma_3', 'ma_4', 
             'position_size', 'source', 'slippage', 'commission')(
                 self._refresh_params_default(
                     ticker=ticker, lookback=lookback, ma_1=ma_1, ma_2=ma_2, 
                     ma_3=ma_3, ma_4=ma_4, position_size=position_size, 
                     source=source, slippage=slippage, commission=commission))
        
        # Set the strategy label based on the 4 moving averages         
        strategy_label = str('Double MA Cross : '+str(ma_1)+'-'+str(ma_2)+
                       ' '+str(ma_3)+'-'+str(ma_4))

        # Set the start and end dates if not provided
        self._date_set(
            start_date=start_date, end_date=end_date, lookback=lookback)
        
        # Create DataFrame of OHLC prices from NorgateData or Yahoo Finance
        self.df = self.create_base_data(
            ticker=ticker, start_date=self.start_date, end_date=self.end_date, 
            source=source)
        
        # Update the DataFrame with the Quad moving average signal
        self.df = self._quad_ma_signal(
            self.df, ma_1=ma_1, ma_2=ma_2, ma_3=ma_3, ma_4=ma_4, 
            position_size=position_size)    
        
        # Calculate the trades and pnl for the strategy
        self.df = self._profit_data(self.df, position_size=position_size, 
                                   slippage=slippage, commission=commission)

        # Create dictionary of performance data and print out results
        self._output_results(df=self.df, ticker=ticker, 
                             strategy_label=strategy_label)
    
        return self
    
    
    def _date_set(self, start_date, end_date, lookback):
        """
        Create start and end dates if not supplied

        Parameters
        ----------
        start_date : Str, optional
            Date to begin backtest. Format is YYYY-MM-DD. The default is 500 
            business days prior (circa 2 years).
        end_date : Str, optional
            Date to end backtest. Format is YYYY-MM-DD. The default is the 
            last business day.
        lookback : Int, optional
            Number of business days to use for the backtest. The default is 500 
            business days (circa 2 years).

        Returns
        -------
        Str
            Assigns start and end dates to the object.

        """
        
        # If end date is not supplied, set to previous working day
        if end_date is None:
            end_date_as_dt = (dt.datetime.today() - BDay(1)).date()
            end_date = str(end_date_as_dt)
        self.end_date = end_date    
        
        # If start date is not supplied, set to today minus lookback period
        if start_date is None:
            start_date_as_dt = (dt.datetime.today() - 
                                pd.Timedelta(days=lookback*(365/250))).date()
            start_date = str(start_date_as_dt)
        self.start_date = start_date

        return self

    
    @classmethod
    def create_base_data(cls, ticker, start_date, end_date, source):
        """
        Create DataFrame of OHLC prices from NorgateData or Yahoo Finance

        Parameters
        ----------
        ticker : Str, optional
            Underlying to test. The default '$SPX'.
        start_date : Str, optional
            Date to begin backtest. Format is YYYY-MM-DD. The default is 500 
            business days prior (circa 2 years).
        end_date : Str, optional
            Date to end backtest. Format is YYYY-MM-DD. The default is the 
            last business day.
        source : Str, optional
            The data source to use, either 'norgate' or 'yahoo'. The default 
            is 'norgate'.

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
            df = cls._return_yahoo_data(ticker=ticker, start_date=start_date, 
                                         end_date=end_date)
        
            return df
        
        # Otherwise return error message
        else:
            print('Choose norgate or yahoo as data source')
            
    
    @staticmethod
    def _return_yahoo_data(ticker, start_date, end_date):
        """
        Create DataFrame of historic prices for specified ticker.

        Parameters
        ----------
        ticker : Int
            Stock to be returned in the form of Reuters RIC code as a 
            string. 
        start_date : Str
            Start Date represented as a string in the 
            format 'YYYY-MM-DD'.
        end_date : Str
            End Date represented as a string in the 
            format 'YYYY-MM-DD'.
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

    
    @classmethod        
    def _triple_ma_signal(cls, df, short_ma, medium_ma, long_ma, 
                         position_size):
        """
        Create trading signals for Triple MA strategy

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        short_ma : Int, optional
            The fastest of the 3 moving averages. The default is 4 periods.
        medium_ma : Int, optional
            The middle of the 3 moving averages. The default is 9 periods.
        long_ma : Int, optional
            The slowest of the 3 moving averages. The default is 18 periods.
        position_size : Int, optional
            The number of units to trade. The default is 100.

        Returns
        -------
        df : DataFrame
            Returns the input DataFrame with additional columns.

        """
        
        # Set the values of long and short positions
        long_position = 1 * position_size
        short_position = -1 * position_size
        
        # Create short, medium and long simple moving averages  
        df['short_ma'] = df['Close'].rolling(short_ma).mean()
        df['medium_ma'] = df['Close'].rolling(medium_ma).mean()
        df['long_ma'] = df['Close'].rolling(long_ma).mean()
        
        # Create the trade signal fields
        start, pos, trade_signal, trade_number, trade_count = (
            cls._create_signal_fields(column=df['long_ma']))
        
        # for each row in the DataFrame after the longest MA has started
        for row in range(start + 1, len(df['long_ma'])):
            
            # If the medium MA is above the slow MA 
            if df['medium_ma'][row - 1] > df['long_ma'][row - 1]:
                
                # If the fast MA is above the medium MA
                if df['short_ma'][row - 1] > df['medium_ma'][row - 1]:
                    
                    # Set the position to the long position
                    pos[row] = long_position
                
                # If the fast MA is below the medium MA
                else:
                    
                    # Set the position to flat
                    pos[row] = 0
            
            # If the medium MA is below the slow MA
            else:
                
                # If the fast MA is below the medium MA
                if df['short_ma'][row - 1] < df['medium_ma'][row - 1]:
                    
                    # Set the position to the short position
                    pos[row] = short_position
                
                # If the fast MA is above the medium MA
                else:
                    
                    # Set the position to flat
                    pos[row] = 0

            # Calculate trade details    
            trade_signal, pos, trade_number, \
                trade_count = cls._calculate_trades(
                    row=row, trade_signal=trade_signal, pos=pos, 
                    trade_number=trade_number, trade_count=trade_count)
    
        # Set the DataFrame columns to the numpy arrays
        df['position'] = pos
        df['trade_signal'] = trade_signal
        df['trade_number'] = trade_number
    
        return df
    
    
    @classmethod
    def _quad_ma_signal(cls, df, ma_1, ma_2, ma_3, ma_4, position_size):
        """
        Create trading signals for Quad MA strategy

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        ma_1 : Int, optional
            The fastest of the 4 moving averages. The default is 5 periods.
        ma_2 : Int, optional
            The 2nd fastest of the 4 moving averages. The default is 12 
            periods.
        ma_3 : Int, optional
            The second slowest of the 4 moving averages. The default is 20 
            periods.
        ma_4 : Int, optional
            The slowest of the 4 moving averages. The default is 40 periods.
        position_size : Int, optional
            The number of units to trade. The default is 100.

        Returns
        -------
        df : DataFrame
            Returns the input DataFrame with additional columns.

        """
        
        # Set the values of long and short positions
        long_position = 1 * position_size
        short_position = -1 * position_size
        
        # Create the 4 simple moving averages
        df['ma_1'] = df['Close'].rolling(ma_1).mean()
        df['ma_2'] = df['Close'].rolling(ma_2).mean()
        df['ma_3'] = df['Close'].rolling(ma_3).mean()
        df['ma_4'] = df['Close'].rolling(ma_4).mean()
        
        # Create the trade signal fields
        start, pos, trade_signal, trade_number, trade_count = (
            cls._create_signal_fields(column=df['ma_4']))
    
        # for each row in the DataFrame after the longest MA has started
        for row in range(start + 1, len(df['ma_4'])):
            
            # If the second slowest MA is above the slowest MA
            if df['ma_3'][row - 1] > df['ma_4'][row - 1]:
                
                # If the fastest MA is above the second fastest MA
                if df['ma_1'][row - 1] > df['ma_2'][row - 1]:
                    
                    # Set the position to the long position
                    pos[row] = long_position
                
                # If the fastest MA is below the second fastest MA
                else:
                    
                    # Set the position to flat
                    pos[row] = 0
            
            # If the second slowest MA is below the slowest MA
            else:

                # If the fastest MA is below the second fastest MA
                if df['ma_1'][row - 1] < df['ma_2'][row - 1]:

                    # Set the position to the short position
                    pos[row] = short_position

                # If the fastest MA is above the second fastest MA
                else:

                    # Set the position to flat
                    pos[row] = 0
    
            # Calculate trade details
            trade_signal, pos, trade_number, \
                trade_count = cls._calculate_trades(
                    row=row, trade_signal=trade_signal, pos=pos, 
                    trade_number=trade_number, trade_count=trade_count)
    
        # Set the DataFrame columns to the numpy arrays
        df['position'] = pos
        df['trade_signal'] = trade_signal
        df['trade_number'] = trade_number
    
        return df
    
    
    @staticmethod            
    def _create_signal_fields(column):
        """
        Initialize fields for trade signal calculations 

        Parameters
        ----------
        column : Series
            Reference column to start calculation (usually longest MA).

        Returns
        -------
        start : Int
            First row to start calculating from.
        pos : Series
            Empty Array to store position data.
        trade_signal : Series
            Empty Array to store trade signals.
        trade_number : Series
            Empty Array to store trade numbers.
        trade_count : Int
            Current trade number.

        """
        
        # Create start point from first valid number
        start = np.where(~np.isnan(column))[0][0]
        
        # Create numpy array of zeros to store positions
        pos = np.array([0]*len(column))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(column))
        
        # Create numpy array of zeros to store trade numbers
        trade_number = np.array([0]*len(column))
        
        # Set the first trade number to zero
        trade_number[start] = 0
        
        # Set initial trade count to zero
        trade_count = 0
        
        return start, pos, trade_signal, trade_number, trade_count
    

    @staticmethod
    def _calculate_trades(row, trade_signal, pos, trade_number, trade_count):
        """
        Calculate trade information by row. 

        Parameters
        ----------
        row : Int
            The current row number.
        trade_signal : Series
            Array to store trade signals.
        pos : Series
            Array to store position data.
        trade_number : Series
            Array to store trade numbers.
        trade_count : Int
            Number of trades.

        Returns
        -------
        trade_signal : Series
            Array to store trade signals.
        pos : Series
            Array to store position data.
        trade_number : Series
            Array to store trade numbers.
        trade_count : Int
            Current trade number.

        """
        
        # Set the trade signal to todays position minus yesterdays position
        trade_signal[row] = pos[row] - pos[row - 1]
        
        # If today's position is zero
        if pos[row] == 0:
            
            # If yesterday's position is zero
            if pos[row - 1] == 0:
                
                # There is no open trade so set trade number to zero
                trade_number[row] = 0
            
            # If yesterday's position is not zero
            else:
                
                # Set the trade number to the current trade count
                trade_number[row] = trade_count
        
        # If today's position is the same as yesterday        
        elif pos[row] == pos[row - 1]:
            
            # Set the trade number to yesterdays trade number
            trade_number[row] = trade_number[row - 1]
        
        # If today's position is non-zero and different from yesterday    
        else:
            
            # Increase trade count by one for a new trade
            trade_count += 1
            
            # Set the trade number to the current trade count
            trade_number[row] = trade_count    
        
        return trade_signal, pos, trade_number, trade_count
    
           
    @classmethod         
    def _profit_data(cls, df, position_size, slippage, commission):
        """
        Adds profit and drawdown fields to the OHLC data

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        position_size : Int, optional
            The number of units to trade. The default is 100.
        slippage : Float, optional
            The amount of slippage to apply to traded prices. The default is 
            $0.05 per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.

        Returns
        -------
        df : DataFrame
            Returns the input DataFrame with additional columns.

        """
        
        # Create pnl data
        df = cls._pnl_mtm(df=df, slippage=slippage, commission=commission)
        
        # Create perfect profit data
        df = cls._perfect_profit(df=df, position_size=position_size)
        
        # Create max drawdown and max gain data
        df = cls._max_dd_gain(df=df)
            
        return df    


    @classmethod
    def _pnl_mtm(cls, df, slippage, commission):
        """
        Calculate pnl and mark to market columns

        Parameters
        ----------
        df : DataFrame
            The OHLC data and trades signals.
        slippage : Float, optional
            The amount of slippage to apply to traded prices. The default is 
            $0.05 per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.

        Returns
        -------
        df : DataFrame
            The input data with additional columns.

        """
        
        # daily pnl
        df = cls._daily_pnl(df=df, slippage=slippage, commission=commission)
    
        # total pnl
        df['total_pnl'] = np.array([0]*len(df['Close']), dtype=float)
        df['total_pnl'] = df['daily_pnl'].cumsum()
        
        # position mtm
        df['position_mtm'] = df['position'] * df['Close']
        
        return df
    

    @staticmethod
    def _daily_pnl(df, slippage, commission):
        """
        Calculate daily PNL

        Parameters
        ----------
        df : DataFrame
            The OHLC data and trades signals.
        slippage : Float, optional
            The amount of slippage to apply to traded prices. The default is 
            $0.05 per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.

        Returns
        -------
        df : DataFrame
            The input data with additional columns.

        """
        
        # Create series of open, close and position
        open = df['Open']
        close = df['Close']
        pos = df['position']
        
        # Create array of zeros 
        day_pnl = np.array([0]*len(close), dtype=float)
        
        # For each row of closing prices
        for row in range(1, len(close)):
            
            # If the current position is flat
            if pos[row] == 0:
                
                # If the previous days position was flat
                if pos[row - 1] == 0:
                    
                    # Set the pnl to zero
                    day_pnl[row] = 0
                # Otherwise:
                else:
                    # Set the pnl to the previous day's position multiplied by 
                    # the difference between todays open and yesterdays close
                    # less the cost of slippage and commission
                    day_pnl[row] = (
                        pos[row - 1] * (open[row] - close[row - 1]) - 
                        abs(pos[row - 1] * slippage)) - commission
            
            # If the current position is not flat
            else:
                
                # If the position is the same as the previous day
                if pos[row] == pos[row - 1]:
                    
                    # Set the pnl to the current position * the difference 
                    # between todays close and yesterdays close
                    day_pnl[row] = pos[row] * (close[row] - close[row - 1])
                
                # If the position is not the same as the previous day
                else:
                    
                    # Set the pnl to the current position * the difference 
                    # between todays open and todays close less the cost of 
                    # slippage and commission
                    day_pnl[row] = (
                        pos[row] * (close[row] - open[row]) - 
                        abs(pos[row] * slippage)) - commission
        
        # Create daily pnl column in DataFrame, rounding to 2dp            
        df['daily_pnl'] = np.round(day_pnl, 2)    
    
        return df
    
    
    @staticmethod    
    def _perfect_profit(df, position_size):
        """
        Theoretical optimal of buying every low and selling every high

        Parameters
        ----------
        df : DataFrame
            The OHLC data, trades signals and pnl.
        position_size : Int
            Number of units traded.

        Returns
        -------
        df : DataFrame
            The input data with additional columns.

        """
        
        # Absolute difference between high and low price multiplied by 
        # position size
        df['daily_perfect_profit'] = (abs(df['High'] - df['Low']) 
                                      * position_size)
        
        # Create array of zeros
        df['total_perfect_profit'] = np.array(
            [0]*len(df['Close']), dtype=float)
        
        # Cumulative sum of daily perfect profit column
        df['total_perfect_profit'] = df['daily_perfect_profit'].cumsum()
    
        return df
    
    
    @staticmethod
    def _max_dd_gain(df):
        """
        Create maximum drawdown and maximum gain columns

        Parameters
        ----------
        df : DataFrame
            The OHLC data, trades signals and pnl.

        Returns
        -------
        df : DataFrame
            The input data with additional columns.

        """
        
        # Create total pnl series from DataFrame column
        total_pnl = df['total_pnl']
        
        # Create max drawdown and max gain numpy arrays of zeros
        max_draw = np.array([0]*len(total_pnl), dtype=float)
        max_gain = np.array([0]*len(total_pnl), dtype=float)
        
        # For each row of pnl in the DataFrame
        for row in range(1, len(total_pnl)):
            
            # Maximum drawdown is the smallest value of the current cumulative 
            # pnl less the max of all previous rows cumulative pnl and zero
            max_draw[row] = min(total_pnl[row] - max(total_pnl.iloc[0:row]), 0)
            
            # Maximum gain is the largest value of the current cumulative 
            # pnl less the min of all previous rows and zero
            max_gain[row] = max(total_pnl[row] - min(total_pnl.iloc[0:row]), 0)
        
        # Set the DataFrame columns to the numpy arrays 
        df['max_dd'] = max_draw
        df['max_gain'] = max_gain
        
        return df        
   
        
    def _output_results(self, df, ticker, strategy_label):
        """
        Create dictionary of performance data and print out results of backtest

        Parameters
        ----------
        df : DataFrame
            The OHLC data, trades signals and pnl.
        ticker : Str, optional
            Underlying to test. The default '$SPX'.
        strategy : Str
            Label of the strategy being tested.

        Returns
        -------
        Results
            Prints out performance data for the strategy.

        """
        
        # Create performance data dictionary
        self.performance_data(df=df, contract=ticker, 
                              strategy_label=strategy_label)
        
        # Print out results
        self.report_table(input_dict=self.perf_dict)
    
        return self


    def performance_data(self, df, contract, strategy_label):
        """
        Create dictionary of performance data

        Parameters
        ----------
        df : DataFrame
            The OHLC data, trades signals and pnl.
        contract : Str
            The underlying being tested.
        strategy : Str
            Label of the strategy being tested.

        Returns
        -------
        Dict
            Dictionary of performance data.

        """
        
        # Create empty dictionary
        perf_dict = {}
        
        # Contract and strategy details
        perf_dict['contract'] = contract
        perf_dict['strategy'] = strategy_label
        
        # Slippage and commission in dollars
        perf_dict['slippage'] = self.slippage
        perf_dict['commission'] = self.commission
        
        # Start and end dates
        perf_dict['start_date'] = df.index[0].date().strftime("%d/%m/%y")
        perf_dict['end_date'] = df.index[-1].date().strftime("%d/%m/%y")
               
        # Maximum margin required 
        perf_dict['margin'] = math.ceil(
            max(df['position_mtm']) / 10000) * 10000
        
        # Net Profit
        perf_dict['net_pnl'] = df['total_pnl'].iloc[-1]
        
        # Convert the start and end date strings to dates 
        start = dt.datetime.strptime(perf_dict['start_date'], "%d/%m/%y")
        end = dt.datetime.strptime(perf_dict['end_date'], "%d/%m/%y")
        
        # Number of days between start and end dates
        period = (end-start).days
        
        # Set the return period in years
        perf_dict['return_period'] = period / 365

        # Annualized profit
        perf_dict['annualized_profit '] = (
            perf_dict['net_pnl'] / perf_dict['return_period'])

        # Total return 
        perf_dict['total_return_rate'] = (
            perf_dict['net_pnl'] / perf_dict['margin'])
        
        # Annualized total return
        perf_dict['annualized_return'] = (
            (1 + perf_dict['total_return_rate']) 
            ** (1 / perf_dict['return_period']) - 1) 
        
        # Calculate trade data
        trades, perf_dict['total_trades'], trades_win_dict, trades_win_list, \
            trades_loss_dict, trades_loss_list = self._trade_data(df)
        
        # number of wins and losses
        perf_dict['num_wins'] = len(trades_win_dict)
        perf_dict['num_losses'] = len(trades_loss_dict)
        
        # Sum of all profits
        perf_dict['total_profit'] = np.round(sum(trades_win_dict.values()), 2)
        
        # Sum of all losses
        perf_dict['total_loss'] = np.round(sum(trades_loss_dict.values()), 2)
    
        # Percentage of winning trades    
        perf_dict['win_percent'] = int(
            np.round(perf_dict['num_wins'] / len(trades) * 100, 0))
        
        # Percentage of losing trades
        perf_dict['loss_percent'] = int(
            np.round(perf_dict['num_losses'] / len(trades) * 100, 0))
        
        # Max winning trade
        perf_dict['max_win'] = np.round(max(trades_win_dict.values()), 2)
        
        # Min winning trade
        perf_dict['min_win'] = np.round(min(trades_win_dict.values()), 2)
        
        # average winning trade
        perf_dict['av_win'] = np.round(
            sum(trades_win_dict.values()) / len(trades_win_list), 2)
        
        # standard deviation of winning trades
        perf_dict['std_wins'] = np.round(np.std(trades_win_list), 2)
        
        # Max losing trade
        perf_dict['max_loss'] = min(trades_loss_dict.values())
        
        # Maximum win/loss ratio
        perf_dict['max_win_loss_ratio'] = np.round(
            abs(perf_dict['max_win'] / perf_dict['max_loss']), 2)
        
        # Min losing trade
        perf_dict['min_loss'] = max(trades_loss_dict.values())
        
        # Minimum win/loss ratio
        perf_dict['min_win_loss_ratio'] = np.round(
            abs(perf_dict['min_win'] / perf_dict['min_loss']), 2)
        
        # average losing trade
        perf_dict['av_loss'] = np.round(
            sum(trades_loss_dict.values()) / len(trades_loss_list), 2)
        
        # standard deviation of losing trades
        perf_dict['std_losses'] = np.round(np.std(trades_loss_list), 2)
        
        # avwin / avloss
        perf_dict['av_win_loss_ratio'] = np.round(
            abs(perf_dict['av_win'] / perf_dict['av_loss']), 2)
        
        # average trade
        perf_dict['av_trade'] = np.round(perf_dict['net_pnl'] / len(trades), 2)
        
        # Pessimistic Return on Margin
        adjusted_gross_profit = perf_dict['av_win'] * (
            perf_dict['num_wins'] - np.round(np.sqrt(perf_dict['num_wins'])))
        adjusted_gross_loss = perf_dict['av_loss'] * (
            perf_dict['num_losses'] - np.round(
                np.sqrt(perf_dict['num_losses'])))
        perf_dict['prom'] = np.round(
            (adjusted_gross_profit - adjusted_gross_loss) / 
            perf_dict['margin'], 2) 
        
        # PROM minus biggest win
        perf_dict['prom_minus_max_win'] = np.round(
            (adjusted_gross_profit - perf_dict['max_win'] 
             - adjusted_gross_loss) / perf_dict['margin'], 2)
        
        # Perfect profit - Buy every dip & sell every peak
        perf_dict['perfect_profit'] = df['total_perfect_profit'].iloc[-1]
    
        # Model Efficiency - ratio of net pnl to perfect profit
        perf_dict['model_efficiency'] = np.round(
            perf_dict['net_pnl'] / perf_dict['perfect_profit'], 2)
        
        # Winning run data
        perf_dict['max_win_run_pnl'], perf_dict['max_win_run_count'], \
            perf_dict['min_win_run_pnl'], perf_dict['min_win_run_count'], \
                perf_dict['num_win_runs'], perf_dict['av_win_run_count'], \
                    perf_dict['av_win_run_pnl'], \
                        perf_dict['win_pnl'] = self._trade_runs(
                            trades_win_list, run_type='win') 
        
        # Losing run data
        perf_dict['max_loss_run_pnl'], perf_dict['max_loss_run_count'], \
            perf_dict['min_loss_run_pnl'], perf_dict['min_loss_run_count'], \
                perf_dict['num_loss_runs'], perf_dict['av_loss_run_count'], \
                    perf_dict['av_loss_run_pnl'], \
                        perf_dict['loss_pnl'] = self._trade_runs(
                            trades_loss_list, run_type='loss')
        
        # Maximum Equity drawdown
        perf_dict['max_balance_drawback'] = np.round(min(df['max_dd']), 2)
        
        # Reward / Risk ratio
        perf_dict['reward_risk'] = (perf_dict['net_pnl'] / 
                                    abs(perf_dict['max_balance_drawback']))
        
        # Annual Rate of Return
        perf_dict['annual_ror'] = (perf_dict['annualized_profit '] / 
                                   perf_dict['margin']) * 100
        
        # Profit Index
        perf_dict['profit_index'] = (perf_dict['total_profit'] / 
                                     abs(perf_dict['total_loss']))
        
        # Get the index of the first trade entry
        # Select just the rows of the first trade from the DataFrame
        first_trade = df[df['trade_number']==1]
        
        # Find the date of the trade entry 
        target = first_trade.index[0]
        
        # Find the location of this in the original index
        first_trade_start = df.index.get_loc(target)
        
        # Return of a buy and hold strategy since the first trade entry
        perf_dict['long_only_pnl'] = (
            (df['Close'].iloc[-1] - df['Open'].iloc[first_trade_start]) * 
            self.position_size) 
        
        # Annual Rate of return of buy and hold
        perf_dict['annual_long_only_ror'] = (
            (perf_dict['long_only_pnl'] / perf_dict['return_period']) / 
            perf_dict['margin']) * 100
        
        # Maximum Equity gain
        perf_dict['max_gain'] = np.round(max(df['max_gain']), 2)
        
        # Values still to be worked out
        placeholder_dict = {
            'beg_balance':0.00,
            'open_equity':0.00,
            'open_equity_dd':0.00,
            'max_equity_profit':0.00,
            'pessimistic_margin':0.00,
            'adj_pess_margin':0.00,
            'pess_month_avg':0.00,
            'pess_month_variance':0.00,
            'month_net_pnl_large':0.00,
            'month_net_pnl_small':0.00,
            'month_net_pnl_av':0.00,    
            'mod_pess_margin':0.00,
            }
        
        perf_dict.update(placeholder_dict)
        
        # assign perf dict to the object
        self.perf_dict = perf_dict
        
        return self
        
    
    @staticmethod
    def _trade_data(df):
        """
        Create dictionary of trades, count of the number of trades and lists / 
        dictionaries of winning and losing trades 

        Parameters
        ----------
        df : DataFrame
            The OHLC data with trade signals.

        Returns
        -------
        trades : Dict
            Dictionary containing all the trades.
        num_trades : Int
            Number of trades.
        trades_win_dict : Dict
            Dictionary of winning trades.
        trades_win_list : TYPE
            List of winning trades.
        trades_loss_dict : TYPE
            Dictionary of losing trades.
        trades_loss_list : TYPE
            List of losing trades.

        """
        # profit per trade
        # Create empty trades dictionary
        trades = {}
        
        # Count the number of unique trade numbers (less 1 for the 0 start)
        num_trades = len(pd.unique(df['trade_number'])) - 1
        
        # For each trade number
        for trade_number in range(1, num_trades+1):
            
            # Calculate profit as the sum of daily pnl for that trade number
            profit = df[df['trade_number']==trade_number]['daily_pnl'].sum()
            
            # Assign this number (rounded to 2dp) to the trades dictionary
            trades[trade_number] = np.round(profit, 2)
    
        # Split winning and losing trades
        # Create empty win/loss dictionaries and lists
        trades_win_dict = {}
        trades_win_list = list()
        trades_loss_dict = {}
        trades_loss_list = list()
        
        # for each trade and profit in the trades dictionary
        for key, value in trades.items():
            
            # If the profit is negative
            if value < 0:
                
                # Add the trade to the loss dictionary and list (as a tuple)
                trades_loss_dict[key] = value
                trades_loss_list.append((key, value))
            
            # If the profit is positive
            else:
                
                # Otherwise add to the win dictionary and list (as a tuple)
                trades_win_dict[key] = value
                trades_win_list.append((key, value))
    
        return (trades, num_trades, trades_win_dict, trades_win_list, 
                trades_loss_dict, trades_loss_list)
    
    
    @staticmethod
    def _trade_runs(input_trades_list, run_type='win'):
        """
        Produce data for winning or losing runs of trades

        Parameters
        ----------
        input_trades_list : List
            List of winning or losing trades.
        run_type : Str, optional
            Whether winning or losing run data is being calculated. The 
            default is 'win'.

        Returns
        -------
        max_run_pnl : Float
            Largest profit or loss.
        max_run_count : Int
            Number of trades in largest profit or loss.
        min_run_pnl : Float
            Smallest profit or loss.
        min_run_count : Int
            Number of trades in smallest profit or loss.
        num_runs : Int
            Count of number of winning or losing runs.
        av_run_count : Int
            Count of average number of trades in winning or losing runs.
        av_run_pnl : Float
            Average profit or loss.
        pnl : Tuple
            PNL for each run and number of trades.

        """
        
        # Set initial values
        max_run_count = 1
        run_count = 1
        run_trades_list = []
        total_run_trades_list = []
        last_trade_count = 0
        
        # For each trade in the winning or losing trades list, sorting by the 
        # trade number
        for num, trade in enumerate(sorted(input_trades_list)):
            # For the first trade
            if num == 0:
                
                # Add the trade pnl to the winning / losing trades run list 
                run_trades_list.append(trade[1])
            
            # Otherwise, if the trade number is next in sequence after the 
            # last stored trade number
            elif trade[0] == last_trade_count + 1:
                
                # Increase the run count by one
                run_count +=1
                
                # Update the longest run count
                max_run_count = max(max_run_count, run_count)
                
                # Add the trade pnl to the winning / losing trades run list
                run_trades_list.append(trade[1])
        
            # If the trade is not the next in sequence:
            else:
                
                # Add the current run to the list of all runs
                total_run_trades_list.append(run_trades_list)   
                
                # If this is not the last trade
                if num != len(input_trades_list) - 1:
                    
                    # Reset the winning / losing trades run list
                    run_trades_list = []
                    
                    # Add the trade pnl to the winning / losing trades run list
                    run_trades_list.append(trade[1])        
                    
                    # Increase the run count by one
                    run_count = 1
 
                # If it is the last trade                 
                else:
                    
                    # Reset the winning / losing trades run list
                    run_trades_list = []
                    
                    # Add the trade pnl to the winning / losing trades run list
                    run_trades_list.append(trade[1])
                    
                    # Add the current run to the list of all runs
                    total_run_trades_list.append(run_trades_list)

            # Set the last trade count number to the current trade number        
            last_trade_count = trade[0]
        
        # Tuple for each run of PNL and number of trades.
        pnl = sorted([(sum(x), len(x)) for x in total_run_trades_list])    
        
        # Values to select for winning runs
        if run_type == 'win':
            max_run_pnl = pnl[-1][0]
            max_run_count = pnl[-1][-1]
            min_run_pnl = pnl[0][0]
            min_run_count = pnl[0][-1]
    
        # Values to select for losing runs
        else:
            max_run_pnl = pnl[0][0]
            max_run_count = pnl[0][-1]
            min_run_pnl = pnl[-1][0]
            min_run_count = pnl[-1][-1]
            
        # Count number of runs as the length of the pnl list     
        num_runs = len(pnl)
        
        # Take the average number of runs as the sum of run lengths in pnl 
        # tuple divided by the number of runs
        av_run_count = int(np.round(sum(j for i, j in pnl) / len(pnl), 0))
        
        # Take the average run pnl as the sum of run pnls in pnl tuple
        # divided by the number of runs
        av_run_pnl = np.round(sum(i for i, j in pnl) / len(pnl), 2)    
        
        return max_run_pnl, max_run_count, min_run_pnl, min_run_count, \
            num_runs, av_run_count, av_run_pnl, pnl
       

    @classmethod
    def report_table(cls, input_dict):
        """
        Print out backtesting performance results. 
        
        Modelled on performance results presented in Robert Pardo, 1992, 
        Design, Testing and Optimization of Trading Systems

        Parameters
        ----------
        input_dict : Dict
            Dictionary of performance data.

        Returns
        -------
        Prints results to the console.

        """
        
        # Format the performance dictionary so that the financial data is 
        # rounded to 2 decimal places and set values as strings
        input_dict = cls._dict_format(input_dict)
        
        # Format header - centred and with lines above and below
        print('='*78)
        print('{:^78}'.format('Performance Analysis Report'))
        print('-'*78)
        
        # Contract traded on left and period covered on right
        print('Contract Traded : {:<41}{} - {}'.format(
            input_dict['contract'], 
            input_dict['start_date'], 
            input_dict['end_date']))
        
        # Strategy
        print('Strategy        : {:>10}'.format(input_dict['strategy']))
        
        # Beginning balance on left
        print('Beg. Balance    :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['beg_balance'],
            '',
            'Margin           : $',
            input_dict['margin']))
        
        # Commission, slippage and margin 
        print('Commission      :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['commission'],
            '',
            'Slippage         : $',
            input_dict['slippage']))
        
        # Empty line
        print()
        
        # Profit stats
        print('Net P/L         :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['net_pnl'],
            '',
            'Perfect Profit   : $',
            input_dict['perfect_profit']))
        print('Total Profit    :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['total_profit'],
            '',
            'Total Loss       : $',
            input_dict['total_loss']))
        print('Av. Trade       :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['av_trade'],
            '',
            'Model Efficiency :  ',
            input_dict['model_efficiency']))
        
        # Headers for # trades, Max, Min and Average with lines above and below
        print('-'*78)
        print('{:>16}:{:^10}:{:^17}:{:^17}:{:^14}'.format(
            '',
            '# Trades',
            'Maximum',
            'Minimum',
            'Average'))
        print('-'*78)
        
        # Total trades summary
        print('Total...........:{:^10}:{:^17}:{:^17}:'.format(
            input_dict['total_trades'], 
            'Win {} / {}%'.format(
                input_dict['num_wins'], input_dict['win_percent']), 
            'Loss {} / {}%'.format(
                input_dict['num_losses'], input_dict['loss_percent'])))
        
        # Winning trades and winning runs 
        print('Win.............:{:^10}:{:>11}{:>6}:{:>10}{:>7}:{:>9}'.format(
            input_dict['num_wins'], 
            input_dict['max_win'],
            '',
            input_dict['min_win'], 
            '',
            input_dict['av_win']))
        
        print('Win runs........:{:^10}:{:>11}{:<6}:{:>10}{:<7}:{:>9}{}'.format(
            input_dict['num_win_runs'], 
            input_dict['max_win_run_pnl'],
            ' / {}'.format(input_dict['max_win_run_count']),
            input_dict['min_win_run_pnl'],
            ' / {}'.format(input_dict['min_win_run_count']), 
            input_dict['av_win_run_pnl'],
            ' / {}'.format(input_dict['av_win_run_count'])))
        
        # Losing trades and losing runs
        print('Loss............:{:^10}:{:>11}{:>6}:{:>10}{:>7}:{:>9}'.format(
            input_dict['num_losses'], 
            input_dict['max_loss'], 
            '',
            input_dict['min_loss'], 
            '',
            input_dict['av_loss']))
        
        print('Loss runs.......:{:^10}:{:>11}{:<6}:{:>10}{:<7}:{:>9}{}'.format(
            input_dict['num_loss_runs'], 
            input_dict['max_loss_run_pnl'],
            ' / {}'.format(input_dict['max_loss_run_count']),
            input_dict['min_loss_run_pnl'],
            ' / {}'.format(input_dict['min_loss_run_count']), 
            input_dict['av_loss_run_pnl'],
            ' / {}'.format(input_dict['av_loss_run_count'])))
        
        # Win loss ratio
        print('Win/Loss Ratio..:{:>10}:{:>11}{:>6}:{:>10}{:>7}:{:>9}'.format(
            '', 
            input_dict['max_win_loss_ratio'],
            '',
            input_dict['min_win_loss_ratio'], 
            '',
            input_dict['av_win_loss_ratio']))
    
        # Separating line
        print('-'*78)
    
        # Open equity
        print('Open Equity.......... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['open_equity'],
            '',
            'Open Equity Drawdown.. $',
            input_dict['open_equity_dd']))    
        
        # Max Balance drawback / equity profit
        print('Max Balance Drawback. ${:>10}{:<6}{}{:>10}'.format(
            input_dict['max_balance_drawback'],
            '',
            'Max Equity Profit..... $',
            input_dict['max_equity_profit']))
        
        # Reward / Risk
        print('Reward / Risk........  {:^16}{}{:>10}'.format(
            '{} TO 1'.format(input_dict['reward_risk']),
            'Annual Rate of Return. %',
            input_dict['annual_ror']))
        
        # Profit Index
        print('Profit Index.........  {:>10}'.format(
            input_dict['profit_index']))
    
        # Pessimistic Margin
        print('Pessimistic Margin...  {:>10}{:<6}{}{:>10}'.format(
            input_dict['pessimistic_margin'],
            '',
            'Adjusted Pess. Margin.  ',
            input_dict['adj_pess_margin']))
        
        print('Pess. Month Avg......  {:>10}{:<6}{}{:>10}'.format(
            input_dict['pess_month_avg'],
            '',
            'Pess. Month Variance..  ',
            input_dict['pess_month_variance']))
        
        # Monthly P/L
        print('Monthly Net PL Large.  {:>10}{:<6}{}{:>10}'.format(
            input_dict['month_net_pnl_large'],
            '',
            'Monthly Net PL Small..  ',
            input_dict['month_net_pnl_small']))
        
        print('Monthly Net PL Ave...  {:>10}{:<6}{}{:>10}'.format(
            input_dict['month_net_pnl_av'],
            '',
            'Modified Pess. Margin.  ',
            input_dict['mod_pess_margin']))
        
        print('Long Only Net PL.....  {:>10}{:<6}{}{:>10}'.format(
            input_dict['long_only_pnl'],
            '',
            'Long Only Annual RoR.. %',
            input_dict['annual_long_only_ror']))
        
        print('='*78)


    @staticmethod
    def _dict_format(input_dict):
        """
        Format the performance dictionary so that the financial data is 
        rounded to 2 decimal places and set values as strings

        Parameters
        ----------
        input_dict : Dict
            The performance data dictionary.

        Returns
        -------
        str_input_dict : Dict
            The formatted performance data dictionary.

        """
        
        # Create empty dictionary
        str_input_dict = {}
        
        # Set decimal format
        dp2 = Decimal(10) ** -2  # (equivalent to Decimal '0.01')
        
        # For each entry in the dictionary
        for key, value in input_dict.items():
            
            # If the value is a floating point number
            if type(value) in (float, np.float64):
                
                # Apply the decimal formatting and convert to string
                str_input_dict[key] = str(Decimal(value).quantize(dp2))
            else:
                
                # Otherwise just convert to string
                str_input_dict[key] = str(value)
    
        return str_input_dict
    
    
