# Imports
import datetime as dt
import math
import norgatedata
import numpy as np
import os
import pandas as pd
import tradingsystems.systems_params as sp
import random
import requests
from decimal import Decimal
from operator import itemgetter
from pandas.tseries.offsets import BDay
from scipy.stats import skew, kurtosis
from technicalmethods.methods import Indicators
from yahoofinancials import YahooFinancials

class Data():
    
    def __init__(self):
        
        # Import dictionary of default parameters 
        self.df_dict = sp.system_params_dict
        
        # Extract longnames for Norgate Tickers
        self._norgate_name_dict()
        
        # Extract the Entry, Exit and Stop labels
        self.entry_signal_labels = self.df_dict['df_entry_signal_labels']
        self.exit_signal_labels = self.df_dict['df_exit_signal_labels']
        self.stop_signal_labels = self.df_dict['df_stop_signal_labels']
        
        # Extract the Entry, Exit and Stop dictionaries of defaults
        self.entry_signal_dict = self.df_dict['df_entry_signal_dict']
        self.exit_signal_dict = self.df_dict['df_exit_signal_dict']
        self.stop_signal_dict = self.df_dict['df_stop_signal_dict']        
  
    
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
   

    def _refresh_signals_default(self, **kwargs):
        """
        Set parameters for use in various pricing functions
        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.
        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults 
            if no data provided
        """
 
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Check if the key is in the entry signal dict 
                try:                    
                    # Extract these from the df_combo_dict
                    v = self.entry_signal_dict[str(
                        self.entry_type)][str(k)]
                    
                    # Assign to input dictionary
                    kwargs[k] = v

                except:
                    pass
                        
                # Check if the key is in the exit signal dict    
                try:                    
                    # Extract these from the exit signal dict
                    v = self.exit_signal_dict[str(
                        self.exit_type)][str(k)]
                    
                    # Assign to input dictionary
                    kwargs[k] = v

                except:
                    pass    
                    
                # Check if the key is in the entry signal dict                
                try:                    
                    # Extract these from the df_combo_dict
                    v = self.stop_signal_dict[str(
                        self.stop_type)][str(k)]
                    
                    # Assign to input dictionary
                    kwargs[k] = v

                except:
                   
                    if v is None:
                        # Otherwise set to the standard default 
                        # value
                        v = self.df_dict['df_'+str(k)]
                        kwargs[k] = v
            
                # Now assign this to the object and input dictionary
                self.__dict__[k] = v
            
            # If the parameter has been provided as an input, 
            # assign this to the object
            else:
                self.__dict__[k] = v
       
                        
        return kwargs            
    
    
    def test_strategy(self, **kwargs):
        
        try:
            if kwargs['reversal']:
                self.test_strategy_reversal(**kwargs)
            else:
                self.test_strategy_exit_stop(**kwargs)
        except:
            self.test_strategy_exit_stop(**kwargs)
    
    
    def test_strategy_reversal(
            self, ticker=None, start_date=None, end_date=None, lookback=None, 
            short_ma=None, medium_ma=None, long_ma=None, ma_1=None, ma_2=None,
            ma_3=None, ma_4=None,position_size=None, source=None, 
            slippage=None, commission=None, strategy=None, equity=None, 
            reversal=None):
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
            The number of units to trade. The default is based on equity.
        source : Str, optional
            The data source to use, either 'norgate' or 'yahoo'. The default 
            is 'norgate'.
        slippage : Float, optional
            The amount of slippage to apply to traded prices in basis points. 
            The default is 5 bps per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.
        equity : Float
            The initial account equity level.    

        Returns
        -------
        Results
            Prints out performance data for the strategy.

        """
                
        # If data is not supplied as an input, take default values 
        (ticker, lookback, short_ma, medium_ma, long_ma, ma_1, ma_2, ma_3, 
         ma_4, source, slippage, commission,
         strategy, equity) = itemgetter(
             'ticker', 'lookback', 'short_ma', 'medium_ma', 'long_ma', 'ma_1', 
             'ma_2', 'ma_3', 'ma_4', 'source', 'slippage', 
             'commission', 'strategy', 'equity')(self._refresh_params_default(
                 ticker=ticker, lookback=lookback, short_ma=short_ma, 
                 medium_ma=medium_ma, long_ma=long_ma, ma_1=ma_1, ma_2=ma_2, 
                 ma_3=ma_3, ma_4=ma_4, source=source, slippage=slippage, 
                 commission=commission, strategy=strategy, equity=equity))
        
       
        # Set the start and end dates if not provided
        self._date_set(
            start_date=start_date, end_date=end_date, lookback=lookback)
        
        # Create DataFrame of OHLC prices from NorgateData or Yahoo Finance
        df = self.create_base_data(
            ticker=ticker, start_date=self.start_date, end_date=self.end_date, 
            source=source)

        # Set the position size
        position_size = self._position_size(df=df, equity=equity)    

        # Set the strategy labels
        df, self.strategy_label = self._strategy_selection(
            short_ma=short_ma, medium_ma=medium_ma, long_ma=long_ma, ma_1=ma_1, 
            ma_2=ma_2, ma_3=ma_3, ma_4=ma_4, df=df, 
            position_size=position_size, strategy=strategy)
                       
        # Calculate the trades and pnl for the strategy
        self.df = self._profit_data(df=df, position_size=position_size, 
                                    slippage=slippage, commission=commission, 
                                    equity=equity)
        
        # Create monthly summary data
        self.monthly_data = self._create_monthly_data(df=self.df, 
                                                      equity=equity)
        
        # Create dictionary of performance data and print out results
        self._output_results(df=self.df, monthly_data=self.monthly_data, 
                             reversal=True)

        return self
    
    
    def test_strategy_exit_stop(
            self, ticker=None, ccy_1=None, ccy_2=None, asset_type=None, 
            start_date=None, end_date=None, lookback=None, 
            ma1=None, ma2=None, ma3=None, ma4=None, simple_ma=None, 
            position_size=None, pos_size_fixed=None, source=None, 
            slippage=None, commission=None, strategy=None, entry_type=None, 
            exit_type=None, stop_type=None, entry_period=None, 
            exit_period=None, stop_period=None,
            entry_oversold=None, entry_overbought=None, 
            exit_oversold=None, exit_overbought=None, 
            entry_threshold=None, exit_threshold=None,
            entry_acceleration_factor=None, exit_acceleration_factor=None, 
            sip_price=None, equity=None, exit_amount=None, stop_amount=None, 
            reversal=None):
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
            The number of units to trade. The default is based on equity.
        source : Str, optional
            The data source to use, either 'norgate' or 'yahoo'. The default 
            is 'norgate'.
        slippage : Float, optional
            The amount of slippage to apply to traded prices in basis points. 
            The default is 5 bps per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.
        equity : Float
            The initial account equity level.   
        exit_amount : Float
            The dollar exit amount
        stop_amount : Float
            The dollar stop amount    

        Returns
        -------
        Results
            Prints out performance data for the strategy.

        """
        
        if reversal:
            strategy = self.df_dict['df_strategy']
        else:
            (entry_type, exit_type, stop_type) = itemgetter(
                'entry_type', 'exit_type', 
                'stop_type')(self._refresh_params_default(
                    entry_type=entry_type, exit_type=exit_type, 
                    stop_type=stop_type))
        
        
        # Basic parameters
        # If data is not supplied as an input, take default values 
        (ticker, ccy_1, ccy_2, asset_type, lookback, simple_ma, position_size, 
         pos_size_fixed, source, slippage, commission, strategy, 
         equity) = itemgetter(
             'ticker', 'ccy_1', 'ccy_2', 'asset_type', 'lookback', 'simple_ma', 
             'position_size', 'pos_size_fixed', 'source', 'slippage', 
             'commission', 'strategy', 'equity')(self._refresh_params_default(
                 ticker=ticker, ccy_1=ccy_1, ccy_2=ccy_2, 
                 asset_type=asset_type, lookback=lookback, simple_ma=simple_ma, 
                 position_size=position_size, pos_size_fixed=pos_size_fixed, 
                 source=source, slippage=slippage, commission=commission, 
                 strategy=strategy, equity=equity))
                
        # Strategy specific parameters         
        # If data is not supplied as an input, take default values 
        (ma1, ma2, ma3, ma4, entry_period, exit_period, stop_period,
         entry_oversold, entry_overbought, exit_oversold,
         exit_overbought, entry_threshold, exit_threshold,
         entry_acceleration_factor, exit_acceleration_factor, sip_price, 
         exit_amount, stop_amount) = itemgetter(
             'ma1', 'ma2', 'ma3', 'ma4', 'entry_period', 'exit_period', 
             'stop_period', 'entry_oversold', 'entry_overbought', 
             'exit_oversold', 'exit_overbought', 'entry_threshold', 
             'exit_threshold', 'entry_acceleration_factor', 
             'exit_acceleration_factor', 'sip_price', 'exit_amount', \
                 'stop_amount')(self._refresh_signals_default(
                 ma1=ma1, ma2=ma2, ma3=ma3, ma4=ma4, entry_period=entry_period, 
                 exit_period=exit_period, stop_period=stop_period, 
                 entry_oversold=entry_oversold, 
                 entry_overbought=entry_overbought, 
                 exit_oversold=exit_oversold, 
                 exit_overbought=exit_overbought, 
                 entry_threshold=entry_threshold, 
                 exit_threshold=exit_threshold, 
                 entry_acceleration_factor=entry_acceleration_factor, 
                 exit_acceleration_factor=exit_acceleration_factor,
                 sip_price=sip_price, 
                 exit_amount=exit_amount, stop_amount=stop_amount))
        
                             
        # Set the start and end dates if not provided
        self._date_set(
            start_date=start_date, end_date=end_date, lookback=self.lookback)
        
        # Create DataFrame of OHLC prices from NorgateData or Yahoo Finance
        df = self.create_base_data(
            ticker=self.ticker, ccy_1=self.ccy_1, ccy_2=self.ccy_2, 
            start_date=self.start_date, end_date=self.end_date, 
            lookback=self.lookback, source=self.source, 
            asset_type=self.asset_type)

        # Extract SPX data for Beta calculation
        self.spx = norgatedata.price_timeseries(
            symbol='$SPX', start_date=self.start_date, end_date=self.end_date, 
            format='pandas-dataframe')
            
        # Set the position size
        position_size = self._position_size(df=df, equity=self.equity)    

        # Set the strategy labels
        self.entry_label, self.exit_label, \
            self.stop_label = self._strategy_labels(
            df=df, ma1=self.ma1, ma2=self.ma2, ma3=self.ma3, ma4=self.ma4, 
            entry_period=self.entry_period, exit_period=self.exit_period, 
            stop_period=self.stop_period, entry_oversold=self.entry_oversold, 
            exit_oversold=self.exit_oversold, 
            entry_overbought=self.entry_overbought, 
            exit_overbought=self.exit_overbought, 
            entry_threshold=self.entry_threshold, 
            exit_threshold=self.exit_threshold, simple_ma=self.simple_ma, 
            position_size=self.position_size, 
            entry_type=self.entry_type, exit_type=self.exit_type, 
            stop_type=self.stop_type, exit_amount=self.exit_amount,
            stop_amount=self.stop_amount, 
            entry_acceleration_factor=self.entry_acceleration_factor, 
            exit_acceleration_factor=self.exit_acceleration_factor,
            sip_price=self.sip_price)
        
        # Generate initial trade data
        df, self.start = self._raw_signal_data(
            df=df, entry_type=self.entry_type, position_size=position_size, 
            exit_amount=self.exit_amount, ma1=self.ma1, ma2=self.ma2, 
            ma3=self.ma3, ma4=self.ma4, entry_period=self.entry_period, 
            entry_oversold=self.entry_oversold, 
            entry_overbought=self.entry_overbought, 
            entry_threshold=self.entry_threshold,
            simple_ma=self.simple_ma, 
            entry_acceleration_factor=self.entry_acceleration_factor)
        
        # Create exit and stop signals
        df = self._exit_and_stop_signals(
            df=df, position_size=position_size, exit_type=self.exit_type, 
            stop_type=self.stop_type, exit_amount=self.exit_amount, 
            exit_period=self.exit_period, stop_amount=self.stop_amount, 
            stop_period=self.stop_period, exit_threshold=self.exit_threshold, 
            exit_oversold=self.exit_oversold, 
            exit_overbought=self.exit_overbought,
            exit_acceleration_factor=self.exit_acceleration_factor, 
            sip_price=self.sip_price)
        
        # Combine signals
        df['combined_signal'] = self._signal_combine(
            df=df, start=self.start, raw_trade_signal=df['raw_td_signal'], 
            end_of_day_position=df['raw_end_of_day_position'], 
            trade_number=df['raw_td_number'], 
            exit_signal=df['exit_signal'], 
            stop_signal=df['stop_signal'])
        
        # Create trade and position data
        df['start_of_day_position'], df['trade_signal'], \
            df['position'] = self._positions_and_trade_actions(
                df=df, signal=df['combined_signal'], start=self.start, 
                position_size=self.position_size)
            
        df['trade_number'] = self._trade_numbers(
            df=df, end_of_day_position=df['position'], 
            start=self.start)    
                
        # Calculate the trades and pnl for the strategy
        self.df = self._profit_data(df=df, position_size=self.position_size, 
                                    slippage=self.slippage, 
                                    commission=self.commission, 
                                    equity=self.equity)
        
        # Create monthly summary data
        self.monthly_data = self._create_monthly_data(
            df=self.df, equity=self.equity)
        
        # Create dictionary of performance data and print out results
        self._output_results(df=self.df, monthly_data=self.monthly_data, 
                             reversal=False)

        return self


    def _raw_signal_data(
            self, df, entry_type, position_size, exit_amount, ma1, ma2, ma3, 
            ma4, entry_period, entry_oversold, entry_overbought, 
            entry_threshold, simple_ma, entry_acceleration_factor):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        entry_type : TYPE
            DESCRIPTION.
        position_size : TYPE
            DESCRIPTION.
        exit_amount : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.

        """
        # Generate entry signals
        df, start, df['raw_td_signal'] = self._entry_signal(
            df=df, entry_type=entry_type, ma1=ma1, ma2=ma2, ma3=ma3, ma4=ma4,
            entry_period=entry_period, entry_oversold=entry_oversold, 
            entry_overbought=entry_overbought, entry_threshold=entry_threshold, 
            simple_ma=simple_ma, 
            entry_acceleration_factor=entry_acceleration_factor)

        # Calculate initial position info        
        df['raw_start_of_day_position'], df['raw_td_action'], \
            df['raw_end_of_day_position'] = self._positions_and_trade_actions(
                df=df, signal=df['raw_td_signal'], start=start, 
                position_size=position_size)
        
        # Generate trade numbers    
        df['raw_td_number'] = self._trade_numbers(
            df=df, end_of_day_position=df['raw_end_of_day_position'], 
            start=start)

        # Generate initial trade prices
        df['raw_td_entry_price'], df['raw_td_exit_price'], \
            df['raw_td_high_price'], df['raw_td_low_price'], \
                df['raw_td_close_high_price'], \
                    df['raw_td_close_low_price'] = self._trade_prices(
                        df=df, trade_number=df['raw_td_number'])
            
        return df, start    


    @classmethod
    def _exit_and_stop_signals(
            cls, df, position_size, exit_type, exit_amount, exit_period, 
            stop_type, stop_amount, stop_period, exit_threshold, exit_oversold, 
            exit_overbought, exit_acceleration_factor, sip_price):
                
        # Generate the exit signals
        df, df['exit_signal'] = cls._exit_signal(
            df=df, position_size=position_size, exit_amount=exit_amount, 
            exit_type=exit_type, exit_period=exit_period, 
            exit_threshold=exit_threshold, trade_number=df['raw_td_number'], 
            end_of_day_position=df['raw_end_of_day_position'], 
            exit_oversold=exit_oversold, exit_overbought=exit_overbought,
            exit_acceleration_factor=exit_acceleration_factor, 
            sip_price=sip_price)    
       
        # Generate the stop signals
        df, df['stop_signal'] = cls._stop_signal(
            df=df, stop_type=stop_type, stop_period=stop_period, 
            stop_amount=stop_amount, position_size=position_size, 
            trade_number=df['raw_td_number'], 
            end_of_day_position=df['raw_end_of_day_position'])
                    
        return df    

    
    @staticmethod
    def _positions_and_trade_actions(df, signal, start, position_size):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        signal : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        position_size : TYPE
            DESCRIPTION.

        Returns
        -------
        start_of_day_position : TYPE
            DESCRIPTION.
        trade_action : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.

        """
        # Extract the trade signal from the OHLC Data
        eod_trade_signal = np.array(signal)
 
        # Create empty arrays to store data       
        start_of_day_position = np.array([0]*len(df))
        trade_action = np.array([0]*len(df))
        end_of_day_position = np.array([0]*len(df))    
       
        # For each valid row in the data
        for row in range(start + 1, len(df)):
            
            # The start of day position is equal to the close of day position
            # of the previous day
            start_of_day_position[row] = end_of_day_position[row-1]
            
            # The trade action is the previous days trade signal multiplied by 
            # the position size
            trade_action[row] = eod_trade_signal[row-1] * position_size
            
            # The end of day position is the start of day position plus any 
            # trade action
            end_of_day_position[row] = (start_of_day_position[row] 
                                        + trade_action[row])    
        
        return start_of_day_position, trade_action, end_of_day_position 
    

    @staticmethod
    def _trade_numbers(df, end_of_day_position, start):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.

        Returns
        -------
        trade_number : TYPE
            DESCRIPTION.

        """
        # Extract the end of day position from the OHLC Data
        end_of_day_position = np.array(end_of_day_position)    
    
        # Create numpy array of zeros to store trade numbers
        trade_number = np.array([0]*len(df))
        
        # Set initial trade count to zero
        trade_count = 0
    
        for row in range(start + 1, len(df)):        
            # If today's position is zero
            if end_of_day_position[row] == 0:
                
                # If yesterday's position is zero
                if end_of_day_position[row - 1] == 0:
                    
                    # There is no open trade so set trade number to zero
                    trade_number[row] = 0
                
                # If yesterday's position is not zero
                else:
                    
                    # Set the trade number to the current trade count
                    trade_number[row] = trade_count
            
            # If today's position is the same as yesterday        
            elif end_of_day_position[row] == end_of_day_position[row - 1]:
                
                # Set the trade number to yesterdays trade number
                trade_number[row] = trade_number[row - 1]
            
            # If today's position is non-zero and different from yesterday    
            else:
                
                # Increase trade count by one for a new trade
                trade_count += 1
                
                # Set the trade number to the current trade count
                trade_number[row] = trade_count
        
        return trade_number


    @staticmethod
    def _trade_prices(df, trade_number):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.

        Returns
        -------
        trade_entry_price : TYPE
            DESCRIPTION.
        trade_exit_price : TYPE
            DESCRIPTION.
        trade_high_price : TYPE
            DESCRIPTION.
        trade_low_price : TYPE
            DESCRIPTION.
        trade_close_high_price : TYPE
            DESCRIPTION.
        trade_close_low_price : TYPE
            DESCRIPTION.

        """        
        # Initialize price arrays with zeroes
        trade_entry_price = np.array([0.0]*len(df))
        trade_exit_price = np.array([0.0]*len(df))
        trade_high_price = np.array([0.0]*len(df))
        trade_low_price = np.array([0.0]*len(df))
        trade_close_high_price = np.array([0.0]*len(df)) 
        trade_close_low_price = np.array([0.0]*len(df))
        
        # For each row in the DataFrame
        for row in range(len(df)):
            
            # Get the currenct trade number
            trade_num = trade_number[row]
            
            # Get the index location of the trade entry date 
            trade_first_row = df.index.get_loc(
                df[trade_number==trade_num].index[0])
            
            # Get the number of days since trade entry
            trade_row_num = row - trade_first_row
            
            # If there is no current trade, set the fields to zero
            if trade_num == 0:
                trade_entry_price[row] = 0
                trade_high_price[row] = 0
                trade_low_price[row] = 0 
            
            # Otherwise
            else:
                # Set the trade entry price to the opening price on the first 
                # trade date
                trade_entry_price[row] = df[
                    trade_number==trade_num].iloc[0]['Open']
                
                # Set the trade exit price to the opening price on the last 
                # trade date
                trade_exit_price[row] = df[
                    trade_number==trade_num].iloc[-1]['Open']
                
                # Calculate the maximum close price during the trade
                trade_close_high_price[row] = max(
                    df[trade_number==trade_num].iloc[
                        0:trade_row_num+1]['Close'])
                
                # Calculate the minimum close price during the trade
                trade_close_low_price[row] = min(
                    df[trade_number==trade_num].iloc[
                        0:trade_row_num+1]['Close'])
                
                # Calculate the maximum high price during the trade
                trade_high_price[row] = max(
                    df[trade_number==trade_num].iloc[
                        0:trade_row_num+1]['High'])
                
                # Calculate the minimum low price during the trade
                trade_low_price[row] = min(
                    df[trade_number==trade_num].iloc[0:trade_row_num+1]['Low'])
    
        return trade_entry_price, trade_exit_price, trade_high_price, \
            trade_low_price, trade_close_high_price, trade_close_low_price
        

    @staticmethod    
    def _position_values(
            df, trade_entry_price, end_of_day_position, trade_high_price, 
            trade_close_high_price, trade_low_price, trade_close_low_price, 
            trade_number):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        trade_entry_price : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.
        trade_high_price : TYPE
            DESCRIPTION.
        trade_close_high_price : TYPE
            DESCRIPTION.
        trade_low_price : TYPE
            DESCRIPTION.
        trade_close_low_price : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.

        Returns
        -------
        initial_position_value : TYPE
            DESCRIPTION.
        current_position_value : TYPE
            DESCRIPTION.
        max_trade_position_value : TYPE
            DESCRIPTION.
        max_trade_close_position_value : TYPE
            DESCRIPTION.
        min_trade_position_value : TYPE
            DESCRIPTION.
        min_trade_close_position_value : TYPE
            DESCRIPTION.

        """

        initial_position_value = trade_entry_price * end_of_day_position
        current_position_value = df['Close'] * end_of_day_position
        max_trade_position_value = trade_high_price * end_of_day_position
        max_trade_close_position_value = (trade_close_high_price 
                                          * end_of_day_position)
        min_trade_position_value = trade_low_price * end_of_day_position
        min_trade_close_position_value = (trade_close_low_price 
                                          * end_of_day_position)
        
        for row in range(1, len(df)):
            if (end_of_day_position[row] == 0) and (
                    trade_number[row] == trade_number[row-1]):
                
                initial_position_value[row] = initial_position_value[row-1]
                current_position_value[row] = (df['Open'][row] 
                                               * end_of_day_position[row-1])
                max_trade_position_value[row] = max_trade_position_value[row-1]
                max_trade_close_position_value[
                    row] = max_trade_close_position_value[row-1]
                min_trade_position_value[row] = min_trade_position_value[row-1]
                min_trade_close_position_value[
                    row] = min_trade_close_position_value[row-1]
           
        return initial_position_value, current_position_value, \
            max_trade_position_value, max_trade_close_position_value, \
                min_trade_position_value, min_trade_close_position_value


    @staticmethod
    def _pnl_targets(
            df, dollar_amount, position_size, trade_number, 
            end_of_day_position, trade_entry_price, trade_high_price, 
            trade_low_price, trade_close_high_price, trade_close_low_price):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        exit_amount : TYPE
            DESCRIPTION.
        position_size : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.
        trade_entry_price : TYPE
            DESCRIPTION.
        trade_high_price : TYPE
            DESCRIPTION.
        trade_low_price : TYPE
            DESCRIPTION.
        trade_close_high_price : TYPE
            DESCRIPTION.
        trade_close_low_price : TYPE
            DESCRIPTION.

        Returns
        -------
        profit_target : TYPE
            DESCRIPTION.
        initial_dollar_loss : TYPE
            DESCRIPTION.
        trailing_close : TYPE
            DESCRIPTION.
        trailing_high_low : TYPE
            DESCRIPTION.

        """

        trade_target = np.round(dollar_amount / position_size, 2)

        profit_target = np.array([0.0]*len(df))
        initial_dollar_loss = np.array([0.0]*len(df))
        trailing_close = np.array([0.0]*len(df))
        trailing_high_low = np.array([0.0]*len(df))
        
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                if end_of_day_position[row] > 0:
                    profit_target[row] = (trade_entry_price[row] 
                                          + trade_target)
                    initial_dollar_loss[row] = (
                        trade_entry_price[row] - trade_target)
                    trailing_close[row] = (
                        trade_close_high_price[row] - trade_target)
                    trailing_high_low[row] = (
                        trade_high_price[row] - trade_target)
                else:
                    profit_target[row] = (trade_entry_price[row] 
                                          - trade_target)
                    initial_dollar_loss[row] = (
                        trade_entry_price[row] + trade_target)
                    trailing_close[row] = (
                        trade_close_low_price[row] + trade_target)
                    
                    trailing_high_low[row] = (
                        trade_low_price[row] + trade_target)
        
        return profit_target, initial_dollar_loss, trailing_close, \
            trailing_high_low            


    @staticmethod
    def _signal_combine(
            df, start, raw_trade_signal, end_of_day_position, trade_number, 
            exit_signal, stop_signal):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        raw_trade_signal : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.
        exit_signals : TYPE
            DESCRIPTION.

        Returns
        -------
        combined_signal : TYPE
            DESCRIPTION.

        """
        combined_signal = np.array([0.0]*len(df))
        all_signals = pd.concat(
            [raw_trade_signal, exit_signal, stop_signal], axis=1)
        flag = True
        for row in range(start, len(df)):        
            trade_num = trade_number[row]
            trade_last_row = df.index.get_loc(
                df[trade_number==trade_num].index[-1])
            
            if ((raw_trade_signal[row] != 0) and (
                    end_of_day_position[row] == 0)): 
                combined_signal[row] = raw_trade_signal[row]
            
            else:
                if trade_number[row] != 0:
                    if flag:
                        if end_of_day_position[row] > 0:
                        
                            combined_signal[row] = int(
                                min(all_signals.iloc[row]))
                        
                        elif end_of_day_position[row] < 0:
                            
                            combined_signal[row] = int(
                                max(all_signals.iloc[row]))        
                        
                        else:
                            combined_signal[row] = raw_trade_signal[row]            
                    
                        if (combined_signal[row] != 0):
                            flag = False
        
                    elif (trade_number[row] != trade_number[row-1]):
                        
                        combined_signal[row] = raw_trade_signal[row]
                        flag=True
                    
                    elif row == trade_last_row and abs(
                            raw_trade_signal[row]) > 1:
                        
                        combined_signal[row] = int(
                            raw_trade_signal[row] / 2)
                        flag=True
                    
                    else:
                        combined_signal[row] = 0
   
        return combined_signal


    def _position_size(self, df, equity):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        equity : TYPE
            DESCRIPTION.

        Returns
        -------
        position_size : TYPE
            DESCRIPTION.

        """
        # Set the position size to the number of shares that can be bought with 
        # the initial equity 
        
        # Set the number of units to use 75% of starting equity
        units = (equity / df['Close'].iloc[0]) * 0.75
        spx_units = (equity / self.spx['Close'].iloc[0]) * 0.75
        
        # If the number of units would be greater than 50 then make the 
        # position size a multiple of 10        
        if df['Close'].iloc[0] < equity / 50:
            position_size = math.floor(
                int(units / 10)) * 10
        
        # Otherwise take the most units that can be afforded 
        else:
            position_size = math.floor(units)
                
        self.spx_position_size = spx_units
        self.position_size = position_size
        
        return position_size
        

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
        
        if kwargs['strategy'] == '3ma':
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
            
        elif kwargs['strategy'] == '4ma':
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
            raise ValueError("Please enter a valid strategy")
       
        return df, strategy_label
    
    
    def _strategy_labels(self, **kwargs):
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
        
        # Simple or Exponential Moving Average label
        if kwargs['simple_ma']:
            ma_type_label = 'S'
        else:
            ma_type_label = 'E'
        
        
        # Entry labels
        
        # Double Moving Average Crossover
        if kwargs['entry_type'] == '2ma':
            
            # Set the entry label
            entry_label = (str(kwargs['ma1'])
                           +'-, '
                           +str(kwargs['ma2'])
                           +'-day :'
                           +self.df_dict['df_entry_signal_labels'][
                               kwargs['entry_type']][0]
                           +ma_type_label
                           +self.df_dict['df_entry_signal_labels'][
                               kwargs['entry_type']][1])
        
        
        # Triple Moving Average Crossover                                
        elif kwargs['entry_type'] == '3ma':
            
            # Set the entry label
            entry_label = (str(kwargs['ma1'])
                           +'-, '
                           +str(kwargs['ma2'])
                           +'-, '
                           +str(kwargs['ma3'])
                           +'-day : '
                           +self.df_dict['df_entry_signal_labels'][
                               kwargs['entry_type']][0]
                           +ma_type_label
                           +self.df_dict['df_entry_signal_labels'][
                               kwargs['entry_type']][1])
        
        
        # Quad Moving Average Crossover    
        elif kwargs['entry_type'] == '4ma':
            
            # Set the entry label
            entry_label = (str(kwargs['ma1'])
                           +'-, '
                           +str(kwargs['ma2'])
                           +'-, '
                           +str(kwargs['ma3'])
                           +'-, '
                           +str(kwargs['ma4'])
                           +'-day : '
                           +self.df_dict['df_entry_signal_labels'][
                               kwargs['entry_type']][0]
                           +ma_type_label
                           +self.df_dict['df_entry_signal_labels'][
                               kwargs['entry_type']][1])    
        
        
        # Parabolic SAR 
        elif kwargs['entry_type'] == 'sar':
            
            # Set the entry label
            entry_label = (str(kwargs['entry_period'])
                          +'-day '
                          +str(np.round(kwargs[
                              'entry_acceleration_factor'] * 100, 1))
                          +'% AF '
                          +self.df_dict['df_entry_signal_labels'][
                              kwargs['entry_type']])
            
        
        # Channel Breakout 
        elif kwargs['entry_type'] == 'channel_breakout':
            
            # Set the entry label
            entry_label = (str(kwargs['entry_period'])
                           +'-day : '
                           +self.df_dict['df_entry_signal_labels'][
                               kwargs['entry_type']])
        
        
        # Stochastic Crossover, Stochastic Pop, Stochastic Over Under and 
        # Relative Strength Index   
        elif kwargs['entry_type'] in ['stoch_cross', 'stoch_over_under', 
                                      'stoch_pop', 'rsi']:
            
            # Set the entry label
            entry_label = (str(kwargs['entry_period'])
                           +'-day '
                           +str(kwargs['entry_overbought'])
                           +'-'
                           +str(kwargs['entry_oversold'])
                           +' : '
                           +self.df_dict['df_entry_signal_labels'][
                               kwargs['entry_type']])
        
        
        # Commodity Channel Index, Momentum and Volatility
        elif kwargs['entry_type'] in ['cci', 'momentum', 'volatility']:
            
            # Set the entry label
            entry_label = (str(kwargs['entry_period'])
                           +'-day '
                           +str(int(kwargs['entry_threshold']*100))
                           +'% : '
                           +self.df_dict['df_entry_signal_labels'][
                               kwargs['entry_type']])
        
        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid entry type")
            
        
        # Exit labels
        
        # Parabolic SAR
        if kwargs['exit_type'] == 'sar':
            
            # Set the exit label         
            exit_label = (str(kwargs['exit_period'])
                          +'-day '
                          +str(np.round(kwargs[
                              'exit_acceleration_factor'] * 100, 1))
                          +'% AF '
                          +self.df_dict['df_exit_signal_labels'][
                              kwargs['exit_type']])
        
        
        # Stochastic Crossover and Trailing Relative Strength Index
        elif kwargs['exit_type'] in ['stoch_cross', 'rsi_trail']:
            
            # Set the exit label         
            exit_label = (str(kwargs['exit_period'])
                           +'-day '
                           +str(kwargs['exit_overbought'])
                           +'-'
                           +str(kwargs['exit_oversold'])
                           +' : '
                           +self.df_dict['df_exit_signal_labels'][
                               kwargs['exit_type']]) 
        
        
        # Volatility
        elif kwargs['exit_type'] in ['volatility']:
            
            # Set the exit label         
            exit_label = (str(kwargs['exit_period'])
                          +'-day '
                          +str(int(kwargs['exit_threshold']*100))
                          +'% : '
                          +self.df_dict['df_exit_signal_labels'][
                              kwargs['exit_type']])
            
            
        # Trailing Stop and Profit Target
        elif kwargs['exit_type'] in ['trailing_stop', 'profit_target']:
            
            # Set the exit label         
            exit_label = ('$'
                          +str(int(kwargs['exit_amount']))
                          +' '                          
                          +self.df_dict['df_exit_signal_labels'][
                               kwargs['exit_type']])
        

        # Support/Resistance, Key Reversal Day and n-Day Range
        elif kwargs['exit_type'] in ['sup_res', 'key_reversal', 'nday_range']:
            
            # Set the exit label         
            exit_label = (str(kwargs['exit_period'])
                          +'-day '
                          +self.df_dict['df_exit_signal_labels'][
                              kwargs['exit_type']])
        
        
        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid exit type")
        
        
        # Stop labels
        
        # Initial Dollar, Breakeven, Trailing Close and Trailing High Low        
        if kwargs['stop_type'] in ['initial_dollar', 'breakeven', 
                                   'trail_close', 'trail_high_low']:    

            # Set the stop label         
            stop_label = ('$'
                          +str(int(kwargs['stop_amount']))
                          +' '                          
                          +self.df_dict['df_stop_signal_labels'][
                               kwargs['stop_type']])           


        # Support / Resistance and Immediate Profit
        elif kwargs['stop_type'] in ['sup_res', 'immediate_profit']:    

            # Set the stop label         
            stop_label = (str(kwargs['stop_period'])
                          +'-day '
                          +self.df_dict['df_stop_signal_labels'][
                              kwargs['stop_type']])


        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid stop type")

        #self.strategy_label = entry_label
       
        return entry_label, exit_label, stop_label
            
    
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
    def create_base_data(
            cls, ticker=None, ccy_1=None, ccy_2=None, start_date=None, 
            end_date=None, lookback=None, source=None, asset_type=None):
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
        
        elif source == 'alpha':
            df = cls._return_alphavantage_data(
                ccy_1=ccy_1, ccy_2=ccy_2, ticker=ticker, asset_type=asset_type, 
                lookback=lookback)
            
            return df
                
        # Otherwise return error message
        else:
            print('Select a data source from yahoo, norgate or alpha')
            
    
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


    @staticmethod
    def _return_alphavantage_data(
            ccy_1=None, ccy_2=None, ticker=None, asset_type=None, 
            lookback=None):

        api_key = os.getenv('ALPHAVANTAGE_API_KEY')

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
        
        # Trim data to length of the specified lookback
        if lookback < len(df):
            df = df[-lookback:]
        
        return df


    @classmethod
    def _entry_signal(
            cls, df=None, entry_type=None, ma1=None, ma2=None, ma3=None, 
            ma4=None, simple_ma=None, entry_period=None, entry_oversold=None, 
            entry_overbought=None, entry_threshold=None, 
            entry_acceleration_factor=None):
        
        if entry_type == '2ma':
            df, start, signal = cls._entry_double_ma_crossover(
                df=df, ma1=ma1, ma2=ma2, simple_ma=simple_ma)
        
        elif entry_type == '3ma':
            df, start, signal = cls._entry_triple_ma_crossover(
                df, ma1=ma1, ma2=ma2, ma3=ma3, simple_ma=simple_ma)
        
        elif entry_type == '4ma':
            df, start, signal = cls._entry_quad_ma_crossover(
                df, ma1=ma1, ma2=ma2, ma3=ma3, ma4=ma4, simple_ma=simple_ma)        
        
        elif entry_type == 'sar':
            df, start, signal = cls._entry_parabolic_sar(
                df=df, acceleration_factor=entry_acceleration_factor)
        
        elif entry_type == 'channel_breakout':
            df, start, signal = cls._entry_channel_breakout(
                df, time_period=entry_period)
        
        elif entry_type == 'stoch_cross':
            df, start, signal = cls._entry_stochastic_crossover(
                df, time_period=entry_period, oversold=entry_oversold, 
                overbought=entry_overbought)
        
        elif entry_type == 'stoch_over_under':
            df, start, signal = cls._entry_stochastic_over_under(
                df, time_period=entry_period, oversold=entry_oversold, 
                overbought=entry_overbought)
        
        elif entry_type == 'stoch_pop':
            df, start, signal = cls._entry_stochastic_pop(
                df, time_period=entry_period, oversold=entry_oversold, 
                overbought=entry_overbought)
        
        elif entry_type == 'rsi':
            df, start, signal = cls._entry_rsi(
                df, time_period=entry_period, oversold=entry_oversold, 
                overbought=entry_overbought)
        
        elif entry_type == 'cci':
            df, start, signal = cls._entry_commodity_channel_index(
                df, time_period=entry_period, threshold=entry_threshold)
        
        elif entry_type == 'momentum':
            df, start, signal = cls._entry_momentum(
                df, time_period=entry_period, threshold=entry_threshold)
        
        elif entry_type == 'volatility':
            df, start, signal = cls._entry_volatility(
                df, time_period=entry_period, threshold=entry_threshold)
            
        return df, start, signal    


    @classmethod
    def _exit_signal(cls, df, exit_type=None, exit_period=None, 
                     exit_amount=None, exit_threshold=None,
                     position_size=None, trade_number=None, 
                     end_of_day_position=None, exit_oversold=None, 
                     exit_overbought=None, exit_acceleration_factor=None, 
                     sip_price=None):
        
        # Generate profit targets / trailing stops        
        df['exit_profit_target'], df['exit_initial_dollar_loss'], \
            df['exit_trailing_close'], \
                df['exit_trailing_high_low'] = cls._pnl_targets(
                    df=df, dollar_amount=exit_amount, 
                    position_size=position_size,
                    trade_number=df['raw_td_number'], 
                    end_of_day_position=df['raw_end_of_day_position'], 
                    trade_entry_price=df['raw_td_entry_price'], 
                    trade_high_price=df['raw_td_high_price'], 
                    trade_close_high_price=df['raw_td_close_high_price'], 
                    trade_low_price=df['raw_td_low_price'], 
                    trade_close_low_price=df['raw_td_close_low_price']) 
        
        if exit_type == 'sar':
            df, exit = cls._exit_parabolic_sar(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position, 
                time_period=exit_period, 
                acceleration_factor=exit_acceleration_factor, 
                sip_price=sip_price)
        
        elif exit_type == 'sup_res':
            df, exit = cls._exit_support_resistance(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position,
                time_period=exit_period)        
        
        elif exit_type == 'rsi_trail':
            df, exit = cls._exit_rsi_trail(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position, 
                time_period=exit_period, oversold=exit_oversold, 
                overbought=exit_overbought)        
        
        elif exit_type == 'key_reversal':
            df, exit = cls._exit_key_reversal(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position,
                time_period=exit_period)        
        
        elif exit_type == 'trailing_stop':
            df, exit = cls._exit_dollar(
                df=df, trigger_value=df['exit_trailing_close'], 
                exit_level='trail_close', trade_number=trade_number, 
                end_of_day_position=end_of_day_position)        
        
        elif exit_type == 'volatility':
            df, exit = cls._exit_volatility(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position,
                time_period=exit_period, threshold=exit_threshold)        
        
        elif exit_type == 'stoch_cross':
            df, exit = cls._exit_stochastic_crossover(
                df=df, time_period=exit_period, trade_number=trade_number, 
                end_of_day_position=end_of_day_position)        
        
        elif exit_type == 'profit_target':
            df, exit = cls._exit_dollar(
                df=df, trigger_value=df['exit_profit_target'], 
                exit_level='profit_target', trade_number=trade_number, 
                end_of_day_position=end_of_day_position) 
        
        elif exit_type == 'nday_range':    
            df, exit = cls._exit_nday_range(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position, 
                time_period=exit_period)    
        
        elif exit_type == 'random':
            df, exit = cls._exit_random(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position)

        return df, exit


    @classmethod
    def _stop_signal(
            cls, df, stop_type=None, stop_period=None, stop_amount=None, 
            position_size=None, trade_number=None, end_of_day_position=None):
        
        # Generate profit targets / trailing stops        
        df['stop_profit_target'], df['stop_initial_dollar_loss'], \
            df['stop_trailing_close'], \
                df['trailing_high_low'] = cls._pnl_targets(
                    df=df, dollar_amount=stop_amount, 
                    position_size=position_size,
                    trade_number=df['raw_td_number'], 
                    end_of_day_position=df['raw_end_of_day_position'], 
                    trade_entry_price=df['raw_td_entry_price'], 
                    trade_high_price=df['raw_td_high_price'], 
                    trade_close_high_price=df['raw_td_close_high_price'], 
                    trade_low_price=df['raw_td_low_price'], 
                    trade_close_low_price=df['raw_td_close_low_price'])        

        
        if stop_type == 'initial_dollar':
            df, stop = cls._exit_dollar(
                df=df, trigger_value=df['stop_initial_dollar_loss'], 
                exit_level='initial', trade_number=trade_number, 
                end_of_day_position=end_of_day_position)    
        
        elif stop_type == 'sup_res':
            df, stop = cls._exit_support_resistance(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position,
                time_period=stop_period)
        
        elif stop_type == 'immediate_profit':
            df, stop = cls._exit_immediate_profit(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position,
                time_period=stop_period)
        
        elif stop_type == 'breakeven':
            df, stop = cls._exit_dollar(
                df=df, trigger_value=df['stop_profit_target'], 
                exit_level='breakeven', trade_number=trade_number, 
                end_of_day_position=end_of_day_position, 
                trade_high_price=df['raw_td_high_price'], 
                trade_low_price=df['raw_td_low_price'])
        
        elif stop_type == 'trail_close':
            df, stop = cls._exit_dollar(
                df=df, trigger_value=df['stop_trailing_close'], 
                exit_level='trail_close', trade_number=trade_number, 
                end_of_day_position=end_of_day_position)
        
        elif stop_type == 'trail_high_low':
            df, stop = cls._exit_dollar(
                df=df, trigger_value=df['trailing_high_low'], 
                exit_level='trail_high_low', trade_number=trade_number, 
                end_of_day_position=end_of_day_position) 
                    
        return df, stop
    

    @staticmethod
    def _entry_double_ma_crossover(df, ma1, ma2, simple_ma):
        """
        Entry signal for Moving Average Crossover strategy

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        ma1 : Int
            The faster moving average. The default is 9.
        ma2 : Int
            The slower moving average. The default is 18.

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.

        """
        if simple_ma:
            # Create short and long simple moving averages  
            ma_1 = np.array(df['Close'].rolling(ma1).mean())
            ma_2 = np.array(df['Close'].rolling(ma2).mean())
        
        else:
            ma_1 = Indicators.EMA(input_series=df['Close'], time_period=ma1)
            ma_2 = Indicators.EMA(input_series=df['Close'], time_period=ma2)
        
        # Create start point from first valid number
        start = np.where(~np.isnan(ma_2))[0][0]
            
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(ma_2))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(ma_2))
        
        # for each row in the DataFrame after the longest MA has started
        for row in range(start, len(ma_2)):
            
            # If the short MA crosses above the long MA 
            if ma_1[row] > ma_2[row] and ma_1[row-1] < ma_2[row-1]:
                   
                # Set the position signal to long
                position_signal[row] = 1
                
                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]
                
            # If the short MA crosses below the long MA 
            elif ma_1[row] < ma_2[row] and ma_1[row-1] > ma_2[row-1]:
    
                # Set the position signal to short
                position_signal[row] = -1
    
                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]
    
            # Otherwise, take no action
            else:        
                position_signal[row] = position_signal[row-1]
                trade_signal[row] = 0                
    
        df['ma_1'] = ma_1
        df['ma_2'] = ma_2
        
        return df, start, trade_signal


    @staticmethod
    def _entry_triple_ma_crossover(df, ma1, ma2, ma3, simple_ma):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        ma1 : TYPE, optional
            DESCRIPTION. The default is 4.
        ma2 : TYPE, optional
            DESCRIPTION. The default is 9.
        ma3 : TYPE, optional
            DESCRIPTION. The default is 18.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.

        """
        # Create fast, medium and slow simple moving averages  
        if simple_ma:
            ma_1 = np.array(df['Close'].rolling(ma1).mean())
            ma_2 = np.array(df['Close'].rolling(ma2).mean())
            ma_3 = np.array(df['Close'].rolling(ma3).mean())

        else:
            ma_1 = Indicators.EMA(input_series=df['Close'], time_period=ma1)
            ma_2 = Indicators.EMA(input_series=df['Close'], time_period=ma2)
            ma_3 = Indicators.EMA(input_series=df['Close'], time_period=ma3)
            
        # Create start point from first valid number
        start = np.where(~np.isnan(ma_3))[0][0]
            
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(ma_3))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(ma_3))
        
        # for each row in the DataFrame after the longest MA has started
        for row in range(start, len(ma_3)):
            
            # If the medium MA is above the slow MA 
            if ma_2[row] > ma_3[row]:
                
                # If the fast MA crosses above the medium MA
                if (ma_1[row] > ma_2[row] 
                    and ma_1[row-1] < ma_2[row-1]):
                    
                    # Set the position signal to long
                    position_signal[row] = 1
                    
                    # Signal to go long
                    trade_signal[row] = 1 - position_signal[row-1]
                
                # If the fast MA crosses below the medium MA
                elif (ma_1[row] < ma_2[row] 
                      and ma_1[row-1] > ma_2[row-1]):
                    
                    # If currently long
                    if position_signal[row-1] == 1:
    
                        # Set the position signal to flat
                        position_signal[row] = 0                    
                        
                        # Signal to close out long
                        trade_signal[row] = -1
                          
                # Otherwise, take no action
                else:                
                    trade_signal[row] = 0
                    position_signal[row] = position_signal[row-1]
            
            # If the medium MA is below the slow MA
            else:
                
                # If the fast MA crosses below the medium MA
                if (ma_1[row] < ma_2[row] 
                    and ma_1[row-1] > ma_2[row-1]):
                    
                    # Set the position signal to short
                    position_signal[row] = -1
    
                    # Signal to go short
                    trade_signal[row] = -1 - position_signal[row-1] 
                
                # If the fast MA crosses above the medium MA
                elif (ma_1[row] > ma_2[row] 
                    and ma_1[row-1] < ma_2[row-1]):
                    
                    # If currently short
                    if position_signal[row-1] == -1:
                        
                        # Set the position to flat
                        position_signal[row] = 0
    
                        # Signal to close out short
                        trade_signal[row] = 1
              
                # Otherwise, take no action
                else:                
                    trade_signal[row] = 0
                    position_signal[row] = position_signal[row-1]                
    
        # Assign the series to the OHLC data 
        df['ma_1'] = ma_1
        df['ma_2'] = ma_2
        df['ma_3'] = ma_3
    
        return df, start, trade_signal
    
    
    @staticmethod
    def _entry_quad_ma_crossover(df, ma1, ma2, ma3, ma4, simple_ma):
       
        """
        Create trading signals for Quad MA strategy
    
        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        ma1 : Int, optional
            The fastest of the 4 moving averages. The default is 5 periods.
        ma2 : Int, optional
            The 2nd fastest of the 4 moving averages. The default is 12 
            periods.
        ma3 : Int, optional
            The second slowest of the 4 moving averages. The default is 20 
            periods.
        ma4 : Int, optional
            The slowest of the 4 moving averages. The default is 40 periods.
       
        Returns
        -------
        ma_1 : Series
            The series of the fastest of the 4 moving averages.
        ma_2 : Series
            The series of the 2nd fastest of the 4 moving averages.
        ma_3 : Series
            The series of second slowest of the 4 moving averages.
        ma_4 : Series
            The series of slowest of the 4 moving averages.
        start : Int
            The first valid row to start calculating signals from.
        position_signal : Series
            Series of whether to be long, short or neutral on the following 
            date.
        trade_signal : Series
            Series indicating when buy or sell decisions should be made the 
            following day.
        """
        
        # Create the 4 simple moving averages
        if simple_ma:
            ma_1 = df['Close'].rolling(ma1).mean()
            ma_2 = df['Close'].rolling(ma2).mean()
            ma_3 = df['Close'].rolling(ma3).mean()
            ma_4 = df['Close'].rolling(ma4).mean()
        
        else:
            ma_1 = Indicators.EMA(input_series=df['Close'], time_period=ma1)
            ma_2 = Indicators.EMA(input_series=df['Close'], time_period=ma2)
            ma_3 = Indicators.EMA(input_series=df['Close'], time_period=ma3)
            ma_4 = Indicators.EMA(input_series=df['Close'], time_period=ma4)            
        
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(ma_4))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(ma_4))
        
        # Create start point from first valid number
        start = np.where(~np.isnan(ma_4))[0][0]
    
        # for each row in the DataFrame after the longest MA has started
        for row in range(start + 1, len(ma_4)):
            
            # If the second slowest MA is above the slowest MA
            if ma_3[row] > ma_4[row]:
                
                # If the fastest MA crosses above the second fastest MA
                if ma_1[row] > ma_2[row] and ma_1[row - 1] < ma_2[row - 1]:
                    
                    # Set the position signal to long
                    position_signal[row] = 1
                    
                    # Signal to go long
                    trade_signal[row] = 1 - position_signal[row-1]
                
                # If the fastest MA crosses below the second fastest MA
                elif ma_1[row] < ma_2[row] and ma_1[row - 1] > ma_2[row - 1]:
                    
                    # If there is a position on
                    if position_signal[row-1] != 0:
                    
                        # Set the position signal to flat
                        position_signal[row] = 0                    
                            
                        # Signal to close out long
                        trade_signal[row] = -1
                    
                # Otherwise, take no action
                else:                
                    trade_signal[row] = 0
                    position_signal[row] = position_signal[row-1]
                    
            
            # If the second slowest MA is below the slowest MA
            else:
    
                # If the fastest MA crosses below the second fastest MA
                if ma_1[row] < ma_2[row] and ma_1[row - 1] > ma_2[row - 1]:
    
                    # Set the position signal to short
                    position_signal[row] = -1
    
                    # Signal to go short
                    trade_signal[row] = -1 - position_signal[row-1] 
    
                # If the fastest MA crosses above the second fastest MA
                elif ma_1[row] > ma_2[row] and ma_1[row - 1] < ma_2[row - 1]:
    
                    # If there is a position on
                    if position_signal[row-1] != 0:

                        # Set the position to flat
                        position_signal[row] = 0
    
                        # Signal to close out short
                        trade_signal[row] = 1
                    
                # Otherwise, take no action
                else:                
                    trade_signal[row] = 0
                    position_signal[row] = position_signal[row-1]    
    
        # Assign the series to the OHLC data 
        df['ma_1'] = ma_1
        df['ma_2'] = ma_2
        df['ma_3'] = ma_3
        df['ma_4'] = ma_4
    
        return df, start, trade_signal 


    @staticmethod
    def _entry_parabolic_sar(df, acceleration_factor):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        acceleration_factor : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        trade_signal : TYPE
            DESCRIPTION.

        """
        # Extract high, low and close series from the DataFrame
        high = df['High']
        low = df['Low']
        close = df['Close']        
        
        # Create empty arrays to store data    
        sar = np.array([0.0]*len(df))
        direction = np.array([0.0]*len(df))
        trade_signal = np.array([0.0]*len(df))
        ep = np.array([0.0]*len(df))
        af = np.array([0.0]*len(df))
        ep_sar_diff = np.array([0.0]*len(df))
        af_x_diff = np.array([0.0]*len(df))
    
        # Configure initial values
        initial_sip = 0.975 * close[0]
        sar[0] = initial_sip
        direction[0] = 1
        ep[0] = high[0]
        af[0] = 0.02
        ep_sar_diff[0] = abs(sar[0] - ep[0])
        af_x_diff[0] =  ep_sar_diff[0] * af[0]
        start = 1
        init_flag = True
    
        # Loop through, starting from the second row    
        for row in range(start, len(df)):
            
            # If the previous day was long
            if direction[row-1] == 1:
                
                # If the previous day's sar was greater than the previous day's 
                # low
                if sar[row-1] > low[row-1]:
                    
                    # If this is the starting trade
                    if init_flag:
                        
                        # Enter short
                        trade_signal[row-1] = -1
                        
                        # Set the flag to False
                        init_flag = False
                    
                    else:                        
                        # Close the long position and go short
                        trade_signal[row-1] = -2
                    
                    # Set the new sar to the previous trades extreme price
                    sar[row] = ep[row-1]
                    
                    # Switch the trade direction from long to short
                    direction[row] = -direction[row-1] 
                    
                # If the previous day's sar plus the acceleration factor 
                # multiplied by the difference between the extreme price and 
                # the sar is greater than the lowest low of the previous 2 days     
                elif (sar[row-1] + af_x_diff[row-1] 
                      > min(low[row-1], low[row-2])):
                    
                    # Set the sar to the lowest low of the previous 2 days
                    sar[row] = min(low[row-1], low[row-2])
                    
                    # Set the direction to the same as the previous day
                    direction[row] = direction[row-1]
                
                # Otherwise    
                else: 
                    # Set the sar to the previous day's sar plus the 
                    # acceleration factor multiplied by the difference between 
                    # the extreme price and the sar
                    sar[row] = sar[row-1] + af_x_diff[row-1]
                    
                    # Set the direction to the same as the previous day
                    direction[row] = direction[row-1]
            
            # Otherwise if the previous day was short
            else:
                # If the previous day's sar was less than the previous day's 
                # high
                if sar[row-1] < high[row-1]:
                    
                    # Close the short position and go long
                    trade_signal[row-1] = 2
                    
                    # Set the new sar to the previous trades extreme price
                    sar[row] = ep[row-1]
                    
                    # Switch the trade direction from short to long
                    direction[row] = -direction[row-1]    
                
                # If the previous day's sar less the acceleration factor 
                # multiplied by the difference between the extreme price and 
                # the sar is less than the highest high of the previous 2 days 
                elif (sar[row-1] - af_x_diff[row-1] 
                      < max(high[row-1], high[row-2])):
                    
                    # Set the sar to the highest high of the previous 2 days
                    sar[row] = max(high[row-1], high[row-2])    
                    
                    # Set the direction to the same as the previous day
                    direction[row] = direction[row-1]
                
                # Otherwise
                else:    
                    # Set the sar to the previous day's sar minus the 
                    # acceleration factor multiplied by the difference between 
                    # the extreme price and the sar
                    sar[row] = sar[row-1] - af_x_diff[row-1]
                    
                    # Set the direction to the same as the previous day
                    direction[row] = direction[row-1]
            
            # If the current trade direction is long
            if direction[row] == 1:
                
                # If the trade has just reversed direction
                if direction[row] != direction[row-1]:
                    
                    # Set the extreme price to the day's high
                    ep[row] = high[row]
                    
                    # Set the initial acceleration factor to the input value
                    af[row] = acceleration_factor
                
                # If the trade is the same as the previous day
                else:
                    
                    # Set the extreme price to the greater of the previous 
                    # day's extreme price and the current day's high
                    ep[row] = max(ep[row-1], high[row])
                    
                    # If the trade is making a new high
                    if ep[row] > ep[row-1]:
                        
                        # Increment the acceleration factor by the input value 
                        # to a max of 0.2
                        af[row] = min(af[row-1] + acceleration_factor, 0.2)
                    
                    # Otherwise
                    else:    
                        
                        # Set the acceleration factor to the same as the 
                        # previous day
                        af[row] = af[row-1]
            
            # Otherwise if the current trade direction is short
            else:
                
                # If the trade has just reversed direction
                if direction[row] != direction[row-1]:
                
                    # Set the extreme price to the day's low
                    ep[row] = low[row]
                    
                    # Set the initial acceleration factor to the input value
                    af[row] = acceleration_factor
                
                # If the trade is the same as the previous day    
                else:
                    
                    # Set the extreme price to the lesser of the previous day's 
                    # extreme price and the current day's low
                    ep[row] = min(ep[row-1], low[row])
                    
                    # If the trade is making a new low
                    if ep[row] < ep[row-1]:
                        
                        # Increment the acceleration factor by the input value 
                        # to a max of 0.2
                        af[row] = min(af[row-1] + acceleration_factor, 0.2)
                    
                    # Otherwise    
                    else:
                        
                        # Set the acceleration factor to the same as the 
                        # previous day
                        af[row] = af[row-1]
            
            # Calculate the absolute value of the difference between the 
            # extreme price and the sar     
            ep_sar_diff[row] = abs(sar[row] - ep[row])
            
            # Calculate the difference between the extreme price and the sar 
            # multiplied by the acceleration factor
            af_x_diff[row] =  ep_sar_diff[row] * af[row]
        
        df['sar_entry'] = sar
        df['direction_sar_entry'] = direction
        df['ep_sar_entry'] = ep
        df['af_sar_entry'] = af
        df['ep_sar_diff_entry'] = ep_sar_diff
        df['af_x_diff_entry'] = af_x_diff        
        
        return df, start, trade_signal
 

    @staticmethod
    def _entry_channel_breakout(df, time_period):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        time_period : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.

        """
        # Calculate rolling min and max closing prices based on time period
        rolling_high_close = df['Close'].rolling(time_period).max()
        rolling_low_close = df['Close'].rolling(time_period).min()
        
        # Create start point based on lookback window
        start = np.where(~np.isnan(rolling_high_close))[0][0]
            
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(rolling_high_close))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(rolling_high_close))
        
        # for each row in the DataFrame after the longest MA has started
        for row in range(start, len(rolling_high_close)):
            
            # If the price rises to equal or above the n-day high close 
            if df['Close'][row] >= rolling_high_close[row]:
                   
                # Set the position signal to long
                position_signal[row] = 1
                
                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]
                
            # If the price falls to equal or below the n-day low close 
            elif df['Close'][row] <= rolling_low_close[row]:
    
                # Set the position signal to short
                position_signal[row] = -1
    
                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]
    
            # Otherwise, take no action
            else:                
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]
    
        df['rolling_high_close_entry'] = rolling_high_close
        df['rolling_low_close_entry'] = rolling_low_close 
        
        return df, start, trade_signal
    
    
    @staticmethod
    def _entry_stochastic_crossover(df, time_period, oversold, overbought):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        time_period : TYPE
            DESCRIPTION.
        lower : TYPE
            DESCRIPTION.
        upper : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        trade_signal : TYPE
            DESCRIPTION.

        """
        # Create Stochastics for the specified time period
        slow_k, slow_d = Indicators.stochastic(
            df['High'], df['Low'], df['Close'], fast_k_period=time_period)
    
        # Create start point based on slow d
        start = np.where(~np.isnan(slow_d))[0][0]
            
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(slow_d))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(slow_d))
        
        # for each row in the DataFrame after the slow d has started
        for row in range(start, len(slow_d)):
            
            # If the slow k crosses above the slow d from below having been 
            # below the lower level 
            if (slow_k[row] > slow_d[row] 
                and slow_k[row-1] < slow_d[row-1] 
                and slow_k[row] < oversold):
                
                # Set the position signal to long
                position_signal[row] = 1
                
                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]
            
            # If the slow k crosses below the slow d from above having been 
            # above the upper level
            elif (slow_k[row] < slow_d[row] 
                and slow_k[row-1] > slow_d[row-1] 
                and slow_k[row] > overbought):
                
                # Set the position signal to short
                position_signal[row] = -1
    
                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]
            
            # Otherwise, take no action
            else:                
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]
    
        df['slow_k_entry'] = slow_k
        df['slow_d_entry'] = slow_d        
   
        return df, start, trade_signal


    @staticmethod
    def _entry_stochastic_over_under(df, time_period, oversold, overbought):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        time_period : TYPE
            DESCRIPTION.
        lower : TYPE
            DESCRIPTION.
        upper : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        trade_signal : TYPE
            DESCRIPTION.

        """
        # Create Stochastics for the specified time period
        slow_k, slow_d = Indicators.stochastic(
            df['High'], df['Low'], df['Close'], fast_k_period=time_period)
    
        # Create start point based on slow d
        start = np.where(~np.isnan(slow_d))[0][0]
            
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(slow_d))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(slow_d))
        
        # for each row in the DataFrame after the slow d has started
        for row in range(start, len(slow_d)):
            
            # If both slow k and slow d cross above the lower level from below 
            if (min(slow_k[row], slow_d[row]) > oversold 
                and min(slow_k[row-1], slow_d[row-1]) < oversold):
                
                # Set the position signal to long
                position_signal[row] = 1
                
                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]
                
            # If both slow k and slow d cross above the upper level from below 
            elif (min(slow_k[row], slow_d[row]) > overbought 
                and min(slow_k[row-1], slow_d[row-1]) < overbought):
                
                # Set the position signal to flat
                position_signal[row] = 0
                
                # Signal to go flat
                trade_signal[row] = 0 - position_signal[row-1]    
            
            # If both slow k and slow d cross below the upper level from above 
            elif (max(slow_k[row], slow_d[row]) < overbought 
                and max(slow_k[row-1], slow_d[row-1]) > overbought):
                
                # Set the position signal to short
                position_signal[row] = -1
    
                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]
                
            # If both slow k and slow d cross below the lower level from above 
            elif (max(slow_k[row], slow_d[row]) < oversold 
                and max(slow_k[row-1], slow_d[row-1]) > oversold):
                
                # Set the position signal to flat
                position_signal[row] = 0
    
                # Signal to go flat
                trade_signal[row] = 0 - position_signal[row-1]    
            
            # Otherwise, take no action
            else:                
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]
    
        df['slow_k_entry'] = slow_k
        df['slow_d_entry'] = slow_d         
   
        return df, start, trade_signal    
    
    
    @staticmethod
    def _entry_stochastic_pop(df, time_period, oversold, overbought):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        time_period : TYPE
            DESCRIPTION.
        lower : TYPE
            DESCRIPTION.
        upper : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        trade_signal : TYPE
            DESCRIPTION.

        """
        # Create Stochastics for the specified time period
        slow_k, slow_d = Indicators.stochastic(
            df['High'], df['Low'], df['Close'], fast_k_period=time_period)
    
        # Create start point based on slow d
        start = np.where(~np.isnan(slow_d))[0][0]
            
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(slow_d))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(slow_d))
        
        # for each row in the DataFrame after the slow d has started
        for row in range(start, len(slow_d)):
            
            # If both slow k and slow d cross above the upper level from below 
            if (min(slow_k[row], slow_d[row]) > overbought 
                and min(slow_k[row-1], slow_d[row-1]) < overbought):
                
                # Set the position signal to long
                position_signal[row] = 1
                
                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]
                
            # If slow k and slow d cross when the position is long 
            elif ((position_signal[row] == 1) 
                  and np.sign(slow_k[row] - slow_d[row]) != np.sign(
                      slow_k[row-1] - slow_d[row-1])):
                
                # Set the position signal to flat
                position_signal[row] = 0
                
                # Signal to go flat
                trade_signal[row] = 0 - position_signal[row-1]    
            
            # If both slow k and slow d cross below the lower level from above 
            elif (max(slow_k[row], slow_d[row]) < oversold 
                and max(slow_k[row-1], slow_d[row-1]) > oversold):
                
                # Set the position signal to short
                position_signal[row] = -1
    
                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]
                
            # If slow k and slow d cross when the position is short 
            elif ((position_signal[row] == -1) 
                  and np.sign(slow_k[row] - slow_d[row]) != np.sign(
                      slow_k[row-1] - slow_d[row-1])):
                
                # Set the position signal to flat
                position_signal[row] = 0
    
                # Signal to go flat
                trade_signal[row] = 0 - position_signal[row-1]    
            
            # Otherwise, take no action
            else:                
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]
    
        df['slow_k_entry'] = slow_k
        df['slow_d_entry'] = slow_d           
   
        return df, start, trade_signal    
    
    
    @staticmethod
    def _entry_rsi(df, time_period, oversold, overbought):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        time_period : TYPE
            DESCRIPTION.
        lower : TYPE
            DESCRIPTION.
        upper : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        trade_signal : TYPE
            DESCRIPTION.

        """
        # Create RSI for the specified time period
        rsi = Indicators.RSI(df['Close'], time_period)
    
        # Create start point based on lookback window
        start = np.where(~np.isnan(rsi))[0][0]
            
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(rsi))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(rsi))
        
        # for each row in the DataFrame after the cci has started
        for row in range(start, len(rsi)):
            
            # If the cci crosses above the threshold from below
            if rsi[row] < oversold and rsi[row-1] > oversold:
                
                # Set the position signal to long
                position_signal[row] = 1
                
                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]
            
            # If the cci crosses below the threshold from above
            elif rsi[row] > overbought and rsi[row-1] < overbought:
                
                # Set the position signal to short
                position_signal[row] = -1
    
                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]
            
            # Otherwise, take no action
            else:                
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]
    
        df['RSI_entry'] = rsi
   
        return df, start, trade_signal
    
    
    @staticmethod
    def _entry_commodity_channel_index(df, time_period, threshold):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        time_period : TYPE
            DESCRIPTION.
        threshold : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        trade_signal : TYPE
            DESCRIPTION.

        """
        
        # Create CCI for the specified time period
        cci = Indicators.CCI(df['High'], df['Low'], df['Close'], time_period)
        
        # Create start point based on lookback window
        start = np.where(~np.isnan(cci))[0][0]
            
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(cci))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(cci))
        
        # for each row in the DataFrame after the cci has started
        for row in range(start, len(cci)):
            
            # If the cci crosses above the threshold from below
            if cci[row] > threshold and cci[row-1] < threshold:
                
                # Set the position signal to long
                position_signal[row] = 1
                
                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]
            
            # If the cci crosses below the threshold from above
            elif cci[row] < threshold and cci[row-1] > threshold:
                
                # Set the position signal to short
                position_signal[row] = -1
    
                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]
            
            # Otherwise, take no action
            else:                
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]
    
        df['CCI_entry'] = cci
   
        return df, start, trade_signal
    
    
    @staticmethod
    def _entry_momentum(df, time_period, threshold):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        time_period : TYPE, optional
            DESCRIPTION. The default is 10.
        threshold : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.

        """
        # Calculate past close based on time period
        n_day_close = df['Close'].shift(10)
        
        # Create start point based on lookback window
        start = np.where(~np.isnan(n_day_close))[0][0]
            
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(n_day_close))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(n_day_close))
        
        # for each row in the DataFrame after the longest MA has started
        for row in range(start, len(n_day_close)):
            
            # If the price rises to equal or above the n-day high close 
            if df['Close'][row] >= n_day_close[row] + threshold:
                   
                # Set the position signal to long
                position_signal[row] = 1
                
                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]
                
            # If the price falls to equal or below the n-day low close 
            elif df['Close'][row] <= n_day_close[row] - threshold:
    
                # Set the position signal to short
                position_signal[row] = -1
    
                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]
    
            # Otherwise, take no action
            else:                
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]
    
        df['n_day_close'] = n_day_close
    
        return df, start, trade_signal
    
    
    @staticmethod
    def _entry_volatility(df, time_period, threshold):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        time_period : TYPE
            DESCRIPTION.
        threshold : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        trade_signal : TYPE
            DESCRIPTION.

        """
        # Create ATR for the specified time period
        atr = Indicators.ATR(df['High'], df['Low'], df['Close'], time_period)
    
        # Create start point based on lookback window
        start = np.where(~np.isnan(atr))[0][0]
            
        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(atr))
        
        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(atr))
        
        # for each row in the DataFrame after the atr has started
        for row in range(start, len(atr)):
            
            # If the increase in closing price exceeds the atr * threshold
            if (df['Close'][row] - df['Close'][row-1]) > (atr[row] * threshold):
                
                # Set the position signal to long
                position_signal[row] = 1
                
                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]
            
            # If the decrease in closing price exceeds the atr * threshold
            elif (df['Close'][row-1] - df['Close'][row]) > (atr[row] * threshold):
                
                # Set the position signal to short
                position_signal[row] = -1
    
                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]
            
            # Otherwise, take no action
            else:                
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]
    
        df['ATR_entry'] = atr
   
        return df, start, trade_signal
    
    
    @staticmethod        
    def _exit_parabolic_sar(
            df, trade_number, end_of_day_position, time_period, 
            acceleration_factor, sip_price):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.
        time_period : TYPE
            DESCRIPTION.
        acceleration_factor : TYPE
            DESCRIPTION.
        sip_price : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        parabolic_sar_exit : TYPE
            DESCRIPTION.

        """
        
        # Calculate rolling min and max closing prices based on time period
        rolling_high_close = df['Close'].rolling(time_period).max()
        rolling_low_close = df['Close'].rolling(time_period).min()
        
        # Initialize zero arrays to store data
        sar = np.array([0.0]*len(df))
        parabolic_sar_exit = np.array([0]*len(df))        
        
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                
                if trade_number[row] == 1:
                    if end_of_day_position[row] > 0:
                        initial_point = rolling_low_close[row]
                    else:
                        initial_point = rolling_high_close[row]
                
                trade_first_row = df.index.get_loc(
                    df[trade_number==trade_number[row]].index[0])
                trade_row_num = row - trade_first_row
                if end_of_day_position[row] > 0:
                    
                    if trade_number[row] == 1:
                        initial_point = rolling_low_close[row]
                    else:
                        if sip_price:
                            initial_point = rolling_low_close[row]
                        else:
                            initial_point = max(
                                df[trade_number==trade_number[row]-1]['Close'])
                    
                    if trade_row_num == 0:
                        sar[row] = initial_point
                        af = acceleration_factor
                    else:
                        sar[row] = sar[row-1] + (
                            af * (rolling_high_close[row] - sar[row-1]))
                        af = max(af + acceleration_factor, 0.2)
                        
                    if df['Close'][row] < sar[row]:
                        parabolic_sar_exit[row] = -1
                    else:
                        parabolic_sar_exit[row] = 0                    
                        
                elif end_of_day_position[row] < 0:
                    
                    if trade_number[row] == 1:
                        initial_point = rolling_high_close[row]
                    else:
                        if sip_price:
                            initial_point = rolling_high_close[row]
                        else:
                            initial_point = min(
                                df[trade_number==trade_number[row]-1]['Close'])                
                    
                    if trade_row_num == 0:
                        sar[row] = initial_point
                        af = acceleration_factor
                    else:
                        sar[row] = sar[row-1] + (af * (
                            rolling_low_close[row] - sar[row-1]))
                        af = max(af + acceleration_factor, 0.2)
                        
                    if df['Close'][row] > sar[row]:
                        parabolic_sar_exit[row] = 1
                    else:
                        parabolic_sar_exit[row] = 0                                 
                else:
                    sar[row] = 0
                    parabolic_sar_exit[row] = 0
                    
        df['rolling_high_close_sar_exit'] = rolling_high_close            
        df['rolling_low_close_sar_exit'] = rolling_low_close
        df['sar_exit'] = sar
                    
        return df, parabolic_sar_exit 
    
    
    @staticmethod
    def _exit_rsi_trail(df, trade_number, end_of_day_position, time_period, 
                        oversold, overbought):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.
        time_period : TYPE
            DESCRIPTION.
        oversold : TYPE
            DESCRIPTION.
        overbought : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        rsi_trail_exit : TYPE
            DESCRIPTION.

        """
       
        rsi = Indicators.RSI(close=df['Close'], time_period=time_period)
        
        rsi_trail_exit = np.array([0]*len(df))
        
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                if end_of_day_position[row] > 0:
                    if (df['Close'][row] < df['Close'][row-1] 
                        and rsi[row] > overbought):
                        rsi_trail_exit[row] = -1
                    else:
                        rsi_trail_exit[row] = 0
                elif end_of_day_position[row] < 0:
                    if (df['Close'][row] > df['Close'][row-1] 
                        and rsi[row] < oversold):
                        rsi_trail_exit[row] = 1
                    else:
                        rsi_trail_exit[row] = 0
                else:
                    rsi_trail_exit[row] = 0
        
        df['RSI_exit'] = rsi
                    
        return df, rsi_trail_exit
    
    
    @staticmethod
    def _exit_key_reversal(
            df, trade_number, end_of_day_position, time_period):
        
        # Calculate rolling high and low prices based on time period
        rolling_high = df['High'].rolling(time_period).max()
        rolling_low = df['Low'].rolling(time_period).min()
        
        key_reversal_exit = np.array([0]*len(df))
        
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                if end_of_day_position[row] > 0:
                    if (rolling_high[row] > rolling_high[row-1] 
                        and df['Close'][row] < df['Close'][row-1]):
                        key_reversal_exit[row] = -1
                    else:
                        key_reversal_exit[row] = 0
                elif end_of_day_position[row] < 0:
                    if (rolling_low[row] < rolling_low[row-1] 
                        and df['Close'][row] > df['Close'][row-1]):
                        key_reversal_exit[row] = 1
                    else:
                        key_reversal_exit[row] = 0
                else:
                    key_reversal_exit[row] = 0
                        
        df['rolling_high_key'] = rolling_high
        df['rolling_low_key'] = rolling_low
                    
        return df, key_reversal_exit
    
    
    @staticmethod
    def _exit_volatility(
            df, trade_number, end_of_day_position, time_period, threshold):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        time_period : TYPE, optional
            DESCRIPTION. The default is 5.
        threshold : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """
       
        volatility_exit = np.array([0]*len(df))
    
        atr = Indicators.ATR(df['High'], df['Low'], df['Close'], time_period)
        
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                if end_of_day_position[row] > 0:
                    if ((df['Close'][row] - df['Close'][row-1]) 
                        > (atr[row] * threshold)):
                        volatility_exit[row] = -1
                    else:
                        volatility_exit[row] = 0
                
                elif end_of_day_position[row] < 0:
                    if ((df['Close'][row-1] - df['Close'][row]) 
                        > (atr[row] * threshold)):
                        volatility_exit[row] = 1
                    else:
                        volatility_exit[row] = 0
                else:
                    volatility_exit[row] = 0
                        
        df['ATR_exit'] = atr
                   
        return df, volatility_exit
    
    
    @staticmethod
    def _exit_stochastic_crossover(df, trade_number, end_of_day_position, 
                                  time_period):
        
        slow_k, slow_d = Indicators.stochastic(
            df['High'], df['Low'], df['Close'], fast_k_period=time_period, 
            fast_d_period=3, slow_k_period=3, slow_d_period=3, output_type='slow')
    
        stoch_cross_exit = np.array([0]*len(df))
        
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                if end_of_day_position[row] > 0:
                    if slow_k[row] < slow_d[row] and slow_k[row-1] > slow_d[row-1]:
                        stoch_cross_exit[row] = -1
                    else:
                        stoch_cross_exit[row] = 0
                
                elif end_of_day_position[row] < 0:
                    if slow_k[row] > slow_d[row] and slow_k[row-1] < slow_d[row-1]:
                        stoch_cross_exit[row] = 1
                    else:
                        stoch_cross_exit[row] = 0
                else:
                    stoch_cross_exit[row] = 0
                        
        df['slow_k_exit'] = slow_k
        df['slow_d_exit'] = slow_d            
                    
        return df, stoch_cross_exit
    
    
    @staticmethod
    def _exit_random(df, trade_number, end_of_day_position):

        exit_days = random.randint(5,20)
        
        random_exit = np.array([0]*len(df))
        for row in range(1, len(df)):
            trade_first_row = df.index.get_loc(
                    df[trade_number==trade_number[row]].index[0])
            trade_row_num = row - trade_first_row
            
            if trade_number[row] != 0:
                if trade_row_num > exit_days-1:
                    if end_of_day_position[row] > 0:
                        if df['Close'][row] < df['Close'][row-1]:
                            random_exit[row] = -1
                        else:
                            random_exit[row] = 0
                    elif end_of_day_position[row] < 0:
                        if df['Close'][row] > df['Close'][row-1]:
                            random_exit[row] = 1
                        else:
                            random_exit[row] = 0
                    else:
                        random_exit[row] = 0
        
        df['random_days_exit'] = exit_days        
                        
        return df, random_exit
    
    
    @classmethod
    def _exit_dollar(
            cls, df, trigger_value, trade_number, 
                end_of_day_position, trade_high_price=None, 
                trade_low_price=None, exit_level=None):
        
        if exit_level == 'profit_target':
            return cls._exit_profit_target(
                df, trigger_value=trigger_value, 
                trade_number=trade_number, 
                end_of_day_position=end_of_day_position)
        
        elif exit_level == 'initial':
            return cls._exit_initial_dollar_loss(
                df, trigger_value=trigger_value,
                trade_number=trade_number, 
                end_of_day_position=end_of_day_position)
        
        elif exit_level == 'breakeven':
            return cls._exit_breakeven(
                df, trigger_value=trigger_value,
                trade_number=trade_number, 
                end_of_day_position=end_of_day_position, 
                trade_high_price=trade_high_price, 
                trade_low_price=trade_low_price)
    
        elif exit_level == 'trail_close':
            return cls._exit_trailing(
                df, trigger_value=trigger_value, 
                trade_number=trade_number, 
                end_of_day_position=end_of_day_position)
    
        elif exit_level == 'trail_high_low':
            return cls._exit_trailing(
                df, trigger_value=trigger_value, 
                trade_number=trade_number, 
                end_of_day_position=end_of_day_position)
    
    
    @staticmethod
    def _exit_profit_target(
            df, trigger_value, trade_number, end_of_day_position):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        trigger_value : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        profit_target_exit : TYPE
            DESCRIPTION.

        """
        profit_target_exit = np.array([0]*len(df))
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                if end_of_day_position[row] > 0:
                    if df['Close'][row] > trigger_value[row]:
                        profit_target_exit[row] = -1
                elif end_of_day_position[row] < 0:
                    if df['Close'][row] < trigger_value[row]:
                        profit_target_exit[row] = 1
                else:
                    profit_target_exit[row] = 0
    
        return df, profit_target_exit
    
    
    @staticmethod
    def _exit_initial_dollar_loss(
            df, trigger_value, trade_number, end_of_day_position):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        trigger_value : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        initial_dollar_loss_exit : TYPE
            DESCRIPTION.

        """
        initial_dollar_loss_exit = np.array([0]*len(df))
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                if end_of_day_position[row] > 0:
                    if df['Close'][row] < trigger_value[row]:
                        initial_dollar_loss_exit[row] = -1    
                elif end_of_day_position[row] < 0:
                    if df['Close'][row] > trigger_value[row]:
                        initial_dollar_loss_exit[row] = 1
                else:
                    initial_dollar_loss_exit[row] = 0    
    
        return df, initial_dollar_loss_exit
    
    
    @staticmethod
    def _exit_breakeven(
            df, trigger_value, trade_number, end_of_day_position, 
            trade_high_price, trade_low_price):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        trigger_value : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.
        trade_high_price : TYPE
            DESCRIPTION.
        trade_low_price : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        breakeven_exit : TYPE
            DESCRIPTION.

        """
        breakeven_exit = np.array([0.0]*len(df))
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                if end_of_day_position[row] > 0:    
                    if trade_high_price[row] > trigger_value[row]:
                        if df['Close'][row] < trigger_value[row]:
                            breakeven_exit[row] = -1
                elif end_of_day_position[row] < 0:
                    if trade_low_price[row] < trigger_value[row]:
                        if df['Close'][row] > trigger_value[row]:
                            breakeven_exit[row] = 1
                else:
                    breakeven_exit[row] = 0
    
        return df, breakeven_exit
    
    
    @staticmethod
    def _exit_trailing(df, trigger_value, trade_number, end_of_day_position):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        trigger_value : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.

        Returns
        -------
        df : TYPE
            DESCRIPTION.
        trailing_exit : TYPE
            DESCRIPTION.

        """
        trailing_exit = np.array([0.0]*len(df))
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                if end_of_day_position[row] > 0:
                    if df['Close'][row] < trigger_value[row]:
                        trailing_exit[row] = -1
                elif end_of_day_position[row] < 0:
                    if df['Close'][row] > trigger_value[row]:
                        trailing_exit[row] = 1    
                else:
                    trailing_exit[row] = 0    
    
        return df, trailing_exit
        
    
    @staticmethod
    def _exit_support_resistance(
            df, trade_number, end_of_day_position, time_period):
        
        # Calculate rolling min and max closing prices based on time period
        rolling_high_close = df['Close'].rolling(time_period).max()
        rolling_low_close = df['Close'].rolling(time_period).min()
        
        support_resistance_exit = np.array([0.0]*len(df))
        
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                if end_of_day_position[row] > 0:
                    if df['Close'][row] < rolling_low_close[row]:
                        support_resistance_exit[row] = -1
    
                elif end_of_day_position[row] < 0:
                    if df['Close'][row] > rolling_high_close[row]:
                        support_resistance_exit[row] = 1
    
                else:
                    support_resistance_exit[row] = 0
        
        df['rolling_high_close_sr_exit'] = rolling_high_close
        df['rolling_low_close_sr_exit'] = rolling_low_close
                    
        return df, support_resistance_exit                 
    
    
    @staticmethod
    def _exit_immediate_profit(
            df, trade_number, end_of_day_position, time_period):
        """
        

        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        time_period : TYPE, optional
            DESCRIPTION. The default is 5.

        Returns
        -------
        df : TYPE
            DESCRIPTION.

        """        
       
        immediate_profit_exit = np.array([0.0]*len(df))
        for row in range(1, len(df)):
            if trade_number[row] != 0:
                trade_first_row = df.index.get_loc(
                    df[trade_number==trade_number[row]].index[0])
                trade_row_num = row - trade_first_row
                if trade_row_num == time_period-1:
                    if end_of_day_position[row] > 0:
                        if df['Close'][row] < df['Close'][row-time_period]:
                            immediate_profit_exit[row] = -1
                    elif end_of_day_position[row] < 0:
                        if df['Close'][row] > df['Close'][row-time_period]:
                            immediate_profit_exit[row] = 1                
                    else:
                        immediate_profit_exit[row] = 0
       
        return df, immediate_profit_exit


    @staticmethod
    def _exit_nday_range(
            df, trade_number, end_of_day_position, time_period):
        """
        
    
        Parameters
        ----------
        df : TYPE
            DESCRIPTION.
        trade_number : TYPE
            DESCRIPTION.
        end_of_day_position : TYPE
            DESCRIPTION.
        time_period : TYPE, optional
            DESCRIPTION. The default is 5.
    
        Returns
        -------
        df : TYPE
            DESCRIPTION.
        nday_range_exit : TYPE
            DESCRIPTION.
    
        """       
        
        # The highest high minus the lowest low of the last n days
        n_day_low = df['Low'].rolling(time_period).min()
        n_day_high = df['High'].rolling(time_period).max()
        high_low_range = n_day_high - n_day_low
        
        # The largest high low daily range of the last n bars
        bar_range = (df['High'] - df['Low']).rolling(time_period).max()
        
        # Create an empty array to store the signals
        nday_range_exit = np.array([0.0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:

                # Find the row that relates to the trade entry
                trade_first_row = df.index.get_loc(
                    df[trade_number==trade_number[row]].index[0])
                trade_row_num = row - trade_first_row
                
                # If it is the trade entry date
                if trade_row_num == 0:
                    
                    # Set flags for if the targets have been hit
                    pt_1_hit = False
                    pt_2_hit = False
                    
                    # Set target 1 as the lower of the high low range of the last 
                    # n days and the longest bar of the last n days
                    target_1 = min(high_low_range[row], bar_range[row])
                    target_2 = max(high_low_range[row], bar_range[row])
                    
                    # If the position is long, add these to the close to set the 
                    # price targets
                    if end_of_day_position[row] > 0:
                        price_target_1 = df['Close'][row] + target_1
                        price_target_2 = df['Close'][row] + target_2
                    
                    # Otherwise subtract from the price
                    else:
                        price_target_1 = df['Close'][row] - target_1
                        price_target_2 = df['Close'][row] - target_2    
    
                # For every other day in the trade
                else:
                    
                    # If the position is long
                    if end_of_day_position[row] > 0:
                        
                        # If the close is above the target 1 and this has not yet 
                        # been hit 
                        if (df['Close'][row] > price_target_1 
                            and pt_1_hit == False):
                            
                            # Set the exit signal to -1
                            nday_range_exit[row] = -1
                            
                            # Set the profit target 1 hit flag to True
                            pt_1_hit = True
                            
                        # If the close is above the target 2 and this has not yet 
                        # been hit    
                        elif (df['Close'][row] > price_target_2 
                              and pt_2_hit == False):
                            
                            # Set the exit signal to -1
                            nday_range_exit[row] = -1
                            
                            # Set the profit target 2 hit flag to True
                            pt_2_hit = True
                            
                        # Otherwise    
                        else:
                            # Set the exit signal to 0
                            nday_range_exit[row] = 0
                    
                    # If the position is short        
                    else:
                        
                        # If the close is below the target 1 and this has not yet 
                        # been hit 
                        if (df['Close'][row] < price_target_1 
                            and pt_1_hit == False):
                            
                            # Set the exit signal to 1
                            nday_range_exit[row] = 1
                            
                            # Set the profit target 1 hit flag to True
                            pt_1_hit = True
                            
                        # If the close is above the target 2 and this has not yet 
                        # been hit    
                        elif (df['Close'][row] < price_target_2 
                              and pt_2_hit == False):
                            
                            # Set the exit signal to 1
                            nday_range_exit[row] = 1
                            
                            # Set the profit target 2 hit flag to True
                            pt_2_hit = True
                            
                        # Otherwise    
                        else:
                            # Set the exit signal to 0
                            nday_range_exit[row] = 0
       
        return df, nday_range_exit
    

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
    def _profit_data(cls, df, position_size, slippage, commission, equity):
        """
        Adds profit and drawdown fields to the OHLC data

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        position_size : Int, optional
            The number of units to trade. The default is 100.
        slippage : Float, optional
            The amount of slippage to apply to traded prices in basis points. 
            The default is 5 bps per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.
        equity : Float
            The initial account equity level.    

        Returns
        -------
        df : DataFrame
            Returns the input DataFrame with additional columns.

        """
        
        # Create pnl data
        df = cls._pnl_mtm(df=df, slippage=slippage, commission=commission)
        
        # Create cumulative trade pnl, equity and drawdown data
        df = cls._cumulative_trade_pnl_and_equity(df=df, equity=equity)
        
        # Create perfect profit data
        df = cls._perfect_profit(df=df, position_size=position_size)
           
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
            The amount of slippage to apply to traded prices in basis points. 
            The default is 5 bps per unit.
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
            The amount of slippage to apply to traded prices in basis points. 
            The default is 5 bps per unit.
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
        last_day_trade_pnl = np.array([0]*len(close), dtype=float)
        
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
                        (pos[row - 1] * (open[row] - close[row - 1]) 
                         - abs(pos[row - 1] * slippage * 0.0001 * open[row])) 
                        - commission)
            
            # If the current position is not flat
            else:
                
                # If the position is the same as the previous day
                if pos[row] == pos[row - 1]:
                    
                    # Set the pnl to the current position * the difference 
                    # between todays close and yesterdays close
                    day_pnl[row] = pos[row] * (close[row] - close[row - 1])
                
                # If the position is reversed from the previous day
                elif pos[row] == (-1) * pos[row - 1]:
                    day_pnl[row] = (
                        (pos[row] * (close[row] - open[row]) 
                         - abs(pos[row] * slippage * 0.0001 * open[row])) 
                        - commission)
                    last_day_trade_pnl[row] = (
                        (pos[row - 1] * (open[row] - close[row - 1]) 
                         - abs(pos[row - 1] * slippage * 0.0001 * open[row])) 
                        - commission)
                
                # If the position was opened from flat
                else:
                    
                    # Set the pnl to the current position * the difference 
                    # between todays open and todays close less the cost of 
                    # slippage and commission
                    day_pnl[row] = (
                        (pos[row] * (close[row] - open[row]) 
                         - abs(pos[row] * slippage * 0.0001 * open[row])) 
                        - commission)
        
        # Create daily pnl column in DataFrame, rounding to 2dp            
        df['current_trade_pnl'] = np.round(day_pnl, 2)    
        df['last_day_trade_pnl'] = last_day_trade_pnl
        df['daily_pnl'] = df['current_trade_pnl'] + df['last_day_trade_pnl']    
    
        return df
    
    
    @staticmethod
    def _cumulative_trade_pnl_and_equity(df, equity):
        """
        Calculate cumulative per trade pnl and various account equity series

        Parameters
        ----------
        df : DataFrame
            The OHLC data and trades signals.
        equity : Float
            The initial account equity level.

        Returns
        -------
        df : DataFrame
            The input data with additional columns.

        """
        # Take the trade number and daily pnl series from df
        trade_number = df['trade_number']
        current_trade_pnl = df['current_trade_pnl']    
        last_day_trade_pnl = df['last_day_trade_pnl']        
        daily_pnl = df['daily_pnl']
    
        # Create arrays of zeros
        cumulative_trade_pnl = np.array([0.0]*len(daily_pnl))
        mtm_equity = np.array([0.0]*len(daily_pnl))
        max_mtm_equity = np.array([0.0]*len(daily_pnl))
        min_mtm_equity = np.array([0.0]*len(daily_pnl))
        closed_equity = np.array([0.0]*len(daily_pnl))
        max_closed_equity = np.array([0.0]*len(daily_pnl))
        open_equity = np.array([0.0]*len(daily_pnl))
        max_retracement = np.array([0.0]*len(daily_pnl))
        ulcer_index_d_sq = np.array([0.0]*len(daily_pnl))
        max_trade_pnl = np.array([0.0]*len(daily_pnl))
        trade_pnl_drawback = np.array([0.0]*len(daily_pnl))
        trade_pnl_drawback_perc = np.array([0.0]*len(daily_pnl))
        
        # Create max drawdown and max gain numpy arrays of zeros
        max_drawdown = np.array([0.0]*len(daily_pnl))
        max_drawdown_perc = np.array([0.0]*len(daily_pnl))
        max_gain = np.array([0.0]*len(daily_pnl))
        max_gain_perc = np.array([0.0]*len(daily_pnl))
    
        # Set the initial equity values
        mtm_equity[0] = equity
        max_mtm_equity[0] = equity
        min_mtm_equity[0] = equity
        closed_equity[0] = equity
        max_closed_equity[0] = equity
        
        # For each row of data
        for row in range(1, len(daily_pnl)):
            
            # The index location of the trade entry date
            trade_first_row = df.index.get_loc(
                df[trade_number==trade_number[row]].index[0])
            
            # The number of days since trade entry
            trade_row_num = row - trade_first_row
            
            # The index location of the trade entry date
            trade_last_row = df.index.get_loc(
                df[trade_number==trade_number[row]].index[-1])
    
            # If there is a current trade
            if trade_number[row] != 0:   
                
                # If it is the trade entry date and there was no prior trade 
                if (trade_row_num == 0 
                    and trade_number[row-1] == 0):
                       
                    # Set cumulative trade pnl to the days pnl
                    cumulative_trade_pnl[row] = daily_pnl[row]

                    # The maximum of the initial days pnl and zero
                    max_trade_pnl[row] = max(daily_pnl[row], 0)
                    
                    # Set the mtm equity to the previous days mtm equity plus
                    # the days pnl
                    mtm_equity[row] = mtm_equity[row-1] + daily_pnl[row]
                    
                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = closed_equity[row-1]


                # If it is the trade entry date and this reverses the position 
                # of a prior trade 
                elif (trade_row_num == 0 
                    and trade_number[row-1] != 0):
                       
                    # Set cumulative trade pnl to the previous days cumulative 
                    # pnl plus the last days pnl
                    cumulative_trade_pnl[row] = (cumulative_trade_pnl[row-1] 
                                                 + last_day_trade_pnl[row])
                   
                    #  Set cumulative trade pnl to the previous days cumulative 
                    # pnl plus the last days pnl
                    max_trade_pnl[row] = max(
                        cumulative_trade_pnl[row], max_trade_pnl[row-1])
                    
                    # Set the mtm equity to the previous days mtm equity plus
                    # the days pnl
                    mtm_equity[row] = mtm_equity[row-1] + daily_pnl[row]
                    
                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = (mtm_equity[row-1] 
                                          + last_day_trade_pnl[row])

    
                # If it is the trade exit date and not a reversal
                elif (trade_last_row - row == 0 
                      and trade_number[row] == trade_number[row-1]):
                    
                    # Set cumulative trade pnl to the previous days cumulative 
                    # pnl plus the days pnl
                    cumulative_trade_pnl[row] = (cumulative_trade_pnl[row-1] 
                                                 + daily_pnl[row]) 
                    
                    # The maximum of the current trade equity and the maximum 
                    # trade equity of the previous day
                    max_trade_pnl[row] = max(
                        cumulative_trade_pnl[row], max_trade_pnl[row-1])
                    
                    # Set the mtm equity to the previous days mtm equity plus
                    # the days pnl
                    mtm_equity[row] = mtm_equity[row-1] + daily_pnl[row]
                    
                    # Set the closed equity to the mtm equity
                    closed_equity[row] = mtm_equity[row] 

                
                # If it is the second day of a reversal trade
                elif (trade_row_num == 1 
                      and trade_number[row-1] != trade_number[row-2]):
                    
                    # Set cumulative trade pnl to the previous days current 
                    # pnl plus the days pnl
                    cumulative_trade_pnl[row] = (current_trade_pnl[row-1] 
                                                 + daily_pnl[row])
                    
                    # The maximum of the first and second days pnl and zero
                    max_trade_pnl[row] = max(cumulative_trade_pnl[row], 
                                             current_trade_pnl[row-1])
                    
                    # Set the mtm equity to the previous days mtm equity plus
                    # the days pnl
                    mtm_equity[row] = mtm_equity[row-1] + daily_pnl[row]
                    
                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = closed_equity[row-1]

                
                # For every other day in the trade 
                else:
                    # Set cumulative trade pnl to the previous days cumulative 
                    # pnl plus the days pnl
                    cumulative_trade_pnl[row] = (cumulative_trade_pnl[row-1] 
                                                 + daily_pnl[row])
                    
                    # The maximum of the current trade equity and the maximum 
                    # trade equity of the previous day
                    max_trade_pnl[row] = max(
                        cumulative_trade_pnl[row], max_trade_pnl[row-1])
                    
                    # Set the mtm equity to the previous days mtm equity plus
                    # the days pnl
                    mtm_equity[row] = mtm_equity[row-1] + daily_pnl[row]
                    
                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = closed_equity[row-1]


            # If there is no current trade 
            else:
                # Set cumulative trade pnl to zero
                cumulative_trade_pnl[row] = 0
                
                # Set the mtm equity to the previous days mtm equity 
                mtm_equity[row] = mtm_equity[row-1]
                
                # Set the closed equity to the previous days closed equity                    
                closed_equity[row] = closed_equity[row-1]

            # Maximum mtm equity to this point            
            max_mtm_equity[row] = np.max(mtm_equity[0:row+1])
            
            # Minimum mtm equity to this point            
            min_mtm_equity[row] = np.min(mtm_equity[0:row+1])
            
            # Maximum closed equity to this point
            max_closed_equity[row] = np.max(closed_equity[:row+1])    
            
            # Current open equity
            open_equity[row] = mtm_equity[row] - closed_equity[row]
            
            # Maximum of max closed equity and current mtm equity, used in 
            # calculating Average Max Retracement
            max_retracement[row] = max(
                (max_closed_equity[row] - mtm_equity[row]), 0)
            
            # Maximum drawdown is the smallest value of the current cumulative 
            # pnl less the max of all previous rows cumulative pnl and zero
            max_drawdown[row] = mtm_equity[row] - max_mtm_equity[row]
            
            # Percentage Maximum drawdown
            max_drawdown_perc[row] = ((mtm_equity[row] - max_mtm_equity[row]) 
                                      / max_mtm_equity[row])            
          
            # Maximum gain is the largest value of the current cumulative 
            # pnl less the min of all previous rows and zero
            max_gain[row] = mtm_equity[row] - min_mtm_equity[row]
        
            # Percentage Maximum gain
            max_gain_perc[row] = ((mtm_equity[row] - min_mtm_equity[row]) 
                                  / min_mtm_equity[row])
            
            # Squared difference between max mtm equity and current mtm equity,
            # used in calculating Ulcer Index
            ulcer_index_d_sq[row] = (
                (((max_mtm_equity[row] - mtm_equity[row]) 
                 / max_mtm_equity[row]) * 100) ** 2)
            
            # The difference between the highest equity peak of the trade and 
            # the current trade open equity
            trade_pnl_drawback[row] = (max_trade_pnl[row] 
                                       - cumulative_trade_pnl[row])
            
            # The percentage difference between the highest equity peak of the 
            # trade and the current trade open equity
            if max_trade_pnl[row] !=0:
                trade_pnl_drawback_perc[row] = (
                    (max_trade_pnl[row] - cumulative_trade_pnl[row]) 
                    / max_trade_pnl[row])
                       
        df['cumulative_trade_pnl'] = cumulative_trade_pnl
        df['max_trade_pnl'] = max_trade_pnl
        df['trade_pnl_drawback'] = trade_pnl_drawback
        df['trade_pnl_drawback_perc'] = trade_pnl_drawback_perc
        df['mtm_equity'] = mtm_equity
        df['closed_equity'] = closed_equity
        df['open_equity'] = open_equity
        df['max_closed_equity'] = max_closed_equity
        df['max_retracement'] = max_retracement
        df['max_mtm_equity'] = max_mtm_equity
        df['min_mtm_equity'] = min_mtm_equity
        df['max_dd'] = max_drawdown
        df['max_dd_perc'] = max_drawdown_perc
        df['max_gain'] = max_gain
        df['max_gain_perc'] = max_gain_perc
        df['ulcer_index_d_sq'] = ulcer_index_d_sq
            
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
        
        dpp = np.array([0]*len(df))
        
        for row in range(len(dpp)):
            
            # Calculate Daily Perfect Profit            
            dpp[row] = (abs(df['High'][row] - df['Low'][row]) * position_size)
            
            # If the High and Low are the same
            if dpp[row] == 0:
            
                # Use the previous close
                dpp[row] = (
                    abs(df['High'][row] - df['Close'][row-1]) * position_size)        
        
        # Set this to the daily perfect profit
        df['daily_perfect_profit'] = dpp
        
        # Create array of zeros
        df['total_perfect_profit'] = np.array(
            [0.0]*len(df['Close']), dtype=float)
        
        # Cumulative sum of daily perfect profit column
        df['total_perfect_profit'] = df['daily_perfect_profit'].cumsum()
    
        return df
    
    
    @staticmethod
    def _create_monthly_data(df, equity):
        """
        Create monthly summary data 

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        equity : Int
            The initial equity level.

        Returns
        -------
        monthly_data : DataFrame
            The monthly summary data.

        """
        # Create empty DataFrame
        monthly_data = pd.DataFrame()
        
        # Summarize daily pnl data by resampling to monthly
        monthly_data['total_net_profit'] = df['daily_pnl'].resample('1M').sum()
        monthly_data['average_net_profit'] = df[
            'daily_pnl'].resample('1M').mean()
        monthly_data['max_net_profit'] = df['daily_pnl'].resample('1M').max()
        monthly_data['min_net_profit'] = df['daily_pnl'].resample('1M').min()
        
        # Create arrays of zeros to hold data
        monthly_data['beginning_equity'] = np.array([0.0]*len(monthly_data))
        monthly_data['additions'] = np.array([0.0]*len(monthly_data))
        monthly_data['withdrawals'] = np.array([0.0]*len(monthly_data))    
        monthly_data['end_equity'] = np.array([0.0]*len(monthly_data))
        monthly_data['return'] = np.array([0.0]*len(monthly_data))
        monthly_data['beginning_equity_raw'] = np.array([0.0]*len(monthly_data))
        monthly_data['end_equity_raw'] = np.array([0.0]*len(monthly_data))
        monthly_data['return_raw'] = np.array([0.0]*len(monthly_data))        
        monthly_data['abs_loss'] = np.array([0.0]*len(monthly_data))
        monthly_data['abs_loss_raw'] = np.array([0.0]*len(monthly_data))


        # Set initial values        
        monthly_data['additions'].iloc[0] = equity
        monthly_data['end_equity'].iloc[0] = (
            monthly_data['beginning_equity'].iloc[0]
            + monthly_data['additions'].iloc[0]
            + monthly_data['withdrawals'].iloc[0]
            + monthly_data['total_net_profit'].iloc[0])
        monthly_data['return'].iloc[0] = (
            (monthly_data['total_net_profit'].iloc[0]) 
            / (monthly_data['beginning_equity'].iloc[0]
               + monthly_data['additions'].iloc[0]
               + monthly_data['withdrawals'].iloc[0]))
        monthly_data['beginning_equity_raw'].iloc[0] = equity
        monthly_data['end_equity_raw'].iloc[0] = (
            monthly_data['beginning_equity_raw'].iloc[0]
            + monthly_data['total_net_profit'].iloc[0])
        monthly_data['return_raw'].iloc[0] = (
            (monthly_data['total_net_profit'].iloc[0]) 
            / (monthly_data['beginning_equity_raw'].iloc[0]))
    
        # For each month
        for row in range(1, len(monthly_data)):
            
            # Beginning equity is the closing equity from the prior period
            # Raw data keeps all the profits invested whereas the other resets 
            # the equity balance each year
            monthly_data['beginning_equity'][row] = monthly_data[
                'end_equity'][row-1]
            monthly_data['beginning_equity_raw'][row] = monthly_data[
                'end_equity_raw'][row-1]
            
            # For each change in year
            if monthly_data.index.year[row] != monthly_data.index.year[row-1]:
                
                # If the end of year equity level is less than the initial 
                # level
                if monthly_data['end_equity'][row-1] < equity:
                    
                    # Add back the difference to additions
                    monthly_data['additions'][row] = (
                        equity - monthly_data['end_equity'][row-1])
                else:
                    # Otherwise subtract from withdrawals
                    monthly_data['withdrawals'][row] = (
                        equity - monthly_data['end_equity'][row-1])
            
            # Ending equity is the beginning equity plus the sum of additions, 
            # withdrawals and the net pnl over the period        
            monthly_data['end_equity'][row] = (
                monthly_data['beginning_equity'][row] 
                + monthly_data['total_net_profit'][row]
                + monthly_data['additions'][row]
                + monthly_data['withdrawals'][row])
            
            # Ending equity raw is the beginning equity plus the net pnl over 
            # the period
            monthly_data['end_equity_raw'][row] = (
                monthly_data['beginning_equity_raw'][row] 
                + monthly_data['total_net_profit'][row])
            
            # Monthly return is the net pnl over the period divided by the 
            # beginning equity plus the sum of additions and withdrawals 
            monthly_data['return'][row] = (
                (monthly_data['total_net_profit'][row]) 
                / (monthly_data['beginning_equity'][row] 
                   + monthly_data['additions'][row]
                   + monthly_data['withdrawals'][row]))
            
            # Monthly return raw is the net pnl over the period divided by the 
            # beginning equity raw
            monthly_data['return_raw'][row] = (
                (monthly_data['total_net_profit'][row]) 
                / (monthly_data['beginning_equity_raw'][row]))
    
            # For use in Gain to Pain Ratio, absolute loss is the positive 
            # value of the negative monthly returns
            if monthly_data['return'][row] < 0:
                monthly_data['abs_loss'][row] = -monthly_data['return'][row]
            
            # For use in Gain to Pain Ratio, absolute loss is the positive 
            # value of the negative monthly returns
            if monthly_data['return_raw'][row] < 0:
                monthly_data['abs_loss_raw'][row] = -monthly_data[
                    'return_raw'][row]
                   
        return monthly_data


    def _output_results(self, df, monthly_data, reversal):
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
        self.performance_data(df=df, monthly_data=monthly_data, 
                              reversal=reversal)
        
        # Print out results
        self.report_table(input_dict=self.perf_dict, reversal=reversal)
    
        return self


    def performance_data(self, df, monthly_data, reversal):
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
        if self.source == 'alpha' and self.asset_type in ['fx', 'crypto']:
            perf_dict['contract'] = self.ccy_1+self.ccy_2
        else:
            perf_dict['contract'] = self.ticker

        # Reversal strategy has only a single label
        if reversal:
            perf_dict['strategy_label'] = self.strategy_label
        
        # Otherwise take the entry, exit and stop labels
        else:
            perf_dict['entry_label'] = self.entry_label
            perf_dict['exit_label'] = self.exit_label
            perf_dict['stop_label'] = self.stop_label
        
        # Initial Equity
        perf_dict['initial_equity'] = df['mtm_equity'].iloc[0]
        
        # Set Ticker Longname
        if self.source == 'norgate':
            perf_dict['longname'] = self.norgate_name_dict[self.ticker]
        else:
            perf_dict['longname'] = perf_dict['contract']
        
        # Slippage and commission in dollars
        perf_dict['slippage'] = self.slippage
        perf_dict['commission'] = self.commission
        
        # Start and end dates
        perf_dict['start_date'] = df.index[0].date().strftime("%d/%m/%y")
        perf_dict['end_date'] = df.index[-1].date().strftime("%d/%m/%y")
               
        # Maximum margin required 
        perf_dict['margin'] = math.ceil(
            max(abs(df['position_mtm'])) / 100) * 100
        
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
            perf_dict['net_pnl'] / perf_dict['initial_equity'])
        
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
            (perf_dict['net_pnl'] / perf_dict['perfect_profit']) * 100, 2)
        
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
        
        # Maximum Equity drawdown in percentage terms
        perf_dict['max_balance_drawback_perc'] = min(df['max_dd_perc']) * 100
        
        # Time to recover from Max Drawdown
        perf_dict['time_to_recover'] = self._time_to_recover(df)
            
        # Maximum Equity gain
        perf_dict['max_gain'] = np.round(max(df['max_gain']), 2)
        
        # Maximum Equity gain in percentage terms
        perf_dict['max_gain_perc'] = max(df['max_gain_perc']) * 100
        
        # Time taken for maximum gain
        perf_dict['max_gain_time'] = self._time_max_gain(df)
       
        # Reward / Risk ratio
        perf_dict['reward_risk'] = (perf_dict['net_pnl'] / 
                                    abs(perf_dict['max_balance_drawback']))
        
        # Annual Rate of Return
        perf_dict['annual_ror'] = (perf_dict['annualized_profit '] / 
                                   perf_dict['initial_equity']) * 100
        
        # Profit Index
        perf_dict['profit_factor'] = (perf_dict['total_profit'] / 
                                     abs(perf_dict['total_loss']))
        
        # Mathematical Advantage
        perf_dict['mathematical_advantage'] = np.round(
            (perf_dict['num_wins'] / len(trades) * 100) * (
                perf_dict['profit_factor'] + 1) - 100, 2)
        
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
            perf_dict['initial_equity']) * 100
        
        # Return of a buy and hold strategy of SPX since the first trade entry
        perf_dict['long_only_pnl_spx'] = (
            (self.spx['Close'].iloc[-1] 
             - self.spx['Open'].iloc[first_trade_start]) * 
            self.spx_position_size) 
        
        # Annual Rate of return of buy and hold of SPX
        perf_dict['annual_long_only_spx_ror'] = (
            (perf_dict['long_only_pnl_spx'] / perf_dict['return_period']) / 
            perf_dict['initial_equity']) * 100
       
        # Riskfree Rate
        perf_dict['riskfree_rate'] = self.df_dict['df_riskfree']
        
        # Mean Price
        perf_dict['close_price_mean'] = np.round(np.mean(df['Close']), 2)
      
        # Variance Price
        perf_dict['close_price_variance'] = np.round(np.var(df['Close']), 2) 
        
        # Standard Deviation Price
        perf_dict['close_price_std_dev'] = np.round(np.std(df['Close']), 2)

        # Skewness Price
        perf_dict['close_price_skewness'] = np.round(skew(df['Close']), 2)        

        # Kurtosis Price
        perf_dict['close_price_kurtosis'] = np.round(kurtosis(df['Close']), 2)
        
        # Mean Return
        perf_dict['close_return_mean'] = np.round(
            np.mean(df['Close'].pct_change()*100), 2)
      
        # Variance Return
        perf_dict['close_return_variance'] = np.round(
            np.var(df['Close'].pct_change()*100), 2) 
        
        # Standard Deviation Return
        perf_dict['close_return_std_dev'] = np.round(
            np.std(df['Close'].pct_change()*100), 2)

        # Annualized Volatility
        perf_dict['close_return_ann_vol'] = np.round(
            np.std(df['Close'].pct_change()*np.sqrt(252)*100), 2)

        # Skewness Return
        perf_dict['close_return_skewness'] = np.round(
            skew(df['Close'].pct_change()*100, nan_policy='omit'), 2)        

        # Kurtosis Return
        perf_dict['close_return_kurtosis'] = np.round(kurtosis(
            df['Close'].pct_change()*100, nan_policy='omit'), 2)
        
        # Efficiency Ratio
        perf_dict['efficiency_ratio'] = np.round(
            (abs(df['Close'][-1] - df['Close'][0]) 
            / np.nansum(abs(df['Close']-df['Close'].shift()))) * 100, 2)
        
        # MTM Equity Standard Deviation
        perf_dict['equity_std_dev'] = np.round(
            np.std(df['mtm_equity'].pct_change()*100), 2)
        
        # Sharpe Ratio
        perf_dict['sharpe_ratio'] = ((
            perf_dict['annual_ror'] - perf_dict['riskfree_rate']) 
            / perf_dict['equity_std_dev'])
        
        # Information Ratio
        perf_dict['information_ratio'] = (
            perf_dict['annual_ror'] / perf_dict['equity_std_dev'])
        
        # Treynor Ratio
        inv_correl = self.spx.Close.pct_change().corr(
            df.mtm_equity.pct_change())
        #stock_correl = self.spx.Close.pct_change().corr(df.Close.pct_change())
        sd_inv = np.std(df.mtm_equity.pct_change())
        #sd_stock = np.std(df.Close.pct_change())
        sd_index = np.std(self.spx.Close.pct_change())
        #beta = stock_correl * (sd_stock / sd_index)
        beta = inv_correl * (sd_inv / sd_index)
        treynor = np.round((
            perf_dict['annual_ror'] - perf_dict['riskfree_rate']) 
            / beta, 2)
        if treynor > 0:
            perf_dict['treynor_ratio'] = treynor
        else:
            perf_dict['treynor_ratio'] = 'N/A'
        
        # Sortino Ratio
        equity_return = df['mtm_equity'].pct_change()*100
        downside_deviation = np.std(equity_return[equity_return < 0])
        perf_dict['sortino_ratio'] = np.round((
            perf_dict['annual_ror'] - perf_dict['riskfree_rate']) 
            / downside_deviation, 2)
        
        # Calmar Ratio
        perf_dict['calmar_ratio'] = (
            perf_dict['annual_ror'] 
            / abs(perf_dict['max_balance_drawback_perc']))
        
        # Average Maximum Retracement
        perf_dict['average_max_retracement'] = (
            df['max_retracement'].mean())
        
        # Ulcer Index
        perf_dict['ulcer_index'] = np.sqrt(
            df['ulcer_index_d_sq'][df['ulcer_index_d_sq']!=0].mean())
        
        # Gain to Pain
        perf_dict['gain_to_pain'] = np.round(
            monthly_data['return'].sum() / monthly_data['abs_loss'].sum(), 2)
        
        # Maximum MTM Equity Profit
        perf_dict['max_equity_profit'] = (
            df['max_mtm_equity'].iloc[-1] - perf_dict['initial_equity'])
        
        # Largest open equity drawdown
        perf_dict['open_equity_dd'] = (df['trade_pnl_drawback'].max())*(-1)

        # Open equity
        perf_dict['open_equity'] = df['open_equity'].iloc[-1]
        
        # Closed equity
        perf_dict['closed_equity'] = df['closed_equity'].iloc[-1]
        
        # Largest monthly gain
        perf_dict['month_net_pnl_large'] = (
            monthly_data['total_net_profit'].max())
        
        # Largest monthly loss
        perf_dict['month_net_pnl_small'] = (
            monthly_data['total_net_profit'].min())
        
        # Average monthly gain/loss
        perf_dict['month_net_pnl_av'] = (
            monthly_data['total_net_profit'].mean())
        
        # Values still to be worked out
        placeholder_dict = {
            'pessimistic_margin':0.00,
            'adj_pess_margin':0.00,
            'pess_month_avg':0.00,
            'pess_month_variance':0.00,
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
    
    
    @staticmethod    
    def _time_to_recover(df):
        max_dd_val = min(df['max_dd'])
        dd_start = df.index.get_loc(df[df['max_dd']==max_dd_val].index[0])
        if max(df['max_dd'][dd_start:]) == 0:
            dd_length = (df['max_dd'][dd_start:].values == 0).argmax()
            return dd_length
        else:
            return 'N/A'


    @staticmethod
    def _time_max_gain(df):
        max_gain_val = max(df['max_gain'])
        gain_rev = df['max_gain'][::-1]
        max_gain_loc = gain_rev.index.get_loc(
            gain_rev[gain_rev==max_gain_val].index[0])
        gain_length = (gain_rev[max_gain_loc:].values == 0).argmax()
        return gain_length


    def report_table(self, input_dict, reversal):
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
        input_dict = self._dict_format(input_dict)
        
        # Format header - centred and with lines above and below
        print('='*78)
        print('{:^78}'.format('Performance Analysis Report'))
        print('-'*78)
        
        # Contract traded on left and period covered on right
        print('Contract Ticker : {:<41}{} - {}'.format(
            input_dict['contract'],
            input_dict['start_date'], 
            input_dict['end_date']))
        
        # Contract Longname
        print('Contract Name   : {:>10}'.format(input_dict['longname']))
        
        if reversal:
            # Strategy
            print('Strategy        : {:>10}'.format(
                input_dict['strategy_label']))

        else:
            # Entry Method
            print('Entry           : {:>10}'.format(input_dict['entry_label']))
        
            # Exit Method
            print('Exit            : {:>10}'.format(input_dict['exit_label']))        

            # Stop Method
            print('Stop            : {:>10}'.format(input_dict['stop_label']))
        
        # Beginning balance on left
        print('Initial Equity  :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['initial_equity'],
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
        print('Av. Trade P/L   :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['av_trade'],
            '',
            'Model Efficiency : %',
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
    
        # Open and Closed equity 
        print('Open Equity.......... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['open_equity'],
            '',
            'Closed Equity......... $',
            input_dict['closed_equity']))
       
        # Max Open Equity DD / equity profit
        print('Max Open Eq. DD...... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['open_equity_dd'],
            '',
            'Max Equity Profit..... $',
            input_dict['max_equity_profit']))
        
        # Max Closed Equity DD drawback / Max Equity Gain
        print('Peak to Trough DD.... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['max_balance_drawback'],
            '',
            'Trough to Peak Gain... $',
            input_dict['max_gain']))
                       
        # Max Percent drawdown / gain
        print('Max Drawdown......... %{:>10}{:<6}{}{:>10}'.format(
            input_dict['max_balance_drawback_perc'],
            '',
            'Max Gain.............. %',
            input_dict['max_gain_perc']))        
        
        # Time to recover and time for max gain
        print('Time to Recover (Days) {:>10}{:<6}{}{:>10}'.format(
            input_dict['time_to_recover'],
            '',
            'Time for Max Gain (Days)',
            input_dict['max_gain_time']))
        
        # Reward / Risk
        print('Reward / Risk........  {:^16}{}{:>10}'.format(
            '{} TO 1'.format(input_dict['reward_risk']),
            'Annual Rate of Return. %',
            input_dict['annual_ror']))
        
        # Profit Index and Mathematical Advantage
        print('Profit Factor........  {:>10}{:<6}{}{:>10}'.format(
            input_dict['profit_factor'],
            '',
            'Mathematical Advantage %',
            input_dict['mathematical_advantage']))
            
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
        print('Monthly Net PL Large. ${:>10}{:<6}{}{:>10}'.format(
            input_dict['month_net_pnl_large'],
            '',
            'Monthly Net PL Small.. $',
            input_dict['month_net_pnl_small']))
        
        print('Monthly Net PL Ave... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['month_net_pnl_av'],
            '',
            'Modified Pess. Margin.  ',
            input_dict['mod_pess_margin']))
        
        print('Long Only Net PL..... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['long_only_pnl'],
            '',
            'Long Only Annual RoR.. %',
            input_dict['annual_long_only_ror']))
        
        print('Long Only SPX Net PL. ${:>10}{:<6}{}{:>10}'.format(
            input_dict['long_only_pnl_spx'],
            '',
            'Long Only SPX Ann RoR. %',
            input_dict['annual_long_only_spx_ror']))       
    
        
        # Key Performance Measures
        print('-'*78)
        print('{:^78}'.format('Key Performance Measures'))
        print('-'*78)
        
        # Sharpe Ratio & Information Ratio
        print('Sharpe Ratio.........  {:>10}{:<6}{}{:>10}'.format(
            input_dict['sharpe_ratio'],
            '',
            'Information Ratio.....  ',
            input_dict['information_ratio']))
       
        # Sortino Ratio & Treynor Ratio
        print('Sortino Ratio........  {:>10}{:<6}{}{:>10}'.format(
            input_dict['sortino_ratio'],
            '',
            'Treynor Ratio.........  ',
            input_dict['treynor_ratio']))
        
        # Calmar Ratio & Average Max Retracement
        print('Calmar Ratio.........  {:>10}{:<6}{}{:>10}'.format(
            input_dict['calmar_ratio'],
            '',
            'Av. Max Retracement... $',
            input_dict['average_max_retracement']))
       
        # Ulcer Index & Gain to Pain
        print('Ulcer Index..........  {:>10}{:<6}{}{:>10}'.format(
            input_dict['ulcer_index'],
            '',
            'Gain to Pain..........  ',
            input_dict['gain_to_pain']))
       
        # Price Distribution statistics
        print('-'*78)
        print('{:^78}'.format('Data Distribution Statistics'))
        print('-'*78)
        
        # Mean & Standard Deviation of Prices
        print('Mean Price........... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['close_price_mean'],
            '',
            'St. Dev. Price........ $',
            input_dict['close_price_std_dev']))
       
        # Skewness & Kurtosis of Prices
        print('Skewness Price.......  {:>10}{:<6}{}{:>10}'.format(
            input_dict['close_price_skewness'],
            '',
            'Kurtosis Price........  ',
            input_dict['close_price_kurtosis']))
        
        
        # Return Distribution statistics        
        
        # Mean & Standard Deviation of Returns
        print('Mean Return.......... %{:>10}{:<6}{}{:>10}'.format(
            input_dict['close_return_mean'],
            '',
            'St. Dev. Return....... %',
            input_dict['close_return_std_dev']))
       
        # Skewness & Kurtosis of Returns
        print('Skewness Return......  {:>10}{:<6}{}{:>10}'.format(
            input_dict['close_return_skewness'],
            '',
            'Kurtosis Return.......  ',
            input_dict['close_return_kurtosis']))

        # Efficiency Ratio & Annualized Volatility
        print('Efficiency Ratio..... %{:>10}{:<6}{}{:>10}'.format(
            input_dict['efficiency_ratio'],
            '',
            'Annualized Vol........ %',
            input_dict['close_return_ann_vol']))
               
        # Closing line
        print('='*78)


    def _dict_format(self, input_dict):
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
        
        # List of parameters not to be formatted
        format_pass = self.df_dict['df_format_pass']
        
        # Set decimal format
        dp2 = Decimal(10) ** -2  # (equivalent to Decimal '0.01')
        
        # For each entry in the dictionary
        for key, value in input_dict.items():
            
            # If the value is a floating point number
            if type(value) in (float, np.float64) and key not in format_pass:
                
                # Apply the decimal formatting and convert to string
                str_input_dict[key] = str(Decimal(value).quantize(dp2))
            else:
                
                # Otherwise just convert to string
                str_input_dict[key] = str(value)
    
        return str_input_dict
    
    
    def _norgate_name_dict(self):
        alldatabasenames = norgatedata.databases()
        self.norgate_name_dict = {}
        for database in alldatabasenames:
            databasecontents = norgatedata.database(database)
            for dicto in databasecontents:
                key = dicto['symbol']
                value = dicto['securityname']
                if database == 'Continuous Futures':
                    #if '_CCB' in key:
                    self.norgate_name_dict[key] = value
                elif database == 'Futures':
                    pass
                else:
                    self.norgate_name_dict[key] = value
        
        return self    


            