import numpy as np
import random
from technicalmethods.methods import Indicators

class Entry():
    
    @classmethod
    def _entry_signal(
            cls, df=None, entry_type=None, ma1=None, ma2=None, ma3=None, 
            ma4=None, simple_ma=None, entry_period=None, entry_oversold=None, 
            entry_overbought=None, entry_threshold=None, 
            entry_acceleration_factor=None):
        """
        Calculate trade entry signals

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        entry_type : Str, optional
            The entry strategy. The default is '2ma'.
        ma1 : Int, optional
            The first moving average period.
        ma2 : Int, optional
            The second moving average period.
        ma3 : Int, optional
            The third moving average period.
        ma4 : Int, optional
            The fourth moving average period.
        simple_ma : Bool
            Whether to calculate a simple or exponential moving average. The 
            default is True.
        entry_period : Int
            The number of days to use in the entry strategy. The default is 14.
        entry_oversold : Int
            The oversold level to use in the entry strategy. 
        entry_overbought : Int
            The overbought level to use in the entry strategy.
        entry_threshold : Float
            The entry threshold used for momentum / volatility strategies. 
            The default is 0 for momentum and 1.5 for volatility.
        entry_acceleration_factor : Float
            The acceleration factor used in the Parabolic SAR entry signal. 
            The default is 0.02.

        Returns
        -------
        df : DataFrame
            The OHLC data.
        start : Int
            The first valid row to start calculating trade information from.
        signal : Series
            The series of Buy / Sell signals.

        """
        
        # Double Moving Average Crossover
        if entry_type == '2ma':
            df, start, signal = cls._entry_double_ma_crossover(
                df=df, ma1=ma1, ma2=ma2, simple_ma=simple_ma)
        
        # Triple Moving Average Crossover
        elif entry_type == '3ma':
            df, start, signal = cls._entry_triple_ma_crossover(
                df, ma1=ma1, ma2=ma2, ma3=ma3, simple_ma=simple_ma)
        
        # Quad Moving Average Crossover
        elif entry_type == '4ma':
            df, start, signal = cls._entry_quad_ma_crossover(
                df, ma1=ma1, ma2=ma2, ma3=ma3, ma4=ma4, simple_ma=simple_ma)        
        
        # Parabolic SAR
        elif entry_type == 'sar':
            df, start, signal = cls._entry_parabolic_sar(
                df=df, acceleration_factor=entry_acceleration_factor)
        
        # Channel Breakout
        elif entry_type == 'channel_breakout':
            df, start, signal = cls._entry_channel_breakout(
                df, time_period=entry_period)
        
        # Stochastic Crossover
        elif entry_type == 'stoch_cross':
            df, start, signal = cls._entry_stochastic_crossover(
                df, time_period=entry_period, oversold=entry_oversold, 
                overbought=entry_overbought)
        
        # Stochastic Over Under
        elif entry_type == 'stoch_over_under':
            df, start, signal = cls._entry_stochastic_over_under(
                df, time_period=entry_period, oversold=entry_oversold, 
                overbought=entry_overbought)
        
        # Stochastic Pop
        elif entry_type == 'stoch_pop':
            df, start, signal = cls._entry_stochastic_pop(
                df, time_period=entry_period, oversold=entry_oversold, 
                overbought=entry_overbought)
        
        # Relative Strength Index
        elif entry_type == 'rsi':
            df, start, signal = cls._entry_rsi(
                df, time_period=entry_period, oversold=entry_oversold, 
                overbought=entry_overbought)
        
        # Commodity Channel Index
        elif entry_type == 'cci':
            df, start, signal = cls._entry_commodity_channel_index(
                df, time_period=entry_period, threshold=entry_threshold)
        
        # Momentum
        elif entry_type == 'momentum':
            df, start, signal = cls._entry_momentum(
                df, time_period=entry_period, threshold=entry_threshold)
        
        # Volatility
        elif entry_type == 'volatility':
            df, start, signal = cls._entry_volatility(
                df, time_period=entry_period, threshold=entry_threshold)
            
        return df, start, signal    
    
    
    @staticmethod
    def _entry_double_ma_crossover(df, ma1, ma2, simple_ma):
        """
        Entry signal for Double Moving Average Crossover strategy

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        ma1 : Int
            The faster moving average.
        ma2 : Int
            The slower moving average.
        simple_ma : Bool
            Whether to calculate a simple or exponential moving average. The 
            default is True.    

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals.

        """
        
        if simple_ma:
            
            # Create short and long simple moving averages  
            ma_1 = np.array(df['Close'].rolling(ma1).mean())
            ma_2 = np.array(df['Close'].rolling(ma2).mean())
        
        else:
            # Create short and long exponential moving averages
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
        Entry signal for Triple Moving Average Crossover strategy

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        ma1 : Int
            The first moving average. The default is 4.
        ma2 : Int
            The second moving average. The default is 9.
        ma3 : Int
            The third moving average. The default is 18.
        simple_ma : Bool
            Whether to calculate a simple or exponential moving average. The 
            default is True.
            
        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals.

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
        Entry signals for Quad Moving Average strategy
    
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
        simple_ma : Bool
            Whether to calculate a simple or exponential moving average. The 
            default is True.
            
        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals.
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
        Entry signals based on Parabolic SAR

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        acceleration_factor : Float
            The acceleration factor to use.

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

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
        
        # Assign the series to the OHLC data
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
        Entry signals based on a channel breakout.        

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation. The default 
            is 20.

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

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
    
        # Assign the series to the OHLC data
        df['rolling_high_close_entry'] = rolling_high_close
        df['rolling_low_close_entry'] = rolling_low_close 
        
        return df, start, trade_signal
    
    
    @staticmethod
    def _entry_stochastic_crossover(df, time_period, oversold, overbought):
        """
        Entry signals based on slow k / slow d stochastics crossing.

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        oversold : Int
            The oversold level to use.
        overbought : Int
            The overbought level to use.

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

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
    
        # Assign the series to the OHLC data
        df['slow_k_entry'] = slow_k
        df['slow_d_entry'] = slow_d        
   
        return df, start, trade_signal


    @staticmethod
    def _entry_stochastic_over_under(df, time_period, oversold, overbought):
        """
        Entry signals based on slow k / slow d stochastics crossing 
        overbought / oversold levels.

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        oversold : Int
            The oversold level to use.
        overbought : Int
            The overbought level to use.

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

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
    
        # Assign the series to the OHLC data
        df['slow_k_entry'] = slow_k
        df['slow_d_entry'] = slow_d         
   
        return df, start, trade_signal    
    
    
    @staticmethod
    def _entry_stochastic_pop(df, time_period, oversold, overbought):
        """
        Entry signals based on the Stochastic Pop method 

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        oversold : Int
            The oversold level to use.
        overbought : Int
            The overbought level to use.

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

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
    
        # Assign the series to the OHLC data
        df['slow_k_entry'] = slow_k
        df['slow_d_entry'] = slow_d           
   
        return df, start, trade_signal    
    
    
    @staticmethod
    def _entry_rsi(df, time_period, oversold, overbought):
        """
        Entry signals based on the Relative Strength Index 

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        oversold : Int
            The oversold level to use.
        overbought : Int
            The overbought level to use.

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

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
    
        # Assign the series to the OHLC data
        df['RSI_entry'] = rsi
   
        return df, start, trade_signal
    
    
    @staticmethod
    def _entry_commodity_channel_index(df, time_period, threshold):
        """
        Entry signals based on the Commodity Channel Index

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        threshold : TYPE
            DESCRIPTION.

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

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
    
        # Assign the series to the OHLC data
        df['CCI_entry'] = cci
   
        return df, start, trade_signal
    
    
    @staticmethod
    def _entry_momentum(df, time_period, threshold):
        """
        Entry signals based on n-day momentum

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation. The default 
            is 10.
        threshold : Float
            The threshold used for taking signals. The default is 0.

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

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
    
        # Assign the series to the OHLC data
        df['n_day_close'] = n_day_close
        df['momentum'] = df['Close'] - df['n_day_close']
    
        return df, start, trade_signal
    
    
    @staticmethod
    def _entry_volatility(df, time_period, threshold):
        """
        Entry signals based on a volatility breakout.

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        threshold : Float
            The threshold used for taking signals. The default is 1.5.

        Returns
        -------
        df : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

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
            if ((df['Close'][row] - df['Close'][row-1]) 
                > (atr[row] * threshold)):
                
                # Set the position signal to long
                position_signal[row] = 1
                
                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]
            
            # If the decrease in closing price exceeds the atr * threshold
            elif ((df['Close'][row-1] - df['Close'][row]) 
                  > (atr[row] * threshold)):
                
                # Set the position signal to short
                position_signal[row] = -1
    
                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]
            
            # Otherwise, take no action
            else:                
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]
    
        # Assign the series to the OHLC data
        df['ATR_entry'] = atr
   
        return df, start, trade_signal
    

class Exit():
    
    @classmethod
    def _exit_and_stop_signals(
            cls, df, position_size, trade_number, end_of_day_position, 
            sip_price=None ,exit_type=None, exit_amount=None, 
            exit_period=None, stop_type=None, stop_amount=None, 
            stop_period=None, exit_threshold=None, exit_oversold=None, 
            exit_overbought=None, exit_acceleration_factor=None, 
            exit_trailing_close=None, exit_profit_target=None, 
            stop_initial_dollar_loss=None, stop_profit_target=None, 
            stop_trailing_close=None, stop_trailing_high_low=None):
        """
        Calculate trade exit and stop signals.

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        position_size : Int, optional
            The number of units to trade. The default is based on equity.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        sip_price : Bool
            Whether to set the SIP of the Parabolic SAR exit to n-day 
            high / low or to the high of the previous trade. The default is 
            False.
        exit_type : Str, optional
            The exit strategy. The default is 'trailing_stop'.
        exit_amount : Float
            The dollar exit amount. The default is $1000.00.
        exit_period : Int, optional
            The number of days to use in the exit strategy. The default is 5.
        stop_type : Str, optional
            The stop strategy. The default is 'initial_dollar'.
        stop_amount : Float
            The dollar stop amount. The default is $500.00. 
        stop_period : Int, optional
            The number of days to use in the stop strategy. The default is 5.
        exit_threshold : Float
            The exit threshold used for the volatility strategy. 
            The default is 1.
        exit_oversold : Int, optional
            The oversold level to use in the exit strategy. 
        exit_overbought : Int, optional
            The overbought level to use in the exit strategy.
        exit_acceleration_factor : Float
            The acceleration factor used in the Parabolic SAR exit signal. 
            The default is 0.02.
        exit_trailing_close : Series, optional
            The exit levels for each trade based on the trailing close
        exit_profit_target : Series, optional
            The exit levels for each trade based on a profit target
        stop_initial_dollar_loss : Series, optional
            The stop levels for each trade based on a dollar loss from the 
            entry level.
        stop_profit_target : Series, optional
            The stop levels for each trade based on a profit target
        stop_trailing_close : Series, optional
            The stop levels for each trade based on the trailing close
        stop_trailing_high_low : Series, optional
            The stop levels for each trade based on the trailing high / low.

        Returns
        -------
        df : DataFrame
            The OHLC data

        """
        # Generate the exit signals
        df, df['exit_signal'] = cls._exit_signal(
            df=df, position_size=position_size, exit_amount=exit_amount, 
            exit_type=exit_type, exit_period=exit_period, 
            exit_threshold=exit_threshold, trade_number=trade_number, 
            end_of_day_position=end_of_day_position, 
            exit_oversold=exit_oversold, exit_overbought=exit_overbought,
            exit_acceleration_factor=exit_acceleration_factor, 
            sip_price=sip_price, exit_trailing_close=exit_trailing_close, 
            exit_profit_target=exit_profit_target)    
       
        # Generate the stop signals
        df, df['stop_signal'] = cls._stop_signal(
            df=df, stop_type=stop_type, stop_period=stop_period, 
            stop_amount=stop_amount, position_size=position_size, 
            trade_number=trade_number, 
            end_of_day_position=end_of_day_position, 
            stop_initial_dollar_loss=stop_initial_dollar_loss, 
            stop_profit_target=stop_profit_target, 
            stop_trailing_close=stop_trailing_close, 
            stop_trailing_high_low=stop_trailing_high_low)
                    
        return df    
    
    
    @classmethod
    def _exit_signal(cls, df, exit_type=None, exit_period=None, 
                     exit_amount=None, exit_threshold=None,
                     position_size=None, trade_number=None, 
                     end_of_day_position=None, exit_oversold=None, 
                     exit_overbought=None, exit_acceleration_factor=None, 
                     sip_price=None, exit_trailing_close=None, 
                     exit_profit_target=None):
        """
        Calculate trade exit signals.

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        exit_type : Str, optional
            The exit strategy. The default is 'trailing_stop'.
        exit_period : Int, optional
            The number of days to use in the exit strategy. The default is 5.
        exit_amount : Float
            The dollar exit amount. The default is $1000.00.
        exit_threshold : Float
            The exit threshold used for the volatility strategy. 
        position_size : Int, optional
            The number of units to trade. The default is based on equity.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        exit_oversold : Int, optional
            The oversold level to use in the exit strategy. 
        exit_overbought : Int, optional
            The overbought level to use in the exit strategy.
        exit_acceleration_factor : Float
            The acceleration factor used in the Parabolic SAR exit signal. 
            The default is 0.02.
        sip_price : Bool
            Whether to set the SIP of the Parabolic SAR exit to n-day 
            high / low or to the high of the previous trade. The default is 
            False.

        Returns
        -------
        df : DataFrame
            The OHLC data
        exit : Series
            The exit signals.


        df : DataFrame
            The OHLC data

        """

        # Parabolic SAR Exit    
        if exit_type == 'sar':
            df, exit = Exit._exit_parabolic_sar(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position, 
                time_period=exit_period, 
                acceleration_factor=exit_acceleration_factor, 
                sip_price=sip_price)
        
        # Support / Resistance Exit
        elif exit_type == 'sup_res':
            df, exit = Exit._exit_support_resistance(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position,
                time_period=exit_period)        
        
        # Trailing RSI Exit
        elif exit_type == 'rsi_trail':
            df, exit = Exit._exit_rsi_trail(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position, 
                time_period=exit_period, oversold=exit_oversold, 
                overbought=exit_overbought)        
        
        # Key Reversal Day Exit
        elif exit_type == 'key_reversal':
            df, exit = Exit._exit_key_reversal(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position,
                time_period=exit_period)        
        
        # Trailing Stop Exit
        elif exit_type == 'trailing_stop':
            df, exit = Exit._exit_dollar(
                df=df, trigger_value=exit_trailing_close, 
                exit_level='trail_close', trade_number=trade_number, 
                end_of_day_position=end_of_day_position)        
        
        # Volatility Breakout Exit
        elif exit_type == 'volatility':
            df, exit = Exit._exit_volatility(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position,
                time_period=exit_period, threshold=exit_threshold)        
        
        # Stochastic Crossover Exit
        elif exit_type == 'stoch_cross':
            df, exit = Exit._exit_stochastic_crossover(
                df=df, time_period=exit_period, trade_number=trade_number, 
                end_of_day_position=end_of_day_position)        
        
        # Profit Target Exit
        elif exit_type == 'profit_target':
            df, exit = Exit._exit_dollar(
                df=df, trigger_value=exit_profit_target, 
                exit_level='profit_target', trade_number=trade_number, 
                end_of_day_position=end_of_day_position) 
        
        # N-day Range Exit
        elif exit_type == 'nday_range':    
            df, exit = Exit._exit_nday_range(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position, 
                time_period=exit_period)    
        
        # Random Exit
        elif exit_type == 'random':
            df, exit = Exit._exit_random(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position)

        return df, exit


    @classmethod
    def _stop_signal(
            cls, df, stop_type=None, stop_period=None, stop_amount=None, 
            position_size=None, trade_number=None, end_of_day_position=None,
            stop_initial_dollar_loss=None, stop_profit_target=None, 
            stop_trailing_close=None, stop_trailing_high_low=None):
        """
        Calculate trade stop signals

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        stop_type : Str
            The type of stop to use.
        stop_period : Int
            The length of time for the indicator calculation.
        stop_amount : Float
            The dollar stop amount. The default is $500.00. 
        position_size : Int, optional
            The number of units to trade. The default is based on equity.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        stop_initial_dollar_loss : Series, optional
            The stop levels for each trade based on a dollar loss from the 
            entry level.
        stop_profit_target : Series, optional
            The stop levels for each trade based on a profit target
        stop_trailing_close : Series, optional
            The stop levels for each trade based on the trailing close
        stop_trailing_high_low : Series, optional
            The stop levels for each trade based on the trailing high / low.

        Returns
        -------
        df : DataFrame
            The OHLC data
        stop : Series
            The stop signals.

        """
        
        # Initial Dollar Loss Stop
        if stop_type == 'initial_dollar':
            df, stop = Exit._exit_dollar(
                df=df, trigger_value=stop_initial_dollar_loss, 
                exit_level='initial', trade_number=trade_number, 
                end_of_day_position=end_of_day_position)    
        
        # Support / Resistance Stop
        elif stop_type == 'sup_res':
            df, stop = Exit._exit_support_resistance(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position,
                time_period=stop_period)
        
        # Immediate Profit Stop
        elif stop_type == 'immediate_profit':
            df, stop = Exit._exit_immediate_profit(
                df=df, trade_number=trade_number, 
                end_of_day_position=end_of_day_position,
                time_period=stop_period)
        
        # Breakeven Stop
        elif stop_type == 'breakeven':
            df, stop = Exit._exit_dollar(
                df=df, trigger_value=stop_profit_target, 
                exit_level='breakeven', trade_number=trade_number, 
                end_of_day_position=end_of_day_position, 
                trade_high_price=df['raw_td_high_price'], 
                trade_low_price=df['raw_td_low_price'])
        
        # Trailing Stop (Closing Price)
        elif stop_type == 'trail_close':
            df, stop = Exit._exit_dollar(
                df=df, trigger_value=stop_trailing_close, 
                exit_level='trail_close', trade_number=trade_number, 
                end_of_day_position=end_of_day_position)
        
        # Trailing Stop (High / Low Price)
        elif stop_type == 'trail_high_low':
            df, stop = Exit._exit_dollar(
                df=df, trigger_value=stop_trailing_high_low, 
                exit_level='trail_high_low', trade_number=trade_number, 
                end_of_day_position=end_of_day_position) 
                    
        return df, stop
    
    
    
    @staticmethod        
    def _exit_parabolic_sar_v1(
            df, trade_number, end_of_day_position, time_period, 
            acceleration_factor, sip_price):
        """
        Calculate exit based on a Parabolic SAR.

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the initial rolling lookback.
        acceleration_factor : Float
            The acceleration factor to use.
        sip_price : Bool
            Whether to set the SIP of the Parabolic SAR exit to n-day 
            high / low or to the high of the previous trade. The default is 
            False.

        Returns
        -------
        df : DataFrame
            The OHLC data
        parabolic_sar_exit : Series
            The exit signals.

        """
        
        # Calculate rolling min and max closing prices based on time period
        rolling_high_close = df['Close'].rolling(time_period).max()
        rolling_low_close = df['Close'].rolling(time_period).min()
        
        # Initialize zero arrays to store data
        sar = np.array([0.0]*len(df))
        parabolic_sar_exit = np.array([0]*len(df))        
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:
                
                # If it is the first trade in the data
                if trade_number[row] == 1:
                    
                    # If there is a long position
                    if end_of_day_position[row] > 0:
                        
                        # Set the initial point to the n-day low close
                        initial_point = rolling_low_close[row]
                    
                    # If there is a short position
                    else:
                        
                        # Set the initial point to the n-day high close
                        initial_point = rolling_high_close[row]
                
                # Find the row that relates to the trade entry
                trade_first_row = df.index.get_loc(
                    df[trade_number==trade_number[row]].index[0])
                trade_row_num = row - trade_first_row
                
                # If there is a long position
                if end_of_day_position[row] > 0:
                    
                    # If it is the first trade in the data
                    if trade_number[row] == 1:

                        # Set the initial point to the n-day low
                        initial_point = rolling_low_close[row]

                    # For every other trade
                    else:
                        # If the sip_price flag is True
                        if sip_price:
                            
                            # Set the initial point to the n-day low close
                            initial_point = rolling_low_close[row]
                        
                        # If the sip_price flag is False
                        else:
                            # Set the initial point to the previous trade low 
                            # close
                            initial_point = min(
                                df[trade_number==trade_number[row]-1]['Close'])
                    
                    # If it is the trade entry day
                    if trade_row_num == 0:
                        
                        # Set the sar to the initial point
                        sar[row] = initial_point
                        
                        # Set the dynamic af to the initial acceleration factor
                        af = acceleration_factor

                    # For every other day in the trade
                    else:
                        # Set the sar to the previous day's sar plus the 
                        # acceleration factor multiplied by the difference 
                        # between the todays extreme price and yesterdays sar
                        sar[row] = sar[row-1] + (
                            af * (rolling_high_close[row] - sar[row-1]))
                        
                        # Increment the acceleration factor by the input value 
                        # to a max of 0.2
                        af = min(af + acceleration_factor, 0.2)
                    
                    # If todays close is less than the sar    
                    if df['Close'][row] < sar[row]:
                        
                        # Set the exit signal to -1
                        parabolic_sar_exit[row] = -1
                        
                    else:
                        # Set the exit signal to 0
                        parabolic_sar_exit[row] = 0                    
                
                # If there is a short position        
                elif end_of_day_position[row] < 0:
                    
                    # If it is the first trade in the data
                    if trade_number[row] == 1:
                        
                        # Set the initial point to the n-day high close
                        initial_point = rolling_high_close[row]

                    # For every other trade
                    else:
                        # If the sip_price flag is True
                        if sip_price:
                            
                            # Set the initial point to the n-day low close
                            initial_point = rolling_high_close[row]
                        
                        # If the sip_price flag is False
                        else:
                            # Set the initial point to the previous trade high 
                            # close
                            initial_point = max(
                                df[trade_number==trade_number[row]-1]['Close'])                
                    
                    # If it is the trade entry day
                    if trade_row_num == 0:
                        
                        # Set the sar to the initial point
                        sar[row] = initial_point
                        
                        # Set the dynamic af to the initial acceleration factor
                        af = acceleration_factor
                    
                    # For every other day in the trade
                    else:
                        
                        # Set the sar to the previous day's sar plus the 
                        # acceleration factor multiplied by the difference 
                        # between the todays extreme price and yesterdays sar
                        sar[row] = sar[row-1] + (af * (
                            rolling_low_close[row] - sar[row-1]))
                        
                        # Increment the acceleration factor by the input value 
                        # to a max of 0.2
                        af = max(af + acceleration_factor, 0.2)
                    
                    # If todays close is greater than the sar    
                    if df['Close'][row] > sar[row]:
                        
                        # Set the exit signal to 1
                        parabolic_sar_exit[row] = 1
                    else:
                        
                        # Set the exit signal to 0
                        parabolic_sar_exit[row] = 0                                 
                
                else:
                    # Set the sar to zero
                    sar[row] = 0
                    
                    # Set the exit signal to 0
                    parabolic_sar_exit[row] = 0
        
        # Set the DataFrame column to the numpy array            
        df['rolling_high_close_sar_exit'] = rolling_high_close            
        df['rolling_low_close_sar_exit'] = rolling_low_close
        df['sar_exit'] = sar
                    
        return df, parabolic_sar_exit 


    @staticmethod        
    def _exit_parabolic_sar(
            df, trade_number, end_of_day_position, time_period, 
            acceleration_factor, sip_price):
        """
        Calculate exit based on a Parabolic SAR.

        Parameters
        ----------
        df : DataFrame
            The OHLC data
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the initial rolling lookback.
        acceleration_factor : Float
            The acceleration factor to use.
        sip_price : Bool
            Whether to set the SIP of the Parabolic SAR exit to n-day 
            high / low or to the high of the previous trade. The default is 
            False.

        Returns
        -------
        df : DataFrame
            The OHLC data
        parabolic_sar_exit : Series
            The exit signals.

        """
        
        # Extract high, low and close series from the DataFrame
        high = df['High']
        low = df['Low']

        # Calculate rolling min and max closing prices based on time period
        rolling_high = df['High'].rolling(time_period).max()
        rolling_low = df['Low'].rolling(time_period).min()
        
        # Initialize zero arrays to store data
        sar = np.array([0.0]*len(df))
        parabolic_sar_exit = np.array([0]*len(df))  
        ep = np.array([0.0]*len(df))
        af = np.array([0.0]*len(df))
        ep_sar_diff = np.array([0.0]*len(df))
        af_x_diff = np.array([0.0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:

                # Find the row that relates to the trade entry
                trade_first_row = df.index.get_loc(
                    df[trade_number==trade_number[row]].index[0])
                trade_row_num = row - trade_first_row

                # If it is the trade entry day
                if trade_row_num == 0:

                    # If it is the first trade in the data
                    if trade_number[row] == 1:

                        # If there is a long position
                        if end_of_day_position[row] > 0:
                            
                                # Set the initial point to the n-day low
                                initial_point = rolling_low[row]

                                # Set the extreme price to the day's high
                                ep[row] = high[row]

                        # If there is a short position
                        elif end_of_day_position[row] < 0:

                                # Set the initial point to the n-day high
                                initial_point = rolling_high[row]  

                                # Set the extreme price to the day's low
                                ep[row] = low[row]  

                    # For every other trade
                    else:
                        # If there is a long position
                        if end_of_day_position[row] > 0:

                            # If the sip_price flag is True
                            if sip_price:
                            
                                # Set the initial point to the n-day low close
                                initial_point = rolling_low[row]
                        
                            # If the sip_price flag is False
                            else:
                                # Set the initial point to the previous trade low 
                                initial_point = min(
                                    df[trade_number==trade_number[row]-1]['Low'])

                            # Set the extreme price to the day's high
                            ep[row] = high[row]        

                        # If there is a short position
                        elif end_of_day_position[row] < 0:

                            # If the sip_price flag is True
                            if sip_price:
                            
                                # Set the initial point to the n-day low close
                                initial_point = rolling_high[row]
                        
                            # If the sip_price flag is False
                            else:
                                # Set the initial point to the previous trade high 
                                initial_point = max(
                                    df[trade_number==trade_number[row]-1]['High'])

                                # Set the extreme price to the day's high
                                ep[row] = low[row]
                
                        # Set the sar to the initial point
                        sar[row] = initial_point
                            
                        # Set the dynamic af to the initial acceleration factor
                        af[row] = acceleration_factor

                # If it is not the trade entry day
                else:
                    # If the previous day was long
                    if end_of_day_position[row-1] == 1:
                        
                        # If the previous day's sar was greater than the previous day's 
                        # low
                        if sar[row-1] > low[row-1]:

                            # Set the signal to exit
                            parabolic_sar_exit[row-1] = -1

                            # Set the new sar to the previous trades extreme price
                            sar[row] = ep[row-1]

                        # If the previous day's sar plus the acceleration factor 
                        # multiplied by the difference between the extreme price and 
                        # the sar is greater than the lowest low of the previous 2 days     
                        elif (sar[row-1] + af_x_diff[row-1] 
                            > min(low[row-1], low[row-2])):
                            
                            # Set the sar to the lowest low of the previous 2 days
                            sar[row] = min(low[row-1], low[row-2])
                            
                        # Otherwise    
                        else: 
                            # Set the sar to the previous day's sar plus the 
                            # acceleration factor multiplied by the difference between 
                            # the extreme price and the sar
                            sar[row] = sar[row-1] + af_x_diff[row-1]
                            
                    # Otherwise if the previous day was short
                    elif end_of_day_position[row-1] == -1:
                        # If the previous day's sar was less than the previous day's 
                        # high
                        if sar[row-1] < high[row-1]:
                            
                            # Set the signal to exit
                            parabolic_sar_exit[row-1] = 1
                            
                            # Set the new sar to the previous trades extreme price
                            sar[row] = ep[row-1]
                        
                        # If the previous day's sar less the acceleration factor 
                        # multiplied by the difference between the extreme price and 
                        # the sar is less than the highest high of the previous 2 days 
                        elif (sar[row-1] - af_x_diff[row-1] 
                            < max(high[row-1], high[row-2])):
                            
                            # Set the sar to the highest high of the previous 2 days
                            sar[row] = max(high[row-1], high[row-2])    
                        
                        # Otherwise
                        else:    
                            # Set the sar to the previous day's sar minus the 
                            # acceleration factor multiplied by the difference between 
                            # the extreme price and the sar
                            sar[row] = sar[row-1] - af_x_diff[row-1]
            
                # If the current trade direction is long
                if end_of_day_position[row] == 1:
                    
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
        
        # Set the DataFrame columns to the numpy arrays
        df['rolling_high_sar_exit'] = rolling_high            
        df['rolling_low_sar_exit'] = rolling_low
        df['sar_exit'] = sar
        df['ep_sar_exit'] = ep
        df['af_sar_exit'] = af
        df['ep_sar_diff_exit'] = ep_sar_diff
        df['af_x_diff_exit'] = af_x_diff        
                    
        return df, parabolic_sar_exit     
    
    
    @staticmethod
    def _exit_rsi_trail(df, trade_number, end_of_day_position, time_period, 
                        oversold, overbought):
        """
        Calculate exit based on a trailing RSI - a down close when the RSI is 
        overbought when long or an up close when the RSI is oversold when 
        short.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the RSI calculation.
        oversold : Int
            The oversold level to use.
        overbought : Int
            The overbought level to use.

        Returns
        -------
        df : DataFrame
            The OHLC data.
        rsi_trail_exit : Series
            The exit signals.

        """

        # Calculate RSI
        rsi = Indicators.RSI(close=df['Close'], time_period=time_period)
                
        # Create an empty array to store the signals
        rsi_trail_exit = np.array([0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:
                
                # If there is a long position
                if end_of_day_position[row] > 0:
                    
                    # If todays close is less than the previous days close
                    # and todays RSI is greater than the overbought level 
                    if (df['Close'][row] < df['Close'][row-1] 
                        and rsi[row] > overbought):
                        
                        # Set the exit signal to -1
                        rsi_trail_exit[row] = -1

                # If there is a short position
                elif end_of_day_position[row] < 0:
                    
                    # If todays close is greater than the previous days close
                    # and todays RSI is less than the oversold level
                    if (df['Close'][row] > df['Close'][row-1] 
                        and rsi[row] < oversold):
                        
                        # Set the exit signal to 1
                        rsi_trail_exit[row] = 1
                else:
                    # Set the exit signal to 0
                    rsi_trail_exit[row] = 0
        
        # Set the DataFrame column to the numpy array
        df['RSI_exit'] = rsi
                    
        return df, rsi_trail_exit
    
    
    @staticmethod
    def _exit_key_reversal(
            df, trade_number, end_of_day_position, time_period):
        """
        Calculate exit based on a key reversal day - a new high combined with 
        a down close when long or a new low combined with an up close when 
        short.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the rolling lookback calculation.

        Returns
        -------
        df : DataFrame
            The OHLC data.
        key_reversal_exit : Series
            The exit signals.

        """
        # Calculate rolling high and low prices based on time period
        rolling_high = df['High'].rolling(time_period).max()
        rolling_low = df['Low'].rolling(time_period).min()
        
        # Create an empty array to store the signals
        key_reversal_exit = np.array([0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:
                
                # If there is a long position
                if end_of_day_position[row] > 0:
                    
                    # If the n-day high today is greater than yesterdays n-day 
                    # high and todays close is less than the previous days 
                    # close
                    if (rolling_high[row] > rolling_high[row-1] 
                        and df['Close'][row] < df['Close'][row-1]):
                        
                        # Set the exit signal to -1
                        key_reversal_exit[row] = -1

                # If there is a short position
                elif end_of_day_position[row] < 0:
                    
                    # If the n-day low today is less than yesterdays n-day 
                    # low and todays close is greater than the previous days 
                    # close
                    if (rolling_low[row] < rolling_low[row-1] 
                        and df['Close'][row] > df['Close'][row-1]):
                        
                        # Set the exit signal to -1
                        key_reversal_exit[row] = 1

                else:
                    # Set the exit signal to 0
                    key_reversal_exit[row] = 0
        
        # Set the DataFrame columns to the numpy arrays            
        df['rolling_high_key'] = rolling_high
        df['rolling_low_key'] = rolling_low
                    
        return df, key_reversal_exit
    
    
    @staticmethod
    def _exit_volatility(
            df, trade_number, end_of_day_position, time_period, threshold):
        """
        Calculate exit based on an increase in volatility - a fall in price 
        greater than the ATR * Threshold when long or a rise in price greater 
        than the ATR * Threshold when when short.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the ATR calculation.
        threshold : Float
            The multiplier for the ATR used to trigger the exit.

        Returns
        -------
        df : DataFrame
            The OHLC data
        volatility_exit : Series
            The exit signals.

        """
        
        # Create an empty array to store the signals
        volatility_exit = np.array([0]*len(df))
    
        # Calculate ATR
        atr = Indicators.ATR(df['High'], df['Low'], df['Close'], time_period)
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:
                
                # If there is a long position
                if end_of_day_position[row] > 0:
                    
                    # If the decrease in closing price from yesterday to today 
                    # is greater than the ATR * Threshold
                    if ((df['Close'][row] - df['Close'][row-1]) 
                        > (atr[row] * threshold)):
                        
                        # Set the exit signal to -1
                        volatility_exit[row] = -1

                # If there is a short position
                elif end_of_day_position[row] < 0:
                    
                    # If the increase in closing price from yesterday to today 
                    # is greater than the ATR * Threshold
                    if ((df['Close'][row-1] - df['Close'][row]) 
                        > (atr[row] * threshold)):
                        
                        # Set the exit signal to 1
                        volatility_exit[row] = 1

                else:
                    # Set the exit signal to 0
                    volatility_exit[row] = 0
        
        # Set the DataFrame column to the numpy array            
        df['ATR_exit'] = atr
                   
        return df, volatility_exit
    
    
    @staticmethod
    def _exit_stochastic_crossover(
            df, trade_number, end_of_day_position, time_period):
        """
        Calculate exit based on a stochastic crossover - if the slow k crosses 
        below the slow d when long or if the slow k crosses above the slow d 
        when short.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the fast k calculation.

        Returns
        -------
        df : DataFrame
            The OHLC data
        stoch_cross_exit : Series
            The exit signals.

        """
        
        # Calculate slow k  and slow d 
        slow_k, slow_d = Indicators.stochastic(
            df['High'], df['Low'], df['Close'], fast_k_period=time_period, 
            fast_d_period=3, slow_k_period=3, slow_d_period=3, 
            output_type='slow')
    
        # Create an empty array to store the signals
        stoch_cross_exit = np.array([0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:
            
                # If there is a long position
                if end_of_day_position[row] > 0:
                
                    # If the slow k crosses below the slow d 
                    if (slow_k[row] < slow_d[row] 
                        and slow_k[row-1] > slow_d[row-1]): 
                    
                        # Set the exit signal to -1
                        stoch_cross_exit[row] = -1
               
                # If there is a short position    
                elif end_of_day_position[row] < 0:
                
                    # If the slow k crosses above the slow d
                    if (slow_k[row] > slow_d[row] 
                        and slow_k[row-1] < slow_d[row-1]):
                    
                        # Set the exit signal to 1                        
                        stoch_cross_exit[row] = 1
               
                else:
                    # Set the exit signal to 0
                    stoch_cross_exit[row] = 0
        
        # Set the DataFrame columns to the numpy arrays
        df['slow_k_exit'] = slow_k
        df['slow_d_exit'] = slow_d            
                    
        return df, stoch_cross_exit
    
    
    @staticmethod
    def _exit_random(df, trade_number, end_of_day_position):
        """
        Calculate exit based on the first losing day after a random time 
        interval.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.

        Returns
        -------
        df : DataFrame
            The OHLC data
        random_exit : Series
            The exit signals.

        """
        # Generate a random number of days between 5 and 20
        exit_days = random.randint(5,20)
        
        # Create an empty array to store the signals
        random_exit = np.array([0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # Find the row that relates to the trade entry
            trade_first_row = df.index.get_loc(
                    df[trade_number==trade_number[row]].index[0])
            trade_row_num = row - trade_first_row
            
            # If there is a trade on
            if trade_number[row] != 0:
                
                # If the trade has been on for the random number of days
                if trade_row_num > exit_days-1:
                    
                    # If there is a long position
                    if end_of_day_position[row] > 0:
                        
                        # If todays close is less than the previous days close
                        if df['Close'][row] < df['Close'][row-1]:
                            
                            # Set the exit signal to -1
                            random_exit[row] = -1
                        
                    # If there is a short position
                    elif end_of_day_position[row] < 0:

                        # If todays close is greater than the previous days 
                        # close
                        if df['Close'][row] > df['Close'][row-1]:
                            
                            # Set the exit signal to 1
                            random_exit[row] = 1

                    else:
                        # Set the exit signal to 0
                        random_exit[row] = 0
        
        # Set the DataFrame column to the numpy array
        df['random_days_exit'] = exit_days        
                        
        return df, random_exit
    
    
    @classmethod
    def _exit_dollar(
            cls, df, trigger_value, trade_number, 
                end_of_day_position, trade_high_price=None, 
                trade_low_price=None, exit_level=None):
        """
        Calculate exit based on a dollar amount.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trigger_value : Series
            The series to trigger exit.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        trade_high_price : Series
            The high price of the trade.
        trade_low_price : Series
            The low price of the trade.
        exit_level : Str
            The type of exit strategy.

        Returns
        -------
        df : DataFrame
            The OHLC data
        exit : Series
            Exit signals.

        """
        
        # Calculate exit signal based on a profit target
        if exit_level == 'profit_target':
            return cls._exit_profit_target(
                df, trigger_value=trigger_value, 
                trade_number=trade_number, 
                end_of_day_position=end_of_day_position)
        
        # Calculate exit signal based on a loss from entry price
        elif exit_level == 'initial':
            return cls._exit_initial_dollar_loss(
                df, trigger_value=trigger_value,
                trade_number=trade_number, 
                end_of_day_position=end_of_day_position)
        
        # Calculate exit signal based on a breakeven level
        elif exit_level == 'breakeven':
            return cls._exit_breakeven(
                df, trigger_value=trigger_value,
                trade_number=trade_number, 
                end_of_day_position=end_of_day_position, 
                trade_high_price=trade_high_price, 
                trade_low_price=trade_low_price)
    
        # Calculate exit signal based on a trailing stop referencing the close
        elif exit_level == 'trail_close':
            return cls._exit_trailing(
                df, trigger_value=trigger_value, 
                trade_number=trade_number, 
                end_of_day_position=end_of_day_position)
    
        # Calculate exit signal based on a trailing stop referencing the 
        # high/low
        elif exit_level == 'trail_high_low':
            return cls._exit_trailing(
                df, trigger_value=trigger_value, 
                trade_number=trade_number, 
                end_of_day_position=end_of_day_position)
    
    
    @staticmethod
    def _exit_profit_target(
            df, trigger_value, trade_number, end_of_day_position):
        """
        Calculate exit based on a profit target.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trigger_value : Series
            The series to trigger exit.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.

        Returns
        -------
        df : DataFrame
            The OHLC data
        profit_target_exit : Series
            The exit signals.

        """
        
        # Create an empty array to store the signals
        profit_target_exit = np.array([0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
        
            # If there is a trade on
            if trade_number[row] != 0:
            
                # If there is a long position
                if end_of_day_position[row] > 0:
                
                    # If the close is greater than the trigger value
                    if df['Close'][row] > trigger_value[row]:
                    
                        # Set the exit signal to -1
                        profit_target_exit[row] = -1
                
                # If there is a short position
                elif end_of_day_position[row] < 0:
                
                    # If the close is less than the trigger value
                    if df['Close'][row] < trigger_value[row]:
                    
                        # Set the exit signal to 1
                        profit_target_exit[row] = 1
                
                else:
                    # Set the exit signal to 0  
                    profit_target_exit[row] = 0
    
        return df, profit_target_exit
    
    
    @staticmethod
    def _exit_initial_dollar_loss(
            df, trigger_value, trade_number, end_of_day_position):
        """
        Calculate exit based on a given loss from the entry point.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trigger_value : Series
            The series to trigger exit.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.

        Returns
        -------
        df : DataFrame
            The OHLC data.
        initial_dollar_loss_exit : Series
            The exit signals.

        """
        
        # Create an empty array to store the signals
        initial_dollar_loss_exit = np.array([0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:
                
                # If there is a long position
                if end_of_day_position[row] > 0:
                    
                    # If the close is less than the trigger value
                    if df['Close'][row] < trigger_value[row]:
                        
                        # Set the exit signal to -1
                        initial_dollar_loss_exit[row] = -1    
                
                # If there is a short position
                elif end_of_day_position[row] < 0:
                
                    # If the close is greater than the trigger value
                    if df['Close'][row] > trigger_value[row]:
                    
                        # Set the exit signal to 1
                        initial_dollar_loss_exit[row] = 1
                
                else:
                    # Set the exit signal to 0                    
                    initial_dollar_loss_exit[row] = 0    
    
        return df, initial_dollar_loss_exit
    
    
    @staticmethod
    def _exit_breakeven(
            df, trigger_value, trade_number, end_of_day_position, 
            trade_high_price, trade_low_price):
        """
        Calculate exit based on passing a breakeven threshold.      

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trigger_value : Series
            The series to trigger exit.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        trade_high_price : Series
            The high price of the trade.
        trade_low_price : Series
            The low price of the trade.

        Returns
        -------
        df : DataFrame
            The OHLC data.
        breakeven_exit : Series
            The exit signals.

        """
        
        # Create an empty array to store the signals
        breakeven_exit = np.array([0.0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:
                
                # If there is a long position
                if end_of_day_position[row] > 0:
                    
                    # If the high price of the trade is greater than the 
                    # trigger value
                    if trade_high_price[row] > trigger_value[row]:
                        
                        # If the close is less than the trigger value
                        if df['Close'][row] < trigger_value[row]:
                            
                            # Set the exit signal to -1
                            breakeven_exit[row] = -1
                            
                # If there is a short position            
                elif end_of_day_position[row] < 0:
                    
                    # If the low price of the trade is less than the 
                    # trigger value
                    if trade_low_price[row] < trigger_value[row]:
                        
                        # If the close is greater than the trigger value
                        if df['Close'][row] > trigger_value[row]:
                            
                            # Set the exit signal to 1
                            breakeven_exit[row] = 1
                else:
                    # Set the exit signal to 0
                    breakeven_exit[row] = 0
    
        return df, breakeven_exit
    
    
    @staticmethod
    def _exit_trailing(df, trigger_value, trade_number, end_of_day_position):
        """
        Calculate exit based on a trailing stop.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trigger_value : Series
            The series to trigger exit.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.

        Returns
        -------
        df : DataFrame
            The OHLC data.
        trailing_exit : Series
            The exit signals.

        """
        
        # Create an empty array to store the signals
        trailing_exit = np.array([0.0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:
                
                # If there is a long position
                if end_of_day_position[row] > 0:
                    
                    # If the close is less than the trigger value
                    if df['Close'][row] < trigger_value[row]:
                    
                        # Set the exit signal to -1
                        trailing_exit[row] = -1
                
                # If there is a short position
                elif end_of_day_position[row] < 0:
                
                    # If the close is greater than the trigger value
                    if df['Close'][row] > trigger_value[row]:
                
                        # Set the exit signal to 1
                        trailing_exit[row] = 1    
                                
                else:                    
                    # Set the exit signal to 0
                    trailing_exit[row] = 0    
    
        return df, trailing_exit
        
    
    @staticmethod
    def _exit_support_resistance(
            df, trade_number, end_of_day_position, time_period):
        """
        Calculate exit based on an n-day high / low.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the indicator.

        Returns
        -------
        df : DataFrame
            The OHLC data.
        support_resistance_exit : Series
            The exit signals.

        """
        # Calculate rolling min and max closing prices based on time period
        rolling_high_close = df['Close'].rolling(time_period).max()
        rolling_low_close = df['Close'].rolling(time_period).min()
        
        # Create empty arrays to store the signals
        support_resistance_exit = np.array([0.0]*len(df))
        exit_level = np.array([0.0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:
                
                # If there is a long position
                if end_of_day_position[row] > 0:
                    
                    # Set the exit level to the n-day low close
                    exit_level[row] = rolling_low_close[row]

                    # If the close is greater than the exit level
                    if df['Close'][row] < exit_level[row]:
                        
                        # Set the exit signal to -1
                        support_resistance_exit[row] = -1
    
                # If there is a short position
                elif end_of_day_position[row] < 0:
                    
                    # Set the exit level to the n-day high close
                    exit_level[row] = rolling_high_close[row]
                    
                    # If the close is greater than the exit level
                    if df['Close'][row] > exit_level[row]:
                    
                        # Set the exit signal to 1
                        support_resistance_exit[row] = 1
    
                else:
                    # Set the exit signal to 0
                    support_resistance_exit[row] = 0
        
        # Set the DataFrame columns to the numpy arrays
        df['rolling_high_close_sr_exit'] = rolling_high_close
        df['rolling_low_close_sr_exit'] = rolling_low_close
        df['exit_level'] = exit_level
                    
        return df, support_resistance_exit                 
    
    
    @staticmethod
    def _exit_immediate_profit(
            df, trade_number, end_of_day_position, time_period):
        """
        Calculate exit based on an immediate n-day profit.

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.    
        time_period : Int
            The length of time for the indicator.

        Returns
        -------
        df : DataFrame
            The OHLC data.
        immediate_profit_exit : Series
            The exit signals.
        """        
        
        # Create an empty array to store the signals
        immediate_profit_exit = np.array([0.0]*len(df))
        
        # For each row in the data
        for row in range(1, len(df)):
            
            # If there is a trade on
            if trade_number[row] != 0:
                
                # Find the row that relates to the trade entry
                trade_first_row = df.index.get_loc(
                    df[trade_number==trade_number[row]].index[0])
                trade_row_num = row - trade_first_row
                
                # After the given number of days
                if trade_row_num == time_period-1:
                    
                    # If there is a long position
                    if end_of_day_position[row] > 0:
                        
                        # If the trade is losing money 
                        if df['Close'][row] < df['Close'][row-time_period]:
                            
                            # Set the exit signal to -1
                            immediate_profit_exit[row] = -1
                    
                    # If there is a short position        
                    elif end_of_day_position[row] < 0:
                        
                        # If the trade is losing money
                        if df['Close'][row] > df['Close'][row-time_period]:
                            
                            # Set the exit signal to 1
                            immediate_profit_exit[row] = 1                
                    
                    else:
                        # Set the exit signal to 0
                        immediate_profit_exit[row] = 0
       
        return df, immediate_profit_exit


    @staticmethod
    def _exit_nday_range(
            df, trade_number, end_of_day_position, time_period):
        """
        Calculate exit based on an n-day range.    
    
        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the indicator.
    
        Returns
        -------
        df : DataFrame
            The OHLC data.
        nday_range_exit : Series
            The exit signals.
    
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
                    
                    # Set target 1 as the lower of the high low range of the 
                    # last n days and the longest bar of the last n days
                    target_1 = min(high_low_range[row], bar_range[row])
                    target_2 = max(high_low_range[row], bar_range[row])
                    
                    # If the position is long, add these to the close to set 
                    # the price targets
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
                        
                        # If the close is above the target 1 and this has not 
                        # yet been hit 
                        if (df['Close'][row] > price_target_1 
                            and pt_1_hit == False):
                            
                            # Set the exit signal to -1
                            nday_range_exit[row] = -1
                            
                            # Set the profit target 1 hit flag to True
                            pt_1_hit = True
                            
                        # If the close is above the target 2 and this has not 
                        # yet been hit    
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
                        
                        # If the close is below the target 1 and this has not 
                        # yet been hit 
                        if (df['Close'][row] < price_target_1 
                            and pt_1_hit == False):
                            
                            # Set the exit signal to 1
                            nday_range_exit[row] = 1
                            
                            # Set the profit target 1 hit flag to True
                            pt_1_hit = True
                            
                        # If the close is above the target 2 and this has not 
                        # yet been hit    
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

