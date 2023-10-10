"""
Exit signals using indicators.

"""

import random
import numpy as np
import pandas as pd
from technicalmethods.methods import Indicators

class IndicatorExit():
    """
    Calculate exit signals using indicators.

    """

    @staticmethod
    def exit_parabolic_sar(
        prices: pd.DataFrame,
        time_period: int,
        acceleration_factor: float,
        sip_price: bool) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on a Parabolic SAR.

        Parameters
        ----------
        prices : DataFrame
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
        prices : DataFrame
            The OHLC data
        parabolic_sar_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Extract high, low and close series from the DataFrame
        high = prices['High']
        low = prices['Low']

        # Calculate rolling min and max closing prices based on time period
        rolling_high = prices['High'].rolling(time_period).max()
        rolling_low = prices['Low'].rolling(time_period).min()

        # Initialize zero arrays to store data
        sar = np.array([0.0]*len(prices))
        parabolic_sar_exit = np.array([0]*len(prices))
        extreme_price = np.array([0.0]*len(prices))
        incremental_af = np.array([0.0]*len(prices))
        ep_sar_diff = np.array([0.0]*len(prices))
        af_x_diff = np.array([0.0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # Find the row that relates to the trade entry
                trade_first_row = prices.index.get_loc(
                    prices[trade_number==trade_number.iloc[row]].index[0])
                trade_row_num = row - trade_first_row

                # If it is the trade entry day
                if trade_row_num == 0:

                    # If it is the first trade in the data
                    if trade_number.iloc[row] == 1:

                        # If there is a long position
                        if end_of_day_position.iloc[row] > 0:

                            # Set the initial point to the n-day low
                            initial_point = rolling_low.iloc[row]

                            # Set the extreme price to the day's high
                            extreme_price[row] = high.iloc[row]

                        # If there is a short position
                        elif end_of_day_position.iloc[row] < 0:

                            # Set the initial point to the n-day high
                            initial_point = rolling_high.iloc[row]

                            # Set the extreme price to the day's low
                            extreme_price[row] = low.iloc[row]

                    # For every other trade
                    else:
                        # If there is a long position
                        if end_of_day_position.iloc[row] > 0:

                            # If the sip_price flag is True
                            if sip_price:

                                # Set the initial point to the n-day low close
                                initial_point = rolling_low.iloc[row]

                            # If the sip_price flag is False
                            else:
                                # Set the initial point to the previous trade
                                # low
                                initial_point = min(
                                    prices[trade_number==trade_number.iloc[
                                        row]-1]['Low'])

                            # Set the extreme price to the day's high
                            extreme_price[row] = high.iloc[row]

                        # If there is a short position
                        elif end_of_day_position.iloc[row] < 0:

                            # If the sip_price flag is True
                            if sip_price:

                                # Set the initial point to the n-day low close
                                initial_point = rolling_high.iloc[row]

                            # If the sip_price flag is False
                            else:
                                # Set the initial point to the previous trade
                                # high
                                initial_point = max(
                                    prices[trade_number==trade_number.iloc[
                                        row]-1]['High'])

                                # Set the extreme price to the day's high
                                extreme_price[row] = low.iloc[row]

                        # Set the sar to the initial point
                        sar[row] = initial_point

                        # Set the dynamic incremental_af to the initial
                        # acceleration factor
                        incremental_af[row] = acceleration_factor

                # If it is not the trade entry day
                else:
                    # If the previous day was long
                    if end_of_day_position.iloc[row-1] == 1:

                        # If the previous day's sar was greater than the
                        # previous day's
                        # low
                        if sar[row-1] > low.iloc[row-1]:

                            # Set the signal to exit
                            parabolic_sar_exit[row-1] = -1

                            # Set the new sar to the previous trades extreme
                            # price
                            sar[row] = extreme_price[row-1]

                        # If the previous day's sar plus the acceleration
                        # factor multiplied by the difference between the
                        # extreme price and the sar is greater than the lowest
                        # low of the previous 2 days
                        elif (sar[row-1] + af_x_diff[row-1]
                            > min(low.iloc[row-1], low.iloc[row-2])):

                            # Set the sar to the lowest low of the previous 2
                            # days
                            sar[row] = min(low.iloc[row-1], low.iloc[row-2])

                        # Otherwise
                        else:
                            # Set the sar to the previous day's sar plus the
                            # acceleration factor multiplied by the difference
                            # between the extreme price and the sar
                            sar[row] = sar[row-1] + af_x_diff[row-1]

                    # Otherwise if the previous day was short
                    elif end_of_day_position.iloc[row-1] == -1:
                        # If the previous day's sar was less than the previous
                        # day's high
                        if sar[row-1] < high.iloc[row-1]:

                            # Set the signal to exit
                            parabolic_sar_exit[row-1] = 1

                            # Set the new sar to the previous trades extreme
                            # price
                            sar[row] = extreme_price[row-1]

                        # If the previous day's sar less the acceleration
                        # factor multiplied by the difference between the
                        # extreme price and the sar is less than the highest
                        # high of the previous 2 days
                        elif (sar[row-1] - af_x_diff[row-1]
                            < max(high.iloc[row-1], high.iloc[row-2])):

                            # Set the sar to the highest high of the previous
                            # 2 days
                            sar[row] = max(high.iloc[row-1], high.iloc[row-2])

                        # Otherwise
                        else:
                            # Set the sar to the previous day's sar minus the
                            # acceleration factor multiplied by the difference
                            # between the extreme price and the sar
                            sar[row] = sar[row-1] - af_x_diff[row-1]

                # If the current trade direction is long
                if end_of_day_position.iloc[row] == 1:

                    # Set the extreme price to the greater of the previous
                    # day's extreme price and the current day's high
                    extreme_price[row] = max(
                        extreme_price[row-1], high.iloc[row])

                    # If the trade is making a new high
                    if extreme_price[row] > extreme_price[row-1]:

                        # Increment the acceleration factor by the input value
                        # to a max of 0.2
                        incremental_af[row] = min(
                            incremental_af[row-1] + acceleration_factor, 0.2)

                    # Otherwise
                    else:
                        # Set the acceleration factor to the same as the
                        # previous day
                        incremental_af[row] = incremental_af[row-1]

                # Otherwise if the current trade direction is short
                else:
                    # Set the extreme price to the lesser of the previous day's
                    # extreme price and the current day's low
                    extreme_price[row] = min(
                        extreme_price[row-1], low.iloc[row])

                    # If the trade is making a new low
                    if extreme_price[row] < extreme_price[row-1]:

                        # Increment the acceleration factor by the input value
                        # to a max of 0.2
                        incremental_af[row] = min(
                            incremental_af[row-1] + acceleration_factor, 0.2)

                    # Otherwise
                    else:
                        # Set the acceleration factor to the same as the
                        # previous day
                        incremental_af[row] = incremental_af[row-1]

            # Calculate the absolute value of the difference between the
            # extreme price and the sar
            ep_sar_diff[row] = abs(sar[row] - extreme_price[row])

            # Calculate the difference between the extreme price and the sar
            # multiplied by the acceleration factor
            af_x_diff[row] =  ep_sar_diff[row] * incremental_af[row]

        # Set the DataFrame columns to the numpy arrays
        prices['rolling_high_sar_exit'] = rolling_high
        prices['rolling_low_sar_exit'] = rolling_low
        prices['sar_exit'] = sar
        prices['ep_sar_exit'] = extreme_price
        prices['af_sar_exit'] = incremental_af
        prices['ep_sar_diff_exit'] = ep_sar_diff
        prices['af_x_diff_exit'] = af_x_diff

        return prices, parabolic_sar_exit


    @staticmethod
    def exit_rsi_trail(
        prices: pd.DataFrame,
        time_period: int,
        oversold: int,
        overbought: int) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on a trailing RSI - a down close when the RSI is
        overbought when long or an up close when the RSI is oversold when
        short.

        Parameters
        ----------
        prices : DataFrame
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
        prices : DataFrame
            The OHLC data.
        rsi_trail_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Calculate RSI
        rsi = Indicators.RSI(close=prices['Close'], time_period=time_period)

        # Create an empty array to store the signals
        rsi_trail_exit = np.array([0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # If there is a long position
                if end_of_day_position.iloc[row] > 0:

                    # If todays close is less than the previous days close
                    # and todays RSI is greater than the overbought level
                    if (prices['Close'].iloc[row] < prices['Close'].iloc[row-1]
                        and rsi[row] > overbought):

                        # Set the exit signal to -1
                        rsi_trail_exit[row] = -1

                # If there is a short position
                elif end_of_day_position.iloc[row] < 0:

                    # If todays close is greater than the previous days close
                    # and todays RSI is less than the oversold level
                    if (prices['Close'].iloc[row] > prices['Close'].iloc[row-1]
                        and rsi[row] < oversold):

                        # Set the exit signal to 1
                        rsi_trail_exit[row] = 1
                else:
                    # Set the exit signal to 0
                    rsi_trail_exit[row] = 0

        # Set the DataFrame column to the numpy array
        prices['RSI_exit'] = rsi

        return prices, rsi_trail_exit


    @staticmethod
    def exit_key_reversal(
        prices: pd.DataFrame,
        time_period: int) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on a key reversal day - a new high combined with
        a down close when long or a new low combined with an up close when
        short.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the rolling lookback calculation.

        Returns
        -------
        prices : DataFrame
            The OHLC data.
        key_reversal_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Calculate rolling high and low prices based on time period
        rolling_high = prices['High'].rolling(time_period).max()
        rolling_low = prices['Low'].rolling(time_period).min()

        # Create an empty array to store the signals
        key_reversal_exit = np.array([0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # If there is a long position
                if end_of_day_position.iloc[row] > 0:

                    # If the n-day high today is greater than yesterdays n-day
                    # high and todays close is less than the previous days
                    # close
                    if ((rolling_high.iloc[row] > rolling_high.iloc[row-1])
                        and (prices['Close'].iloc[row] <
                             prices['Close'].iloc[row-1])
                        ):

                        # Set the exit signal to -1
                        key_reversal_exit[row] = -1

                # If there is a short position
                elif end_of_day_position.iloc[row] < 0:

                    # If the n-day low today is less than yesterdays n-day
                    # low and todays close is greater than the previous days
                    # close
                    if ((rolling_low.iloc[row] < rolling_low.iloc[row-1])
                        and (prices['Close'].iloc[row] >
                             prices['Close'].iloc[row-1])
                        ):

                        # Set the exit signal to -1
                        key_reversal_exit[row] = 1

                else:
                    # Set the exit signal to 0
                    key_reversal_exit[row] = 0

        # Set the DataFrame columns to the numpy arrays
        prices['rolling_high_key'] = rolling_high
        prices['rolling_low_key'] = rolling_low

        return prices, key_reversal_exit


    @staticmethod
    def exit_volatility(
        prices: pd.DataFrame,
        time_period: int,
        threshold: float) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on an increase in volatility - a fall in price
        greater than the ATR * Threshold when long or a rise in price greater
        than the ATR * Threshold when when short.

        Parameters
        ----------
        prices : DataFrame
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
        prices : DataFrame
            The OHLC data
        volatility_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Create an empty array to store the signals
        volatility_exit = np.array([0]*len(prices))

        # Calculate ATR
        atr = Indicators.ATR(
            prices['High'], prices['Low'], prices['Close'], time_period)

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # If there is a long position
                if end_of_day_position.iloc[row] > 0:

                    # If the decrease in closing price from yesterday to today
                    # is greater than the ATR * Threshold
                    if ((prices['Close'].iloc[row] -
                         prices['Close'].iloc[row-1])
                        > (atr[row] * threshold)):

                        # Set the exit signal to -1
                        volatility_exit[row] = -1

                # If there is a short position
                elif end_of_day_position.iloc[row] < 0:

                    # If the increase in closing price from yesterday to today
                    # is greater than the ATR * Threshold
                    if ((prices['Close'].iloc[row-1] -
                         prices['Close'].iloc[row])
                        > (atr[row] * threshold)):

                        # Set the exit signal to 1
                        volatility_exit[row] = 1

                else:
                    # Set the exit signal to 0
                    volatility_exit[row] = 0

        # Set the DataFrame column to the numpy array
        prices['ATR_exit'] = atr

        return prices, volatility_exit


    @staticmethod
    def exit_stochastic_crossover(
        prices: pd.DataFrame,
        time_period: int) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on a stochastic crossover - if the slow k crosses
        below the slow d when long or if the slow k crosses above the slow d
        when short.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the fast k calculation.

        Returns
        -------
        prices : DataFrame
            The OHLC data
        stoch_cross_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Calculate slow k  and slow d
        slow_k, slow_d = Indicators.stochastic(
            prices['High'], prices['Low'], prices['Close'],
            fast_k_period=time_period, slow_k_period=3, slow_d_period=3,
            output_type='slow')

        # Create an empty array to store the signals
        stoch_cross_exit = np.array([0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # If there is a long position
                if end_of_day_position.iloc[row] > 0:

                    # If the slow k crosses below the slow d
                    if (slow_k[row] < slow_d[row]
                        and slow_k[row-1] > slow_d[row-1]):

                        # Set the exit signal to -1
                        stoch_cross_exit[row] = -1

                # If there is a short position
                elif end_of_day_position.iloc[row] < 0:

                    # If the slow k crosses above the slow d
                    if (slow_k[row] > slow_d[row]
                        and slow_k[row-1] < slow_d[row-1]):

                        # Set the exit signal to 1
                        stoch_cross_exit[row] = 1

                else:
                    # Set the exit signal to 0
                    stoch_cross_exit[row] = 0

        # Set the DataFrame columns to the numpy arrays
        prices['slow_k_exit'] = slow_k
        prices['slow_d_exit'] = slow_d

        return prices, stoch_cross_exit


    @staticmethod
    def exit_random(prices: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on the first losing day after a random time
        interval.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.

        Returns
        -------
        prices : DataFrame
            The OHLC data
        random_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Generate a random number of days between 5 and 20
        exit_days = random.randint(5,20)

        # Create an empty array to store the signals
        random_exit = np.array([0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # Find the row that relates to the trade entry
            trade_first_row = prices.index.get_loc(
                    prices[trade_number==trade_number.iloc[row]].index[0])
            trade_row_num = row - trade_first_row

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # If the trade has been on for the random number of days
                if trade_row_num > exit_days-1:

                    # If there is a long position
                    if end_of_day_position.iloc[row] > 0:

                        # If todays close is less than the previous days close
                        if (prices['Close'].iloc[row] <
                            prices['Close'].iloc[row-1]):

                            # Set the exit signal to -1
                            random_exit[row] = -1

                    # If there is a short position
                    elif end_of_day_position.iloc[row] < 0:

                        # If todays close is greater than the previous days
                        # close
                        if (prices['Close'].iloc[row] >
                            prices['Close'].iloc[row-1]):

                            # Set the exit signal to 1
                            random_exit[row] = 1

                    else:
                        # Set the exit signal to 0
                        random_exit[row] = 0

        # Set the DataFrame column to the numpy array
        prices['random_days_exit'] = exit_days

        return prices, random_exit


    @staticmethod
    def exit_support_resistance(
        prices: pd.DataFrame,
        time_period: int) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on an n-day high / low.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the indicator.

        Returns
        -------
        prices : DataFrame
            The OHLC data.
        support_resistance_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Calculate rolling min and max closing prices based on time period
        rolling_high_close = prices['Close'].rolling(time_period).max()
        rolling_low_close = prices['Close'].rolling(time_period).min()

        # Create empty arrays to store the signals
        support_resistance_exit = np.array([0.0]*len(prices))
        exit_level = np.array([0.0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # If there is a long position
                if end_of_day_position.iloc[row] > 0:

                    # Set the exit level to the n-day low close
                    exit_level[row] = rolling_low_close.iloc[row]

                    # If the close is greater than the exit level
                    if prices['Close'].iloc[row] < exit_level[row]:

                        # Set the exit signal to -1
                        support_resistance_exit[row] = -1

                # If there is a short position
                elif end_of_day_position.iloc[row] < 0:

                    # Set the exit level to the n-day high close
                    exit_level[row] = rolling_high_close.iloc[row]

                    # If the close is greater than the exit level
                    if prices['Close'].iloc[row] > exit_level[row]:

                        # Set the exit signal to 1
                        support_resistance_exit[row] = 1

                else:
                    # Set the exit signal to 0
                    support_resistance_exit[row] = 0

        # Set the DataFrame columns to the numpy arrays
        prices['rolling_high_close_sr_exit'] = rolling_high_close
        prices['rolling_low_close_sr_exit'] = rolling_low_close
        prices['exit_level'] = exit_level

        return prices, support_resistance_exit


    @staticmethod
    def exit_immediate_profit(
        prices: pd.DataFrame,
        time_period: int) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on an immediate n-day profit.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the indicator.

        Returns
        -------
        prices : DataFrame
            The OHLC data.
        immediate_profit_exit : Series
            The exit signals.
        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Create an empty array to store the signals
        immediate_profit_exit = np.array([0.0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # Find the row that relates to the trade entry
                trade_first_row = prices.index.get_loc(
                    prices[trade_number==trade_number.iloc[row]].index[0])
                trade_row_num = row - trade_first_row

                # After the given number of days
                if trade_row_num == time_period-1:

                    # If there is a long position
                    if end_of_day_position.iloc[row] > 0:

                        # If the trade is losing money
                        if (prices['Close'].iloc[row]
                            < prices['Close'].iloc[row-time_period]):

                            # Set the exit signal to -1
                            immediate_profit_exit[row] = -1

                    # If there is a short position
                    elif end_of_day_position.iloc[row] < 0:

                        # If the trade is losing money
                        if (prices['Close'].iloc[row]
                            > prices['Close'].iloc[row-time_period]):

                            # Set the exit signal to 1
                            immediate_profit_exit[row] = 1

                    else:
                        # Set the exit signal to 0
                        immediate_profit_exit[row] = 0

        return prices, immediate_profit_exit


    @staticmethod
    def exit_nday_range(
        prices: pd.DataFrame,
        time_period: int) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on an n-day range.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        time_period : Int
            The length of time for the indicator.

        Returns
        -------
        prices : DataFrame
            The OHLC data.
        nday_range_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # The highest high minus the lowest low of the last n days
        n_day_low = prices['Low'].rolling(time_period).min()
        n_day_high = prices['High'].rolling(time_period).max()
        high_low_range = n_day_high - n_day_low

        # The largest high low daily range of the last n bars
        bar_range = (prices['High'] - prices['Low']).rolling(time_period).max()

        # Create an empty array to store the signals
        nday_range_exit = np.array([0.0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # Find the row that relates to the trade entry
                trade_first_row = prices.index.get_loc(
                    prices[trade_number==trade_number.iloc[row]].index[0])
                trade_row_num = row - trade_first_row

                # If it is the trade entry date
                if trade_row_num == 0:

                    # Set flags for if the targets have been hit
                    pt_1_hit = False
                    pt_2_hit = False

                    # Set target 1 as the lower of the high low range of the
                    # last n days and the longest bar of the last n days
                    target_1 = min(
                        high_low_range.iloc[row], bar_range.iloc[row])
                    target_2 = max(
                        high_low_range.iloc[row], bar_range.iloc[row])

                    # If the position is long, add these to the close to set
                    # the price targets
                    if end_of_day_position.iloc[row] > 0:
                        price_target_1 = prices['Close'].iloc[row] + target_1
                        price_target_2 = prices['Close'].iloc[row] + target_2

                    # Otherwise subtract from the price
                    else:
                        price_target_1 = prices['Close'].iloc[row] - target_1
                        price_target_2 = prices['Close'].iloc[row] - target_2

                # For every other day in the trade
                else:

                    # If the position is long
                    if end_of_day_position.iloc[row] > 0:

                        # If the close is above the target 1 and this has not
                        # yet been hit
                        if (prices['Close'].iloc[row] > price_target_1
                            and pt_1_hit is False):

                            # Set the exit signal to -1
                            nday_range_exit[row] = -1

                            # Set the profit target 1 hit flag to True
                            pt_1_hit = True

                        # If the close is above the target 2 and this has not
                        # yet been hit
                        elif (prices['Close'].iloc[row] > price_target_2
                              and pt_2_hit is False):

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
                        if (prices['Close'].iloc[row] < price_target_1
                            and pt_1_hit is False):

                            # Set the exit signal to 1
                            nday_range_exit[row] = 1

                            # Set the profit target 1 hit flag to True
                            pt_1_hit = True

                        # If the close is above the target 2 and this has not
                        # yet been hit
                        elif (prices['Close'].iloc[row] < price_target_2
                              and pt_2_hit is False):

                            # Set the exit signal to 1
                            nday_range_exit[row] = 1

                            # Set the profit target 2 hit flag to True
                            pt_2_hit = True

                        # Otherwise
                        else:
                            # Set the exit signal to 0
                            nday_range_exit[row] = 0

        return prices, nday_range_exit
