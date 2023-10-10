"""
Indicator Entry signals

"""

import pandas as pd
import numpy as np
from technicalmethods.methods import Indicators

class IndicatorEntry():
    """
    Calculate entry signals using indicators / oscillators.

    """

    @staticmethod
    def entry_parabolic_sar(
        prices: pd.DataFrame,
        acceleration_factor: float) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on Parabolic SAR

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        acceleration_factor : Float
            The acceleration factor to use.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """
        # Extract high, low and close series from the DataFrame
        high = prices['High']
        low = prices['Low']
        close = prices['Close']

        # Create empty arrays to store data
        sar = np.array([0.0]*len(prices))
        direction = np.array([0.0]*len(prices))
        trade_signal = np.array([0.0]*len(prices))
        extreme_price = np.array([0.0]*len(prices))
        incremental_af = np.array([0.0]*len(prices))
        ep_sar_diff = np.array([0.0]*len(prices))
        af_x_diff = np.array([0.0]*len(prices))

        # Configure initial values
        initial_sip = 0.975 * close.iloc[0]
        sar[0] = initial_sip
        direction[0] = 1
        extreme_price[0] = high.iloc[0]
        incremental_af[0] = 0.02
        ep_sar_diff[0] = abs(sar[0] - extreme_price[0])
        af_x_diff[0] =  ep_sar_diff[0] * incremental_af[0]
        start = 1
        init_flag = True

        # Loop through, starting from the second row
        for row in range(start, len(prices)):

            # If the previous day was long
            if direction[row-1] == 1:

                # If the previous day's sar was greater than the previous day's
                # low
                if sar[row-1] > low.iloc[row-1]:

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
                    sar[row] = extreme_price[row-1]

                    # Switch the trade direction from long to short
                    direction[row] = -direction[row-1]

                # If the previous day's sar plus the acceleration factor
                # multiplied by the difference between the extreme price and
                # the sar is greater than the lowest low of the previous 2 days
                elif (sar[row-1] + af_x_diff[row-1]
                      > min(low.iloc[row-1], low.iloc[row-2])):

                    # Set the sar to the lowest low of the previous 2 days
                    sar[row] = min(low.iloc[row-1], low.iloc[row-2])

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
                if sar[row-1] < high.iloc[row-1]:

                    # Close the short position and go long
                    trade_signal[row-1] = 2

                    # Set the new sar to the previous trades extreme price
                    sar[row] = extreme_price[row-1]

                    # Switch the trade direction from short to long
                    direction[row] = -direction[row-1]

                # If the previous day's sar less the acceleration factor
                # multiplied by the difference between the extreme price and
                # the sar is less than the highest high of the previous 2 days
                elif (sar[row-1] - af_x_diff[row-1]
                      < max(high.iloc[row-1], high.iloc[row-2])):

                    # Set the sar to the highest high of the previous 2 days
                    sar[row] = max(high.iloc[row-1], high.iloc[row-2])

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
                    extreme_price[row] = high.iloc[row]

                    # Set the initial acceleration factor to the input value
                    incremental_af[row] = acceleration_factor

                # If the trade is the same as the previous day
                else:

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

                # If the trade has just reversed direction
                if direction[row] != direction[row-1]:

                    # Set the extreme price to the day's low
                    extreme_price[row] = low.iloc[row]

                    # Set the initial acceleration factor to the input value
                    incremental_af[row] = acceleration_factor

                # If the trade is the same as the previous day
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

        # Assign the series to the OHLC data
        prices['sar_entry'] = sar
        prices['direction_sar_entry'] = direction
        prices['ep_sar_entry'] = extreme_price
        prices['af_sar_entry'] = incremental_af
        prices['ep_sar_diff_entry'] = ep_sar_diff
        prices['af_x_diff_entry'] = af_x_diff

        return prices, start, trade_signal


    @staticmethod
    def entry_channel_breakout(
        prices: pd.DataFrame,
        time_period: int) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on a channel breakout.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation. The default
            is 20.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """
        # Calculate rolling min and max closing prices based on time period
        rolling_high_close = prices['Close'].rolling(time_period).max()
        rolling_low_close = prices['Close'].rolling(time_period).min()

        # Create start point based on lookback window
        start = np.where(~np.isnan(rolling_high_close))[0][0]

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(rolling_high_close))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(rolling_high_close))

        # for each row in the DataFrame after the longest MA has started
        for row in range(start, len(rolling_high_close)):

            # If the price rises to equal or above the n-day high close
            if prices['Close'].iloc[row] >= rolling_high_close.iloc[row]:

                # Set the position signal to long
                position_signal[row] = 1

                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]

            # If the price falls to equal or below the n-day low close
            elif prices['Close'].iloc[row] <= rolling_low_close.iloc[row]:

                # Set the position signal to short
                position_signal[row] = -1

                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]

            # Otherwise, take no action
            else:
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]

        # Assign the series to the OHLC data
        prices['rolling_high_close_entry'] = rolling_high_close
        prices['rolling_low_close_entry'] = rolling_low_close

        return prices, start, trade_signal


    @staticmethod
    def entry_stochastic_crossover(
        prices: pd.DataFrame,
        time_period: int,
        oversold: int,
        overbought: int) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on slow k / slow d stochastics crossing.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        oversold : Int
            The oversold level to use.
        overbought : Int
            The overbought level to use.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """
        # Create Stochastics for the specified time period
        slow_k, slow_d = Indicators.stochastic(
            prices['High'], prices['Low'], prices['Close'],
            fast_k_period=time_period)

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
        prices['slow_k_entry'] = slow_k
        prices['slow_d_entry'] = slow_d

        return prices, start, trade_signal


    @staticmethod
    def entry_stochastic_over_under(
        prices: pd.DataFrame,
        time_period: int,
        oversold: int,
        overbought: int) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on slow k / slow d stochastics crossing
        overbought / oversold levels.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        oversold : Int
            The oversold level to use.
        overbought : Int
            The overbought level to use.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """
        # Create Stochastics for the specified time period
        slow_k, slow_d = Indicators.stochastic(
            prices['High'], prices['Low'], prices['Close'],
            fast_k_period=time_period)

        # Create start point based on slow d
        start = np.where(~np.isnan(slow_d))[0][0]

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(slow_d))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(slow_d))

        # for each row in the DataFrame after the slow d has started
        for row in range(start, len(slow_d)):

            # If both slow k and slow d cross above the lower level from below
            if (min(slow_k[row], slow_d[row])
                > oversold
                > min(slow_k[row-1], slow_d[row-1])):

                # Set the position signal to long
                position_signal[row] = 1

                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]

            # If both slow k and slow d cross above the upper level from below
            elif (min(slow_k[row], slow_d[row])
                  > overbought
                  > min(slow_k[row-1], slow_d[row-1])):

                # Set the position signal to flat
                position_signal[row] = 0

                # Signal to go flat
                trade_signal[row] = 0 - position_signal[row-1]

            # If both slow k and slow d cross below the upper level from above
            elif (max(slow_k[row], slow_d[row])
                  < overbought
                  < max(slow_k[row-1], slow_d[row-1])):

                # Set the position signal to short
                position_signal[row] = -1

                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]

            # If both slow k and slow d cross below the lower level from above
            elif (max(slow_k[row], slow_d[row])
                  < oversold
                  < max(slow_k[row-1], slow_d[row-1])):

                # Set the position signal to flat
                position_signal[row] = 0

                # Signal to go flat
                trade_signal[row] = 0 - position_signal[row-1]

            # Otherwise, take no action
            else:
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]

        # Assign the series to the OHLC data
        prices['slow_k_entry'] = slow_k
        prices['slow_d_entry'] = slow_d

        return prices, start, trade_signal


    @staticmethod
    def entry_stochastic_pop(
        prices: pd.DataFrame,
        time_period: int,
        oversold: int,
        overbought: int) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on the Stochastic Pop method

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        oversold : Int
            The oversold level to use.
        overbought : Int
            The overbought level to use.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """
        # Create Stochastics for the specified time period
        slow_k, slow_d = Indicators.stochastic(
            prices['High'], prices['Low'], prices['Close'],
            fast_k_period=time_period)

        # Create start point based on slow d
        start = np.where(~np.isnan(slow_d))[0][0]

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(slow_d))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(slow_d))

        # for each row in the DataFrame after the slow d has started
        for row in range(start, len(slow_d)):

            # If both slow k and slow d cross above the upper level from below
            if (min(slow_k[row], slow_d[row])
                > overbought
                > min(slow_k[row-1], slow_d[row-1])):

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
            elif (max(slow_k[row], slow_d[row])
                  < oversold
                  < max(slow_k[row-1], slow_d[row-1])):

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
        prices['slow_k_entry'] = slow_k
        prices['slow_d_entry'] = slow_d

        return prices, start, trade_signal


    @staticmethod
    def entry_rsi(
        prices: pd.DataFrame,
        time_period: int,
        oversold: int,
        overbought: int) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on the Relative Strength Index

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        oversold : Int
            The oversold level to use.
        overbought : Int
            The overbought level to use.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """
        # Create RSI for the specified time period
        rsi = Indicators.RSI(prices['Close'], time_period)

        # Create start point based on lookback window
        start = np.where(~np.isnan(rsi))[0][0]

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(rsi))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(rsi))

        # for each row in the DataFrame after the cci has started
        for row in range(start, len(rsi)):

            # If the rsi crosses above the threshold from below
            if rsi[row] < oversold < rsi[row-1]:

                # Set the position signal to long
                position_signal[row] = 1

                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]

            # If the rsi crosses below the threshold from above
            elif rsi[row] > overbought > rsi[row-1]:

                # Set the position signal to short
                position_signal[row] = -1

                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]

            # Otherwise, take no action
            else:
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]

        # Assign the series to the OHLC data
        prices['RSI_entry'] = rsi

        return prices, start, trade_signal


    @staticmethod
    def entry_adx(
        prices: pd.DataFrame,
        time_period: int,
        threshold: int) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on the ADX indicator.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        threshold : Int
            The level to determine if the market is trending.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """
        # Create ADX, di_plus, di_minus for the specified time period
        adx, di_plus, di_minus = Indicators.ADX(
            close=prices['Close'],
            high=prices['High'],
            low=prices['Low'],
            time_period=time_period,
            dmi=True)

        # Create start point based on lookback window
        start = np.where(~np.isnan(adx))[0][0]

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(adx))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(adx))


        # for each row in the DataFrame after the adx has started
        for row in range(start, len(adx)):

            # if the adx is above threshold
            if adx[row] > threshold:

                # if prices are trending up
                if di_plus[row] > di_minus[row]:

                    # Set the position signal to long
                    position_signal[row] = 1

                    # Signal to go long
                    trade_signal[row] = 1 - position_signal[row-1]

                # if prices are trending down
                else:

                    # Set the position signal to short
                    position_signal[row] = -1

                    # Signal to go short
                    trade_signal[row] = -1 - position_signal[row-1]

            # if the adx is below the threshold
            else:
                # If the current position is flat
                if position_signal[row-1] == 0:

                    # Take no action
                    trade_signal[row] = 0
                    position_signal[row] = position_signal[row-1]

                # If there is a trade on
                else:

                    # Reverse the current position
                    trade_signal[row] = -position_signal[row-1]
                    position_signal[row] = 0


        prices['ADX_entry'] = adx
        prices['DI_plus_entry'] = di_plus
        prices['DI_minus_entry'] = di_minus

        return prices, start, trade_signal


    @staticmethod
    def entry_macd(
        prices: pd.DataFrame,
        macd_params: tuple) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on the MACD indicator.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        macd_params : Tuple
            The MACD parameter values for MACD line, Signal line and Histogram.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """
        macd, macd_signal, macd_hist = Indicators.MACD(
            close=prices['Close'],
            fast=macd_params[0],
            slow=macd_params[1],
            signal=macd_params[2])

        # Create start point based on lookback window
        start = np.where(~np.isnan(macd_hist))[0][0]


        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(macd_hist))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(macd_hist))


        # for each row in the DataFrame after the MACD Hist has started
        for row in range(start, len(macd_hist)):

            # If the MACD Histogram is positive
            if macd_hist[row] > 0:

                # Set the position signal to long
                position_signal[row] = 1

                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]

            # If the MACD Histogram is negative
            else:

                # Set the position signal to short
                position_signal[row] = -1

                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]

        prices['MACD_entry'] = macd
        prices['MACD_Signal_entry'] = macd_signal
        prices['MACD_Hist_entry'] = macd_hist

        return prices, start, trade_signal


    @staticmethod
    def entry_commodity_channel_index(
        prices: pd.DataFrame,
        time_period: int,
        threshold: float) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on the Commodity Channel Index

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        threshold : Float
            The threshold used for taking signals. The default is 0.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """

        # Create CCI for the specified time period
        cci = Indicators.CCI(
            prices['High'], prices['Low'], prices['Close'], time_period)

        # Create start point based on lookback window
        start = np.where(~np.isnan(cci))[0][0]

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(cci))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(cci))

        # for each row in the DataFrame after the cci has started
        for row in range(start, len(cci)):

            # If the cci crosses above the threshold from below
            if cci[row] > threshold > cci[row-1]:

                # Set the position signal to long
                position_signal[row] = 1

                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]

            # If the cci crosses below the threshold from above
            elif cci[row] < threshold < cci[row-1]:

                # Set the position signal to short
                position_signal[row] = -1

                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]

            # Otherwise, take no action
            else:
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]

        # Assign the series to the OHLC data
        prices['CCI_entry'] = cci

        return prices, start, trade_signal


    @staticmethod
    def entry_momentum(
        prices: pd.DataFrame,
        time_period: int,
        threshold: float) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on n-day momentum

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation. The default
            is 10.
        threshold : Float
            The threshold used for taking signals. The default is 0.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """
        # Calculate past close based on time period
        n_day_close = prices['Close'].shift(time_period)

        # Create start point based on lookback window
        start = np.where(~np.isnan(n_day_close))[0][0]

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(n_day_close))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(n_day_close))

        # for each row in the DataFrame after the longest MA has started
        for row in range(start, len(n_day_close)):

            # If the price rises to equal or above the n-day high close
            if prices['Close'].iloc[row] >= n_day_close.iloc[row] + threshold:

                # Set the position signal to long
                position_signal[row] = 1

                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]

            # If the price falls to equal or below the n-day low close
            elif prices['Close'].iloc[row] <= n_day_close.iloc[row] - threshold:

                # Set the position signal to short
                position_signal[row] = -1

                # Signal to go short
                trade_signal[row] = -1 - position_signal[row-1]

            # Otherwise, take no action
            else:
                trade_signal[row] = 0
                position_signal[row] = position_signal[row-1]

        # Assign the series to the OHLC data
        prices['n_day_close'] = n_day_close
        prices['momentum'] = prices['Close'] - prices['n_day_close']

        return prices, start, trade_signal


    @staticmethod
    def entry_volatility(
        prices: pd.DataFrame,
        time_period: int,
        threshold: float) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signals based on a volatility breakout.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        time_period : Int
            The number of days to use in the indicator calculation.
        threshold : Float
            The threshold used for taking signals. The default is 1.5.

        Returns
        -------
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals

        """
        # Create ATR for the specified time period
        atr = Indicators.ATR(
            prices['High'], prices['Low'], prices['Close'], time_period)

        # Create start point based on lookback window
        start = np.where(~np.isnan(atr))[0][0]

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(atr))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(atr))

        # for each row in the DataFrame after the atr has started
        for row in range(start, len(atr)):

            # If the increase in closing price exceeds the atr * threshold
            if ((prices['Close'].iloc[row] - prices['Close'].iloc[row-1])
                > (atr[row] * threshold)):

                # Set the position signal to long
                position_signal[row] = 1

                # Signal to go long
                trade_signal[row] = 1 - position_signal[row-1]

            # If the decrease in closing price exceeds the atr * threshold
            elif ((prices['Close'].iloc[row-1] - prices['Close'].iloc[row])
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
        prices['ATR_entry'] = atr

        return prices, start, trade_signal
