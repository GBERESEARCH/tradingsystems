"""
Moving Average trading signals

"""

import numpy as np
import pandas as pd
from technicalmethods.methods import Indicators

class MovingAverageEntry():
    """
    Functions to create moving average crossover trading signals

    """
    @staticmethod
    def entry_double_ma_crossover(
        prices: pd.DataFrame,
        params: dict) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signal for Double Moving Average Crossover strategy

        Parameters
        ----------
        prices : DataFrame
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
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals.

        """
        ma1 = params['ma1']
        ma2 = params['ma2']
        simple_ma = params['simple_ma']

        if simple_ma:

            # Create short and long simple moving averages
            ma_1 = prices['Close'].rolling(ma1).mean()
            ma_2 = prices['Close'].rolling(ma2).mean()

        else:
            # Create short and long exponential moving averages
            ma_1 = Indicators.EMA(
                input_series=prices['Close'], time_period=ma1)
            ma_2 = Indicators.EMA(
                input_series=prices['Close'], time_period=ma2)

        ma_1 = np.array(ma_1)
        ma_2 = np.array(ma_2)

        # Create start point from first valid number
        start = np.where(~np.isnan(ma_2))[0][0]

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(ma_2))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(ma_2))

        # Create numpy arrays of zeros to store min and max ma values
        min_ma = np.array([0.0]*len(ma_2))
        max_ma = np.array([0.0]*len(ma_2))

        # for each row in the DataFrame after the longest MA has started
        for row in range(start, len(ma_2)):

            # Calculate the min and max ma values
            min_ma[row] = min(ma_1[row], ma_2[row], prices['Close'].iloc[row])
            max_ma[row] = max(ma_1[row], ma_2[row], prices['Close'].iloc[row])

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

        prices['ma_1'] = ma_1
        prices['ma_2'] = ma_2
        prices['min_ma'] = min_ma
        prices['max_ma'] = max_ma

        return prices, start, trade_signal


    @staticmethod
    def entry_triple_ma_crossover(
        prices: pd.DataFrame,
        params: dict) -> tuple[pd.DataFrame, int, np.ndarray]:
        """
        Entry signal for Triple Moving Average Crossover strategy

        Parameters
        ----------
        prices : DataFrame
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
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals.

        """

        ma1 = params['ma1']
        ma2 = params['ma2']
        ma3 = params['ma3']
        simple_ma = params['simple_ma']

        # Create fast, medium and slow simple moving averages
        if simple_ma:
            ma_1 = prices['Close'].rolling(ma1).mean()
            ma_2 = prices['Close'].rolling(ma2).mean()
            ma_3 = prices['Close'].rolling(ma3).mean()

        else:
            ma_1 = Indicators.EMA(
                input_series=prices['Close'], time_period=ma1)
            ma_2 = Indicators.EMA(
                input_series=prices['Close'], time_period=ma2)
            ma_3 = Indicators.EMA(
                input_series=prices['Close'], time_period=ma3)

        ma_1 = np.array(ma_1)
        ma_2 = np.array(ma_2)
        ma_3 = np.array(ma_3)

        # Create start point from first valid number
        start = np.where(~np.isnan(ma_3))[0][0]

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(ma_3))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(ma_3))

        # Create numpy arrays of zeros to store min and max ma values
        min_ma = np.array([0.0]*len(ma_3))
        max_ma = np.array([0.0]*len(ma_3))

        # for each row in the DataFrame after the longest MA has started
        for row in range(start, len(ma_3)):

            # Calculate the min and max ma values
            min_ma[row] = min(
                ma_1[row], ma_2[row], prices['Close'].iloc[row])
            max_ma[row] = max(
                ma_1[row], ma_2[row], prices['Close'].iloc[row])

            # If the shortest ma is above the medium ma is above the long ma
            if ma_1[row] > ma_2[row] > ma_3[row]:

                # Set the position signal to long
                position_signal[row] = 1

                # If this was not the case previously
                if not ma_1[row-1] > ma_2[row-1] > ma_3[row-1]:

                    # Signal to go long
                    trade_signal[row] = 1 - position_signal[row-1]

            # If the shortest ma is below the medium ma is below the long ma
            elif ma_1[row] < ma_2[row] < ma_3[row]:

                # Set the position signal to short
                position_signal[row] = -1

                # If this was not the case previously
                if not ma_1[row-1] < ma_2[row-1] < ma_3[row-1]:

                    # Signal to go short
                    trade_signal[row] = -1 - position_signal[row-1]

            # Otherwise, the position should be flat
            else:
                position_signal[row] = 0
                trade_signal[row] = 0 - position_signal[row-1]

        # Assign the series to the OHLC data
        prices['ma_1'] = ma_1
        prices['ma_2'] = ma_2
        prices['ma_3'] = ma_3
        prices['min_ma'] = min_ma
        prices['max_ma'] = max_ma

        return prices, start, trade_signal


    @staticmethod
    def entry_quad_ma_crossover(
        prices: pd.DataFrame,
        params: dict) -> tuple[pd.DataFrame, int, np.ndarray]:

        """
        Entry signals for Quad Moving Average strategy

        Parameters
        ----------
        prices : DataFrame
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
        prices : DataFrame
            The OHLC data with additional columns.
        start : Int
            The first valid date row to calculate from.
        trade_signal : Series
            The series of Buy / Sell signals.
        """

        ma1 = params['ma1']
        ma2 = params['ma2']
        ma3 = params['ma3']
        ma4 = params['ma4']
        simple_ma = params['simple_ma']

        # Create the 4 simple moving averages
        if simple_ma:
            ma_1 = prices['Close'].rolling(ma1).mean()
            ma_2 = prices['Close'].rolling(ma2).mean()
            ma_3 = prices['Close'].rolling(ma3).mean()
            ma_4 = prices['Close'].rolling(ma4).mean()

        else:
            ma_1 = Indicators.EMA(
                input_series=prices['Close'], time_period=ma1)
            ma_2 = Indicators.EMA(
                input_series=prices['Close'], time_period=ma2)
            ma_3 = Indicators.EMA(
                input_series=prices['Close'], time_period=ma3)
            ma_4 = Indicators.EMA(
                input_series=prices['Close'], time_period=ma4)

        ma_1 = np.array(ma_1)
        ma_2 = np.array(ma_2)
        ma_3 = np.array(ma_3)
        ma_4 = np.array(ma_4)

        # Create numpy array of zeros to store position signals
        position_signal = np.array([0]*len(ma_4))

        # Create numpy array of zeros to store trade signals
        trade_signal = np.array([0]*len(ma_4))

        # Create start point from first valid number
        start = np.where(~np.isnan(ma_4))[0][0]

        # Create numpy arrays of zeros to store min and max ma values
        min_ma = np.array([0.0]*len(ma_4))
        max_ma = np.array([0.0]*len(ma_4))

        # for each row in the DataFrame after the longest MA has started
        for row in range(start + 1, len(ma_4)):

            if ma_1[row] > ma_2[row] > ma_3[row] > ma_4[row]:
                position_signal[row] = 1
            elif ma_1[row] < ma_2[row] < ma_3[row] < ma_4[row]:
                position_signal[row] = -1
            else:
                position_signal[row] = 0

            # Calculate the min and max ma values
            min_ma[row] = min(
                ma_1[row], ma_2[row], prices['Close'].iloc[row])
            max_ma[row] = max(
                ma_1[row], ma_2[row], prices['Close'].iloc[row])

            # If the position signal is to be long
            if position_signal[row] == 1:

                # If the previous day was not long
                if position_signal[row - 1] != 1:

                    # Signal to go long
                    trade_signal[row] = 1 - position_signal[row - 1]

                # Otherwise, take no action
                else:
                    trade_signal[row] = 0

            # If the position signal is to be short
            elif position_signal[row] == -1:

                # If the previous day was not short
                if position_signal[row - 1] != -1:

                    # Signal to go short
                    trade_signal[row] = -1 - position_signal[row-1]

                # Otherwise, take no action
                else:
                    trade_signal[row] = 0

            # If the position signal is to be flat
            else:
                # If the previous day was not flat
                if position_signal[row - 1] != 0:

                    # Signal to go flat
                    trade_signal[row] = -position_signal[row-1]

                else:
                    # Otherwise, take no action
                    trade_signal[row] = 0

        # Assign the series to the OHLC data
        prices['ma_1'] = ma_1
        prices['ma_2'] = ma_2
        prices['ma_3'] = ma_3
        prices['ma_4'] = ma_4
        prices['position_signal'] = position_signal
        prices['min_ma'] = min_ma
        prices['max_ma'] = max_ma

        return prices, start, trade_signal
