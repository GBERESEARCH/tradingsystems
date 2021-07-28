"""
Calculate trades and positions

"""

import math
import numpy as np

class Positions():
    """
    Functions for calculating trade and position data.

    """
    @staticmethod
    def positions_and_trade_actions(prices, signal, start, position_size):
        """
        Calculate start of day and end of day positions and any buy / sell
        trade actions

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        signal : Series
            The series of Buy / Sell trade signals.
        start : Int
            The first valid row to start calculating trade information from.
        position_size : Int
            The number of units of the chosen ticker to trade.

        Returns
        -------
        start_of_day_position : Series
            The number of units of position held at the start of day.
        trade_action : Series
            Number of units to Buy / Sell at the open.
        end_of_day_position : Series
            The number of units of position held at the end of day.

        """
        # Extract the trade signal from the OHLC Data
        eod_trade_signal = np.array(signal)

        # Create empty arrays to store data
        start_of_day_position = np.array([0]*len(prices))
        trade_action = np.array([0]*len(prices))
        end_of_day_position = np.array([0]*len(prices))

        # For each valid row in the data
        for row in range(start + 1, len(prices)):

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

        pos_dict = {}
        pos_dict['start_of_day_position'] = start_of_day_position
        pos_dict['trade_action'] = trade_action
        pos_dict['end_of_day_position'] = end_of_day_position

        return pos_dict


    @staticmethod
    def trade_numbers(prices, end_of_day_position, start):
        """
        Calculate the trade numbers

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        end_of_day_position : Series
            The number of units of position held at the end of day.
        start : Int
            The first valid row to start calculating trade information from.

        Returns
        -------
        trade_number : Series
            Array of trade numbers.

        """
        # Extract the end of day position from the OHLC Data
        end_of_day_position = np.array(end_of_day_position)

        # Create numpy array of zeros to store trade numbers
        trade_number = np.array([0]*len(prices))

        # Set initial trade count to zero
        trade_count = 0

        # For each valid row in the data
        for row in range(start + 1, len(prices)):

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
    def trade_prices(prices, trade_number):
        """
        Calculate per trade entry, exit, high and low prices.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trade_number : Series
            Array of trade numbers.

        Returns
        -------
        trade_entry_price : Series
            The entry price for each trade.
        trade_exit_price : Series
            The exit price for each trade.
        trade_high_price : Series
            The high price for each trade.
        trade_low_price : Series
            The low price for each trade.
        trade_close_high_price : Series
            The highest closing price for each trade.
        trade_close_low_price : Series
            The lowest closing price for each trade.

        """

        # Initialize price arrays with zeroes
        trade_entry_price = np.array([0.0]*len(prices))
        trade_exit_price = np.array([0.0]*len(prices))
        trade_high_price = np.array([0.0]*len(prices))
        trade_low_price = np.array([0.0]*len(prices))
        trade_close_high_price = np.array([0.0]*len(prices))
        trade_close_low_price = np.array([0.0]*len(prices))

        # For each row in the DataFrame
        for row in range(len(prices)):

            # Get the current trade number
            trade_num = trade_number[row]

            # Get the index location of the trade entry date
            trade_first_row = prices.index.get_loc(
                prices[trade_number==trade_num].index[0])

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
                trade_entry_price[row] = prices[
                    trade_number==trade_num].iloc[0]['Open']

                # Set the trade exit price to the opening price on the last
                # trade date
                trade_exit_price[row] = prices[
                    trade_number==trade_num].iloc[-1]['Open']

                # Calculate the maximum close price during the trade
                trade_close_high_price[row] = max(
                    prices[trade_number==trade_num].iloc[
                        0:trade_row_num+1]['Close'])

                # Calculate the minimum close price during the trade
                trade_close_low_price[row] = min(
                    prices[trade_number==trade_num].iloc[
                        0:trade_row_num+1]['Close'])

                # Calculate the maximum high price during the trade
                trade_high_price[row] = max(
                    prices[trade_number==trade_num].iloc[
                        0:trade_row_num+1]['High'])

                # Calculate the minimum low price during the trade
                trade_low_price[row] = min(
                    prices[trade_number==trade_num].iloc[
                        0:trade_row_num+1]['Low'])

        # Create dict to store arrays
        trade_price_dict = {}
        trade_price_dict['trade_entry_price'] = trade_entry_price
        trade_price_dict['trade_exit_price'] = trade_exit_price
        trade_price_dict['trade_high_price'] = trade_high_price
        trade_price_dict['trade_low_price'] = trade_low_price
        trade_price_dict['trade_close_high_price'] = trade_close_high_price
        trade_price_dict['trade_close_low_price'] = trade_close_low_price

        return trade_price_dict


    @staticmethod
    def position_values(
            prices, end_of_day_position, trade_price_dict):
        """
        Calculate position values

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trade_entry_price : Series
            The entry price for each trade.
        end_of_day_position : Series
            The number of units of position held at the end of day.
        trade_high_price : Series
            The high price for each trade.
        trade_low_price : Series
            The low price for each trade.
        trade_close_high_price : Series
            The highest closing price for each trade.
        trade_close_low_price : Series
            The lowest closing price for each trade.
        trade_number : Series
            Array of trade numbers.

        Returns
        -------
        initial_position_value : Series
            The MTM value of the position at trade entry.
        current_position_value : Series
            The MTM value of the position at the current date.
        max_trade_position_value : Series
            The maximum MTM value of the position up to the current date.
        max_trade_close_position_value : Series
            The maximum MTM closing value of the position up to the current
            date.
        min_trade_position_value : Series
            The minimum MTM value of the position up to the current date.
        min_trade_close_position_value : Series
            The minimum MTM closing value of the position up to the current
            date.

        """

        # Create a empty arrays to store the signals
        initial_position_value = np.array([0]*len(prices))
        current_position_value = np.array([0]*len(prices))
        max_trade_position_value = np.array([0]*len(prices))
        max_trade_close_position_value = np.array([0]*len(prices))
        min_trade_position_value = np.array([0]*len(prices))
        min_trade_close_position_value = np.array([0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if prices['trade_number'][row] != 0:

                # If the end of day position is flat and the trade number is
                # the same as the previous day - i.e. the trade has just been
                # closed out
                if ((end_of_day_position[row] == 0) and (
                        prices['trade_number'][row] == prices[
                            'trade_number'][row-1])):

                    # Set the initial position value to the same as the
                    # previous day
                    initial_position_value[row] = initial_position_value[row-1]

                    # Set the current position value to the opening price
                    # multiplied by the end of day position of the previous day
                    current_position_value[row] = (
                        prices['Open'][row] * end_of_day_position[row-1])

                    # Set the maximum trade position value to the same as the
                    # previous day
                    max_trade_position_value[
                        row] = max_trade_position_value[row-1]

                    # Set the maximum trade closing position value to the same
                    # as the previous day
                    max_trade_close_position_value[
                        row] = max_trade_close_position_value[row-1]

                    # Set the minimum trade position value to the same as the
                    # previous day
                    min_trade_position_value[
                        row] = min_trade_position_value[row-1]

                    # Set the minimum trade closing position value to the same
                    # as the previous day
                    min_trade_close_position_value[
                        row] = min_trade_close_position_value[row-1]

                else:
                    # Set the initial position value to the trade entry price
                    # multiplied by the end of day position
                    initial_position_value[row] = (
                        trade_price_dict['trade_entry_price'][row]
                        * end_of_day_position[row])

                    # Set the current position value to the closing price
                    # multiplied by the end of day position
                    current_position_value[row] = (
                        prices['Close'][row] * end_of_day_position[row])

                    # Set the maximum trade position value to the high price of
                    # the trade multiplied by the end of day position
                    max_trade_position_value[row] = (
                        trade_price_dict['trade_high_price'][row]
                        * end_of_day_position[row])

                    # Set the maximum trade closing position value to the
                    # highest closing price of the trade multiplied by the end
                    # of day position
                    max_trade_close_position_value[row] = (
                        trade_price_dict['trade_close_high_price'][row]
                        * end_of_day_position[row])

                    # Set the minimum trade position value to the low price of
                    # the trade multiplied by the end of day position
                    min_trade_position_value[row] = (
                        trade_price_dict['trade_low_price'][row]
                        * end_of_day_position[row])

                    # Set the minimum trade closing position value to the
                    # lowest closing price of the trade multiplied by the end
                    # of day position
                    min_trade_close_position_value[row] = (
                        trade_price_dict['trade_close_low_price'][row]
                        * end_of_day_position[row])

        # Collect results in dictionary
        pos_val_dict = {}

        pos_val_dict['initial_position_value'] = initial_position_value
        pos_val_dict['current_position_value'] = current_position_value
        pos_val_dict['max_trade_position_value'] = max_trade_position_value
        pos_val_dict[
            'max_trade_close_position_value'] = max_trade_close_position_value
        pos_val_dict['min_trade_position_value'] = min_trade_position_value
        pos_val_dict[
            'min_trade_close_position_value'] = min_trade_close_position_value

        return pos_val_dict


    @staticmethod
    def pnl_targets(
            prices, dollar_amount, position_size, end_of_day_position,
            trade_price_dict):
        """
        Create profit and loss stop and exit points

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        exit_amount : Float
            The dollar exit amount. The default is $1000.00.
        position_size : Int
            The number of units of the chosen ticker to trade.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The number of units of position held at the end of day.
        trade_entry_price : Series
            The entry price for each trade.
        trade_high_price : Series
            The high price for each trade.
        trade_low_price : Series
            The low price for each trade.
        trade_close_high_price : Series
            The highest closing price for each trade.
        trade_close_low_price : Series
            The lowest closing price for each trade.

        Returns
        -------
        profit_target : Series
            The exit levels for each trade based on a dollar loss from the
            entry level.
        initial_dollar_loss : Series
            The exit levels for each trade based on a profit target
        trailing_close : Series
            The exit levels for each trade based on the trailing close.
        trailing_high_low : Series
            The exit levels for each trade based on the trailing high / low.

        """

        # Calculate the trade target (distance that price has to change) by
        # dividing the dollar amount by the number of unitsmaking up the
        # position size
        trade_target = np.round(dollar_amount / position_size, 2)

        # Create empty arrays to store the values
        profit_target = np.array([0.0]*len(prices))
        initial_dollar_loss = np.array([0.0]*len(prices))
        trailing_close = np.array([0.0]*len(prices))
        trailing_high_low = np.array([0.0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if prices['raw_trade_number'][row] != 0:

                # If there is a long position
                if end_of_day_position[row] > 0:

                    # Set the profit target to the trade entry price plus the
                    # trade target
                    profit_target[row] = (
                        trade_price_dict['trade_entry_price'][row]
                        + trade_target)

                    # Set the initial dollar loss target to the trade entry
                    # price minus the trade target
                    initial_dollar_loss[row] = (
                        trade_price_dict['trade_entry_price'][row]
                        - trade_target)

                    # Set the trailing close target to the closing high price
                    # of the trade minus the trade target
                    trailing_close[row] = (
                        trade_price_dict['trade_close_high_price'][row]
                        - trade_target)

                    # Set the trailing high/low target to the high price
                    # of the trade minus the trade target
                    trailing_high_low[row] = (
                        trade_price_dict['trade_high_price'][row]
                        - trade_target)

                # If there is a short position
                else:
                    # Set the profit target to the trade entry price minus the
                    # trade target
                    profit_target[row] = (
                        trade_price_dict['trade_entry_price'][row]
                        - trade_target)

                    # Set the initial dollar loss target to the trade entry
                    # price plus the trade target
                    initial_dollar_loss[row] = (
                        trade_price_dict['trade_entry_price'][row]
                        + trade_target)

                    # Set the trailing close target to the closing low price
                    # of the trade plus the trade target
                    trailing_close[row] = (
                        trade_price_dict['trade_close_low_price'][row]
                        + trade_target)

                    # Set the trailing high/low target to the low price
                    # of the trade minus the trade target
                    trailing_high_low[row] = (
                        trade_price_dict['trade_low_price'][row]
                        + trade_target)

        return profit_target, initial_dollar_loss, trailing_close, \
            trailing_high_low


    @staticmethod
    def signal_combine(
            prices, start, end_of_day_position, trade_signals):
        """
        Combine Entry, Exit and Stop signals into a single composite signal.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        start : Int
            The first valid row to start calculating trade information from.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        trade_signals : DataFrame
            DataFrame of Entry, exit and stop signals:
                raw_trade_signal : Series
                    The series of raw Buy / Sell entry signals.
                exit_signal : Series
                    The series of Buy / Sell exit signals.
                stop_signal : Series
                    The series of Buy / Sell stop signals.

        Returns
        -------
        combined_signal : Series
            The series of combined Buy / Sell signals.

        """

        # Create an empty array to store the signals
        combined_signal = np.array([0.0]*len(prices))

        raw_trade_signal = trade_signals['raw_trade_signal']
        trade_number = prices['raw_trade_number']

        # Set a flag for whether the exit has not been triggered for each trade
        flag = True

        # For each row in the data
        for row in range(start, len(prices)):

            # Find the row that relates to the trade exit
            trade_num = trade_number[row]
            trade_last_row = prices.index.get_loc(
                prices[trade_number==trade_num].index[-1])

            # If the raw trade signal is to change position and the current
            # end of day position is flat
            if ((raw_trade_signal[row] != 0) and (
                    end_of_day_position[row] == 0)):

                # Set the combined signal to the raw entry signal
                combined_signal[row] = raw_trade_signal[row]

            else:
                # If there is a trade on (based on the raw entry signal)
                if trade_number[row] != 0:

                    # If an exit has not yet been triggered
                    if flag:

                        # If there is a long position
                        if end_of_day_position[row] > 0:

                            # Set the trade signal to the minimum of the three
                            # series i.e. take any exit signal
                            combined_signal[row] = int(
                                min(trade_signals.iloc[row]))

                        # If there is a short position
                        elif end_of_day_position[row] < 0:

                            # Set the trade signal to the maximum of the three
                            # series i.e. take any exit signal
                            combined_signal[row] = int(
                                max(trade_signals.iloc[row]))

                        # If the position is flat
                        else:
                            # Set the trade signal to the raw entry signal
                            combined_signal[row] = raw_trade_signal[row]

                        # If there is an exit signalled by any of the three
                        # series
                        if combined_signal[row] != 0:

                            # Set the exit flag to False
                            flag = False

                    # If an exit has been triggered and the current trade
                    # number is not the same as the previous day
                    elif trade_number[row] != trade_number[row-1]:

                        # Set the trade signal to the raw entry signal
                        combined_signal[row] = raw_trade_signal[row]

                        # Reset the exit flag
                        flag=True

                    # If an exit has been triggered and the current trade
                    # number is the same as the previous day and the raw entry
                    # signal is to change position
                    elif row == trade_last_row and abs(
                            raw_trade_signal[row]) > 1:

                        # Set the trade signal to 1/2 raw entry signal
                        combined_signal[row] = int(
                            raw_trade_signal[row] / 2)

                        # Reset the exit flag
                        flag=True

                    else:
                        # Set the trade signal to 0
                        combined_signal[row] = 0

        return combined_signal


    @staticmethod
    def position_size(prices, benchmark, equity):
        """
        Calculate trade position size

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        benchmark : Series
            The closing prices of the benchmark series
        equity : Float
            The account equity.

        Returns
        -------
        position_size : Int
            The number of units of the chosen ticker to trade.
        benchmark_units : Int
            The number of units of the benchmark to trade.

        """
        # Set the position size to the number of shares that can be bought with
        # the initial equity

        # Set the number of units to use 75% of starting equity
        units = (equity / prices['Close'].iloc[0]) * 0.75
        benchmark_units = (equity / benchmark['Close'].iloc[0]) * 0.75

        # If the number of units would be greater than 50 then make the
        # position size a multiple of 10
        if prices['Close'].iloc[0] < equity / 50:
            position_size = math.floor(
                int(units / 10)) * 10

        # Otherwise take the most units that can be afforded
        else:
            position_size = math.floor(units)


        return position_size, benchmark_units
