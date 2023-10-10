"""
Calculate trades data

"""

import pandas as pd
import numpy as np

class Trades():
    """
    Calculate trade numbers, prices and combined trade signals

    """
    @staticmethod
    def trade_numbers(
        prices: pd.DataFrame,
        end_of_day_position: pd.Series,
        start: int) -> np.ndarray:
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
        eod_pos_np = np.array(end_of_day_position)

        # Create numpy array of zeros to store trade numbers
        trade_number = np.array([0]*len(prices))

        # Set initial trade count to zero
        trade_count = 0

        # For each valid row in the data
        for row in range(start + 1, len(prices)):

            # If today's position is zero
            if eod_pos_np[row] == 0:

                # If yesterday's position is zero
                if eod_pos_np[row - 1] == 0:

                    # There is no open trade so set trade number to zero
                    trade_number[row] = 0

                # If yesterday's position is not zero
                else:

                    # Set the trade number to the current trade count
                    trade_number[row] = trade_count

            # If today's position is the same as yesterday
            elif eod_pos_np[row] == eod_pos_np[row - 1]:

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
    def trade_prices(
        prices: pd.DataFrame,
        trade_number: pd.Series) -> dict:
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
            trade_num = trade_number.iloc[row]

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
    def signal_combine(
        prices: pd.DataFrame,
        start: int,
        end_of_day_position: pd.Series,
        trade_signals: pd.DataFrame) -> np.ndarray:
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
            trade_num = trade_number.iloc[row]
            trade_last_row = prices.index.get_loc(
                prices[trade_number==trade_num].index[-1])

            # If the raw trade signal is to change position and the current
            # end of day position is flat
            if ((raw_trade_signal.iloc[row] != 0) and (
                    end_of_day_position.iloc[row] == 0)):

                # Set the combined signal to the raw entry signal
                combined_signal[row] = raw_trade_signal.iloc[row]

            else:
                # If there is a trade on (based on the raw entry signal)
                if trade_number.iloc[row] != 0:

                    # If an exit has not yet been triggered
                    if flag:

                        # If there is a long position
                        if end_of_day_position.iloc[row] > 0:

                            # Set the trade signal to the minimum of the three
                            # series i.e. take any exit signal
                            combined_signal[row] = int(
                                min(trade_signals.iloc[row]))

                        # If there is a short position
                        elif end_of_day_position.iloc[row] < 0:

                            # Set the trade signal to the maximum of the three
                            # series i.e. take any exit signal
                            combined_signal[row] = int(
                                max(trade_signals.iloc[row]))

                        # If the position is flat
                        else:
                            # Set the trade signal to the raw entry signal
                            combined_signal[row] = raw_trade_signal.iloc[row]

                        # If there is an exit signalled by any of the three
                        # series
                        if combined_signal[row] != 0:

                            # Set the exit flag to False
                            flag = False

                    # If an exit has been triggered and the current trade
                    # number is not the same as the previous day
                    elif trade_number.iloc[row] != trade_number.iloc[row-1]:

                        # Set the trade signal to the raw entry signal
                        combined_signal[row] = raw_trade_signal.iloc[row]

                        # Reset the exit flag
                        flag=True

                    # If an exit has been triggered and the current trade
                    # number is the same as the previous day and the raw entry
                    # signal is to change position
                    elif row == trade_last_row and abs(
                            raw_trade_signal.iloc[row]) > 1:

                        # Set the trade signal to 1/2 raw entry signal
                        combined_signal[row] = int(
                            raw_trade_signal.iloc[row] / 2)

                        # Reset the exit flag
                        flag=True

                    else:
                        # Set the trade signal to 0
                        combined_signal[row] = 0

        return combined_signal
