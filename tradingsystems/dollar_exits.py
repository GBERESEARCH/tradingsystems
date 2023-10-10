"""
Exit signals based on a dollar value.

"""
import pandas as pd
import numpy as np

class DollarExit():
    """
    Calculate dollar value based exit signals.

    """

    @classmethod
    def exit_dollar(
        cls,
        prices: pd.DataFrame,
        trigger_value: pd.Series,
        exit_level: str) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on a dollar amount.

        Parameters
        ----------
        prices : DataFrame
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
        prices : DataFrame
            The OHLC data
        exit : Series
            Exit signals.

        """

        # Calculate exit signal based on a profit target
        if exit_level == 'profit_target':
            prices, exit_ = cls._exit_profit_target(
                prices=prices,
                trigger_value=trigger_value)

        # Calculate exit signal based on a loss from entry price
        elif exit_level == 'initial':
            prices, exit_ = cls._exit_initial_dollar_loss(
                prices=prices,
                trigger_value=trigger_value)

        # Calculate exit signal based on a breakeven level
        elif exit_level == 'breakeven':
            prices, exit_ = cls._exit_breakeven(
                prices=prices,
                trigger_value=trigger_value)

        # Calculate exit signal based on a trailing stop referencing the close
        elif exit_level == 'trail_close':
            prices, exit_ = cls._exit_trailing(
                prices=prices,
                trigger_value=trigger_value)

        # Calculate exit signal based on a trailing stop referencing the
        # high/low
        elif exit_level == 'trail_high_low':
            prices, exit_ = cls._exit_trailing(
                prices=prices,
                trigger_value=trigger_value)

        return prices, exit_


    @staticmethod
    def _exit_profit_target(
        prices: pd.DataFrame,
        trigger_value: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on a profit target.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trigger_value : Series
            The series to trigger exit.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.

        Returns
        -------
        prices : DataFrame
            The OHLC data
        profit_target_exit : Series
            The exit signals.

        """
        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Create an empty array to store the signals
        profit_target_exit = np.array([0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # If there is a long position
                if end_of_day_position.iloc[row] > 0:

                    # If the close is greater than the trigger value
                    if prices['Close'].iloc[row] > trigger_value.iloc[row]:

                        # Set the exit signal to -1
                        profit_target_exit[row] = -1

                # If there is a short position
                elif end_of_day_position.iloc[row] < 0:

                    # If the close is less than the trigger value
                    if prices['Close'].iloc[row] < trigger_value.iloc[row]:

                        # Set the exit signal to 1
                        profit_target_exit[row] = 1

                else:
                    # Set the exit signal to 0
                    profit_target_exit[row] = 0

        return prices, profit_target_exit


    @staticmethod
    def _exit_initial_dollar_loss(
        prices: pd.DataFrame,
        trigger_value: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on a given loss from the entry point.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trigger_value : Series
            The series to trigger exit.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.

        Returns
        -------
        prices : DataFrame
            The OHLC data.
        initial_dollar_loss_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Create an empty array to store the signals
        initial_dollar_loss_exit = np.array([0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # If there is a long position
                if end_of_day_position.iloc[row] > 0:

                    # If the close is less than the trigger value
                    if prices['Close'].iloc[row] < trigger_value.iloc[row]:

                        # Set the exit signal to -1
                        initial_dollar_loss_exit[row] = -1

                # If there is a short position
                elif end_of_day_position.iloc[row] < 0:

                    # If the close is greater than the trigger value
                    if prices['Close'].iloc[row] > trigger_value.iloc[row]:

                        # Set the exit signal to 1
                        initial_dollar_loss_exit[row] = 1

                else:
                    # Set the exit signal to 0
                    initial_dollar_loss_exit[row] = 0

        return prices, initial_dollar_loss_exit


    @staticmethod
    def _exit_breakeven(
        prices: pd.DataFrame,
        trigger_value: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on passing a breakeven threshold.

        Parameters
        ----------
        prices : DataFrame
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
        prices : DataFrame
            The OHLC data.
        breakeven_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']
        trade_high_price = prices['raw_trade_high_price']
        trade_low_price = prices['raw_trade_low_price']

        # Create an empty array to store the signals
        breakeven_exit = np.array([0.0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # If there is a long position
                if end_of_day_position.iloc[row] > 0:

                    # If the high price of the trade is greater than the
                    # trigger value
                    if trade_high_price.iloc[row] > trigger_value.iloc[row]:

                        # If the close is less than the trigger value
                        if prices['Close'].iloc[row] < trigger_value.iloc[row]:

                            # Set the exit signal to -1
                            breakeven_exit[row] = -1

                # If there is a short position
                elif end_of_day_position.iloc[row] < 0:

                    # If the low price of the trade is less than the
                    # trigger value
                    if trade_low_price.iloc[row] < trigger_value.iloc[row]:

                        # If the close is greater than the trigger value
                        if prices['Close'].iloc[row] > trigger_value.iloc[row]:

                            # Set the exit signal to 1
                            breakeven_exit[row] = 1
                else:
                    # Set the exit signal to 0
                    breakeven_exit[row] = 0

        return prices, breakeven_exit


    @staticmethod
    def _exit_trailing(
        prices: pd.DataFrame,
        trigger_value: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Calculate exit based on a trailing stop.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        trigger_value : Series
            The series to trigger exit.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.

        Returns
        -------
        prices : DataFrame
            The OHLC data.
        trailing_exit : Series
            The exit signals.

        """

        trade_number = prices['raw_trade_number']
        end_of_day_position = prices['raw_end_of_day_position']

        # Create an empty array to store the signals
        trailing_exit = np.array([0.0]*len(prices))

        # For each row in the data
        for row in range(1, len(prices)):

            # If there is a trade on
            if trade_number.iloc[row] != 0:

                # If there is a long position
                if end_of_day_position.iloc[row] > 0:

                    # If the close is less than the trigger value
                    if prices['Close'].iloc[row] < trigger_value.iloc[row]:

                        # Set the exit signal to -1
                        trailing_exit[row] = -1

                # If there is a short position
                elif end_of_day_position.iloc[row] < 0:

                    # If the close is greater than the trigger value
                    if prices['Close'].iloc[row] > trigger_value.iloc[row]:

                        # Set the exit signal to 1
                        trailing_exit[row] = 1

                else:
                    # Set the exit signal to 0
                    trailing_exit[row] = 0

        return prices, trailing_exit
