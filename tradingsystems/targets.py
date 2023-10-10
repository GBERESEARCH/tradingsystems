"""
Trade exit triggers

"""

import pandas as pd
import numpy as np

class TradeTargets():
    """
    Calculate profit targets, stop loss levels etc.

    """
    @classmethod
    def exit_and_stop_targets(
        cls,
        prices: pd.DataFrame,
        params: dict,
        trade_price_dict: dict) -> pd.DataFrame:
        """
        Calculate exit and stop targets.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        exit_amount : Float
            The dollar exit amount. The default is $1000.00.
        stop_amount : Float
            The dollar stop amount. The default is $500.00.
        position_size : Int, optional
            The number of units to trade. The default is based on equity.
        trade_price_dict : Dict
            Dictionary of trade entry/high/low/close series.

        Returns
        -------
        prices : DataFrame
            The OHLC data.

        """

        if params['exit_type'] is not None:
            prices = cls._exit_targets(
                prices=prices,
                exit_amount=params['exit_amount'],
                trade_price_dict=trade_price_dict,
                params=params)

        if params['stop_type'] is not None:
            prices = cls._stop_targets(
                prices=prices,
                stop_amount=params['stop_amount'],
                trade_price_dict=trade_price_dict,
                params=params)

        return prices


    @classmethod
    def _exit_targets(
        cls,
        prices: pd.DataFrame,
        exit_amount: float,
        trade_price_dict: dict,
        params: dict) -> pd.DataFrame:
        """
        Create 4 series of exit targets

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        exit_amount : Float
            The dollar exit amount. The default is $1000.00.
        position_size : Int, optional
            The number of units to trade. The default is based on equity.
        trade_price_dict : Dict
            Dictionary of trade entry/high/low/close series.

        Returns
        -------
        prices : DataFrame
            The OHLC data..

        """
        # Generate profit targets / trailing stops
        prices['exit_profit_target'], prices['exit_initial_dollar_loss'], \
            prices['exit_trailing_close'], \
                prices['exit_trailing_high_low'] = cls._pnl_targets(
                    prices=prices, dollar_amount=exit_amount,
                    trade_price_dict=trade_price_dict,
                    params=params)

        return prices


    @classmethod
    def _stop_targets(
        cls,
        prices: pd.DataFrame,
        stop_amount: float,
        trade_price_dict: dict,
        params: dict) -> pd.DataFrame:
        """
        Create 4 series of stop targets

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        stop_amount : Float
            The dollar stop amount. The default is $500.00.
        position_size : Int, optional
            The number of units to trade. The default is based on equity.
        trade_price_dict : Dict
            Dictionary of trade entry/high/low/close series.

        Returns
        -------
        prices : DataFrame
            The OHLC data.

        """
        # Generate profit targets / trailing stops
        prices['stop_profit_target'], prices['stop_initial_dollar_loss'], \
            prices['stop_trailing_close'], \
                prices['stop_trailing_high_low'] = cls._pnl_targets(
                    prices=prices, dollar_amount=stop_amount,
                    trade_price_dict=trade_price_dict,
                    params=params)

        return prices


    @staticmethod
    def _pnl_targets(
        prices: pd.DataFrame,
        dollar_amount: float,
        trade_price_dict: dict,
        params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        # Create empty arrays to store the values
        trade_target = np.array([0.0]*len(prices))
        profit_target = np.array([0.0]*len(prices))
        initial_dollar_loss = np.array([0.0]*len(prices))
        trailing_close = np.array([0.0]*len(prices))
        trailing_high_low = np.array([0.0]*len(prices))

        end_of_day_position = prices['raw_end_of_day_position']
        position_size = prices['position_size']

        # For each row in the data
        for row in range(1, len(prices)):

            # Calculate the trade target (distance that price has to change) by
            # dividing the dollar amount by the number of units making up the
            # position size
            if position_size.iloc[row] == 0:
                trade_target[row] = 0
            else:
                trade_target[row] = np.round(
                    (dollar_amount / params['contract_point_value'])
                    / position_size.iloc[row], 2)

            # If there is a trade on
            if prices['raw_trade_number'].iloc[row] != 0:

                # If there is a long position
                if end_of_day_position.iloc[row] > 0:

                    # Set the profit target to the trade entry price plus the
                    # trade target
                    profit_target[row] = (
                        trade_price_dict['trade_entry_price'][row]
                        + trade_target[row])

                    # Set the initial dollar loss target to the trade entry
                    # price minus the trade target
                    initial_dollar_loss[row] = (
                        trade_price_dict['trade_entry_price'][row]
                        - trade_target[row])

                    # Set the trailing close target to the closing high price
                    # of the trade minus the trade target
                    trailing_close[row] = (
                        trade_price_dict['trade_close_high_price'][row]
                        - trade_target[row])

                    # Set the trailing high/low target to the high price
                    # of the trade minus the trade target
                    trailing_high_low[row] = (
                        trade_price_dict['trade_high_price'][row]
                        - trade_target[row])

                # If there is a short position
                else:
                    # Set the profit target to the trade entry price minus the
                    # trade target
                    profit_target[row] = (
                        trade_price_dict['trade_entry_price'][row]
                        - trade_target[row])

                    # Set the initial dollar loss target to the trade entry
                    # price plus the trade target
                    initial_dollar_loss[row] = (
                        trade_price_dict['trade_entry_price'][row]
                        + trade_target[row])

                    # Set the trailing close target to the closing low price
                    # of the trade plus the trade target
                    trailing_close[row] = (
                        trade_price_dict['trade_close_low_price'][row]
                        + trade_target[row])

                    # Set the trailing high/low target to the low price
                    # of the trade minus the trade target
                    trailing_high_low[row] = (
                        trade_price_dict['trade_low_price'][row]
                        + trade_target[row])

        return profit_target, initial_dollar_loss, trailing_close, \
            trailing_high_low
