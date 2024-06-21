"""
Calculate position data

"""

import math
import numpy as np
from technicalmethods.methods import Indicators
import pandas as pd
from pandas.tseries.offsets import BDay
pd.options.mode.chained_assignment = None


class Positions():
    """
    Functions for calculating trade and position data.

    """
    @staticmethod
    def calc_positions(
        prices: pd.DataFrame,
        signal: pd.Series,
        start: int) -> dict:
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
            Whether Long / Short or Flat at the start of the day.
        trade_action : Series
            Whether to Buy / Sell at the open.
        end_of_day_position : Series
            Whether Long / Short or Flat at the end of the day.

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
            trade_action[row] = eod_trade_signal[row-1]

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
    def position_values(
        prices: pd.DataFrame,
        end_of_day_position: pd.Series,
        trade_price_dict: dict) -> dict:
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
                        prices['trade_number'].iat[row] == prices[
                            'trade_number'].iloc[row-1])):

                    # Set the initial position value to the same as the
                    # previous day
                    initial_position_value[row] = initial_position_value[row-1]

                    # Set the current position value to the opening price
                    # multiplied by the end of day position of the previous day
                    current_position_value[row] = (
                        prices['Open'].iat[row] *
                        end_of_day_position.iloc[row-1]
                        )

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
                        * end_of_day_position.iat[row])

                    # Set the current position value to the closing price
                    # multiplied by the end of day position
                    current_position_value[row] = (
                        prices['Close'].iat[row] *
                        end_of_day_position.iat[row]
                        )

                    # Set the maximum trade position value to the high price of
                    # the trade multiplied by the end of day position
                    max_trade_position_value[row] = (
                        trade_price_dict['trade_high_price'][row]
                        * end_of_day_position.iat[row])

                    # Set the maximum trade closing position value to the
                    # highest closing price of the trade multiplied by the end
                    # of day position
                    max_trade_close_position_value[row] = (
                        trade_price_dict['trade_close_high_price'][row]
                        * end_of_day_position.iat[row])

                    # Set the minimum trade position value to the low price of
                    # the trade multiplied by the end of day position
                    min_trade_position_value[row] = (
                        trade_price_dict['trade_low_price'][row]
                        * end_of_day_position.iat[row])

                    # Set the minimum trade closing position value to the
                    # lowest closing price of the trade multiplied by the end
                    # of day position
                    min_trade_close_position_value[row] = (
                        trade_price_dict['trade_close_low_price'][row]
                        * end_of_day_position.iat[row])

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


    @classmethod
    def position_size(
        cls,
        prices: pd.DataFrame,
        benchmark: pd.DataFrame,
        params: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Calculate trade position size

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        benchmark : DataFrame
            The closing prices of the benchmark series
        params : Dict
            equity : Float
                The account equity.
            start : Int
                Row from which trade signals can be generated.

        Returns
        -------
        position_size : Series
            The number of units of the chosen ticker to trade.
        benchmark_position_size : Series
            The number of units of the benchmark to trade.

        """
        # Initialize zero arrays to store position sizes
        prices['position_size'] = np.array(
            [0]*len(prices['Close']), dtype=int)
        prices['position_size_pp'] = np.array(
            [0]*len(prices['Close']), dtype=int)
        benchmark['benchmark_position_size'] = np.array(
            [0]*len(benchmark['Close']), dtype=int)

        # Get the date of the first trade
        trades = prices['raw_trade_number']
        params['first_trade_date'] = trades.loc[trades != 0].index[0]

        # Find the location of the start of the first trade
        params['first_trade_start'] = prices.index.get_loc(
            params['first_trade_date'])

        # Calculate position size based on initial account equity and first
        # trade closing prices
        if params['position_type'] == 'equity_constant':
            prices, params = cls._equity_constant_position_size(
                prices=prices, params=params)

        # Calculate position size based on initial account equity and closing
        # prices for each trade
        elif params['position_type'] == 'equity_variable':
            prices, params = cls._equity_variable_position_size(
                prices=prices, params=params)

        # Calculate position size based on proportion of ATR
        elif params['position_type'] == 'atr':
            prices, params = cls._atr_position_size(
                prices=prices, params=params)

        # Calculate position size as a fixed number of units
        else:
            prices, params = cls._fixed_position_size(
                prices=prices, params=params)

        # Set the initial position size to that of the first trade
        params['init_position_size'] = prices[
            'position_size'].loc[params['first_trade_date']]

        # Calculate benchmark position size
        benchmark, params = cls._benchmark_position_size(
            benchmark=benchmark, params=params)


        return prices, benchmark, params


    @staticmethod
    def _benchmark_position_size(
        benchmark: pd.DataFrame,
        params: dict) -> tuple[pd.DataFrame, dict]:

        # Set the position size to the number of shares that can be bought with
        # the initial equity

        # Set the number of units to use a percentage of starting equity at
        # the point when trade signals begin
        try:
            benchmark_units = math.ceil(
                (params['equity']
                 / benchmark['Close'].loc[params['first_trade_date']])
                * params['equity_inv_perc'])
            params['benchmark_start_date'] = params['first_trade_date']

        except KeyError:
            params['benchmark_start_date'] = params['first_trade_date'] - BDay(1)
            benchmark_units = math.ceil(
                (params['equity']
                 / benchmark['Close'].loc[params['benchmark_start_date']])
                * params['equity_inv_perc'])

        # Set the position size series to the number of units, starting from
        # the point that trade signals can be generated
        bench_start = benchmark.index.get_loc(params['benchmark_start_date'])
        # benchmark['benchmark_position_size'].iloc[
        #     bench_start:] = benchmark_units
        benchmark.loc[
            benchmark.index[bench_start], 
            'benchmark_position_size'
        ] = benchmark_units

        # Set the initial benchmark position size to that calculated as of the
        # same date
        params['init_benchmark_position_size'] = benchmark[
            'benchmark_position_size'].loc[params['benchmark_start_date']]

        return benchmark, params


    @staticmethod
    def _equity_constant_position_size(
        prices: pd.DataFrame,
        params: dict) -> tuple[pd.DataFrame, dict]:

        # Extract the raw trade signal from the OHLC data
        trade_number = prices['raw_trade_number']

        # Set the position size to the number of shares that can be
        # bought with the initial equity

        # Set the number of units to use a percentage of starting equity at
        # the point when trade signals begin
        units = math.ceil(
            (params['equity']
             / prices['Close'].loc[params['first_trade_date']])
            * params['equity_inv_perc']
            / params['contract_point_value'])

        # For each row since the first trade entry
        for row in range(params['first_trade_start'], len(prices['Close'])):

            # If the is a trade on
            if trade_number.iat[row] != 0:

                # Set the position size series to the number of units
                prices['position_size'][row] = units

                # Set the position size for the perfect profit calc to the
                # same as the position size
                prices['position_size_pp'][row] = prices['position_size'].iat[row]

            # If there is no trade on
            else:
                # Set the position size to zero
                prices['position_size'][row] = 0

                # Set the position size for the perfect profit calc to the
                # same as the previous day
                prices['position_size_pp'][row] = prices[
                    'position_size_pp'].iloc[row-1]

        return prices, params


    @staticmethod
    def _equity_variable_position_size(
        prices: pd.DataFrame,
        params: dict) -> tuple[pd.DataFrame, dict]:

        # Extract the raw trade signal from the OHLC data
        trade_number = prices['raw_trade_number']

        # For each row since the first trade entry
        for row in range(params['first_trade_start'], len(prices['Close'])):

            # If the is a trade on
            if trade_number.iat[row] != 0:

                # Get the index location of the trade entry date
                trade_first_row = prices.index.get_loc(
                    prices[trade_number==trade_number.iat[row]].index[0])

                # If it is the trade entry date
                if row == trade_first_row:

                    # Set the number of units to use a percentage of starting
                    # equity at the point when trade signals begin
                    prices['position_size'][row] = math.ceil(
                        (params['equity'] / prices['Close'].iat[row])
                        * params['equity_inv_perc']
                        / params['contract_point_value'])

                # For every other day in the trade take the entry size
                else:
                    prices['position_size'][row] = prices[
                        'position_size'].iloc[row-1]

                # Set the position size for the perfect profit calc to the
                # same as the position size
                prices['position_size_pp'][row] = prices['position_size'].iat[row]

            # If there is no trade on, set the position size to zero.
            else:
                prices['position_size'][row] = 0

                # Set the position size for the perfect profit calc to the
                # same as the previous day
                prices['position_size_pp'][row] = prices[
                    'position_size_pp'].iloc[row-1]

        return prices, params


    @staticmethod
    def _atr_position_size(
        prices: pd.DataFrame,
        params: dict) -> tuple[pd.DataFrame, dict]:

        # Calculate ATR levels
        prices['position_ATR'] = Indicators.ATR(
            prices['High'], prices['Low'], prices['Close'],
            params['atr_pos_size'])

        # Replace nan values with difference between high and low prices
        prices['position_ATR'] = np.where(
            np.isnan(prices['position_ATR']),
            prices['High'] - prices['Low'],
            prices['position_ATR'])

        #if np.isnan(prices['position_ATR'][row]):
        #    prices['position_ATR'][row] = (
        #        prices['High'][row] - prices['Low'][row])

        # Extract the raw trade signal from the OHLC data
        trade_number = prices['raw_trade_number']

        max_contracts = np.array(
            [0] * len(prices['Close']), dtype=int)

        # For each row since the first trade entry
        for row in range(params['first_trade_start'], len(prices['Close'])):

            # Set a limit to the amount of margin used
            if ((params['ticker_source'] == 'norgate')
                and (params['ticker'][0] == '&')):
                max_contracts[row] = math.ceil(
                    (params['equity'] / params['per_contract_margin']) * 0.15)
            else:
                max_contracts[row] = math.ceil(
                    (params['equity'] * params['margin_%'])
                    / prices['Close'].iat[row])

            # If the is a trade on
            if trade_number.iat[row] != 0:

                # Get the index location of the trade entry date
                trade_first_row = prices.index.get_loc(
                    prices[trade_number==trade_number.iat[row]].index[0])

                # If it is the trade entry date
                if row == trade_first_row:

                    # If we can calculate the ATR
                    #if row > params['atr_pos_size']:

                    # Size the position for each trade based on a fraction
                    # of the ATR
                    prices['position_size'].iat[row] = min(math.ceil(
                        (params['equity'] * (params['position_risk_bps']
                                             / 10000))
                        / (prices['position_ATR'].iat[row]
                           * params['contract_point_value'])),
                        max_contracts[row])

                    # Otherwise
                    #else:
                        # Set the position size to 1
                    #    prices['position_size'][row] = 1

                # For every other day in the trade take the entry size
                else:
                    prices['position_size'].iat[row] = prices[
                        'position_size'].iloc[row-1]

                # Set the position size for the perfect profit calc to the
                # same as the position size
                prices['position_size_pp'].iat[row] = prices[
                    'position_size'].iat[row]

            # If there is no trade on
            else:
                # Set the position size to zero.
                prices['position_size'].iat[row] = 0

                # Set the position size for the perfect profit calc to the
                # same as the previous day
                prices['position_size_pp'].iat[row] = prices[
                    'position_size_pp'].iloc[row-1]

        return prices, params


    @staticmethod
    def _fixed_position_size(
        prices: pd.DataFrame,
        params: dict) -> tuple[pd.DataFrame, dict]:

        # Extract the raw trade signal from the OHLC data
        trade_number = prices['raw_trade_number']

        units = params['fixed_pos_size']

        # For each row since the first trade entry
        for row in range(params['first_trade_start'], len(prices['Close'])):

            # If the is a trade on
            if trade_number.iat[row] != 0:
                prices['position_size'][row] = units

            # If there is no trade on
            else:
                prices['position_size'][row] = 0

            # Set position size for perfect profit calculation
            prices['position_size_pp'][row] = units

        return prices, params
