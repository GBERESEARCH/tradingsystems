"""
Profit and Loss functions

"""
# pylint: disable=E1101
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

class Profit():
    """
    Functions for calculating trade profits, mark to market equity, trade
    runs etc.

    """

    @classmethod
    def profit_data(
        cls,
        prices: pd.DataFrame,
        params: dict) -> pd.DataFrame:
        """
        Adds profit fields to the OHLC data

        Parameters
        ----------
        prices : DataFrame
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
        prices : DataFrame
            Returns the input DataFrame with additional columns.

        """

        # Create pnl data
        prices = cls._pnl_mtm(
            prices=prices, params=params)

        # Create cumulative trade pnl, equity and drawdown data
        prices = cls._cumulative_trade_pnl_and_equity(
            prices=prices, equity=params['equity'])

        # Create perfect profit data
        prices = cls._perfect_profit(prices=prices, params=params)

        # Calculate margin utilisation
        prices = cls._margin_calc(prices=prices, params=params)

        return prices


    @classmethod
    def _pnl_mtm(
        cls,
        prices: pd.DataFrame,
        params: dict) -> pd.DataFrame:
        """
        Calculate pnl and mark to market columns

        Parameters
        ----------
        prices : DataFrame
            The OHLC data and trades signals.
        slippage : Float, optional
            The amount of slippage to apply to traded prices in basis points.
            The default is 5 bps per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The
            default is $0.00.

        Returns
        -------
        prices : DataFrame
            The input data with additional columns.

        """

        # daily pnl
        prices = cls._daily_pnl(prices=prices, params=params)

        # total pnl
        prices['total_pnl'] = np.array([0]*len(prices['Close']), dtype=float)
        prices['total_pnl'] = prices['daily_pnl'].cumsum()

        # position mtm
        prices['position_mtm'] = (prices['end_of_day_position']
                                  * prices['Close']
                                  * params['contract_point_value'])

        return prices


    @staticmethod
    def _daily_pnl(
        prices: pd.DataFrame,
        params: dict) -> pd.DataFrame:
        """
        Calculate daily PNL

        Parameters
        ----------
        prices : DataFrame
            The OHLC data and trades signals.
        slippage : Float, optional
            The amount of slippage to apply to traded prices in basis points.
            The default is 5 bps per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The
            default is $0.00.

        Returns
        -------
        prices : DataFrame
            The input data with additional columns.

        """

        # Create series of open, close and position
        open_ = prices['Open']
        close = prices['Close']
        pos = prices['end_of_day_position']

        # Create array of zeros
        day_pnl = np.array([0]*len(close), dtype=float)
        last_day_trade_pnl = np.array([0]*len(close), dtype=float)

        # For each row of closing prices
        for row in range(1, len(close)):

            # If the current position is flat
            if pos.iat[row] == 0:

                # If the previous days position was flat
                if pos.iat[row - 1] == 0:

                    # Set the pnl to zero
                    day_pnl[row] = 0

                # Otherwise:
                else:
                    # Set the pnl to the previous day's position multiplied by
                    # the difference between todays open and yesterdays close
                    # less the cost of slippage and commission
                    day_pnl[row] = (
                        ((pos.iat[row - 1] *
                          (open_.iat[row] - close.iat[row - 1])
                         - abs(pos.iat[row - 1]
                               * params['slippage']
                               * 0.0001
                               * open_.iat[row]))
                        - params['commission'])
                        * params['contract_point_value'])

            # If the current position is not flat
            else:
                # If the position is the same as the previous day
                if pos.iat[row] == pos.iat[row - 1]:

                    # Set the pnl to the current position * the difference
                    # between todays close and yesterdays close
                    day_pnl[row] = (pos.iat[row]
                                    * (close.iat[row] - close.iat[row - 1])
                                    * params['contract_point_value'])

                # If the position is reversed from the previous day
                elif pos.iat[row] == (-1) * pos.iat[row - 1]:
                    day_pnl[row] = (
                        ((pos.iat[row] * (close.iat[row] - open_.iat[row])
                         - abs(pos.iat[row]
                               * params['slippage']
                               * 0.0001
                               * open_.iat[row]))
                        - params['commission'])
                        * params['contract_point_value'])

                    last_day_trade_pnl[row] = (
                        ((pos.iat[row - 1] *
                          (open_.iat[row] - close.iat[row - 1])
                         - abs(
                             pos.iat[row - 1]
                             * params['slippage']
                             * 0.0001
                             * open_.iat[row]))
                        - params['commission'])
                        * params['contract_point_value'])

                # If the position was opened from flat
                else:
                    # Set the pnl to the current position * the difference
                    # between todays open and todays close less the cost of
                    # slippage and commission
                    day_pnl[row] = (
                        ((pos.iat[row] * (close.iat[row] - open_.iat[row])
                         - abs(pos.iat[row]
                               * params['slippage']
                               * 0.0001
                               * open_.iat[row]))
                        - params['commission'])
                        * params['contract_point_value'])

        # Create daily pnl column in DataFrame, rounding to 2dp
        prices['current_trade_pnl'] = np.round(day_pnl, 2)
        prices['last_day_trade_pnl'] = last_day_trade_pnl
        prices['daily_pnl'] = (prices['current_trade_pnl']
                               + prices['last_day_trade_pnl'])

        return prices


    @classmethod
    def _cumulative_trade_pnl_and_equity(
        cls,
        prices: pd.DataFrame,
        equity: float) -> pd.DataFrame:
        """
        Calculate cumulative per trade pnl and various account equity series

        Parameters
        ----------
        prices : DataFrame
            The OHLC data and trades signals.
        equity : Float
            The initial account equity level.

        Returns
        -------
        prices : DataFrame
            The input data with additional columns.

        """

        prices = cls._pnl_equity(prices=prices, equity=equity)

        prices = cls._min_max_equity(prices=prices, equity=equity)

        prices = cls._max_dd_gain(prices=prices)

        prices = cls._trade_drawback(prices=prices)

        return prices


    @staticmethod
    def _pnl_equity(
        prices: pd.DataFrame,
        equity: float) -> pd.DataFrame:
        """
        Calculate cumulative per trade pnl and open, closed and mtm equity
        series

        Parameters
        ----------
        prices : DataFrame
            The OHLC data and trades signals.
        equity : Float
            The initial account equity level.

        Returns
        -------
        prices : DataFrame
            The input data with additional columns.

        """
        # Take the trade number and daily pnl series from prices
        trade_number = prices['trade_number']
        current_trade_pnl = prices['current_trade_pnl']
        last_day_trade_pnl = prices['last_day_trade_pnl']
        daily_pnl = prices['daily_pnl']

        # Create arrays of zeros
        cumulative_trade_pnl = np.array([0.0]*len(daily_pnl))
        mtm_equity = np.array([0.0]*len(daily_pnl))
        closed_equity = np.array([0.0]*len(daily_pnl))
        open_equity = np.array([0.0]*len(daily_pnl))
        max_trade_pnl = np.array([0.0]*len(daily_pnl))

        # Set the initial equity values
        mtm_equity[0] = equity
        closed_equity[0] = equity

        # For each row of data
        for row in range(1, len(daily_pnl)):

            # The index location of the trade entry date
            trade_first_row = prices.index.get_loc(
                prices[trade_number==trade_number.iat[row]].index[0])

            # The number of days since trade entry
            trade_row_num = row - trade_first_row

            # The index location of the trade entry date
            trade_last_row = prices.index.get_loc(
                prices[trade_number==trade_number.iat[row]].index[-1])

            # Set the mtm equity to the previous days mtm equity plus
            # the days pnl
            mtm_equity[row] = mtm_equity[row-1] + daily_pnl.iat[row]

            # If there is a current trade
            if trade_number.iat[row] != 0:

                # If it is the trade entry date and there was no prior trade
                if (trade_row_num == 0
                    and trade_number.iat[row - 1] == 0):

                    # Set cumulative trade pnl to the days pnl
                    cumulative_trade_pnl[row] = daily_pnl.iat[row]

                    # The maximum of the initial days pnl and zero
                    max_trade_pnl[row] = max(daily_pnl.iat[row], 0)

                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = closed_equity[row-1]


                # If it is the trade entry date and this reverses the position
                # of a prior trade
                elif (trade_row_num == 0
                    and trade_number.iat[row - 1] != 0):

                    # Set cumulative trade pnl to the previous days cumulative
                    # pnl plus the last days pnl
                    cumulative_trade_pnl[row] = (cumulative_trade_pnl[row-1]
                                                 + last_day_trade_pnl.iat[row])

                    #  Set cumulative trade pnl to the previous days cumulative
                    # pnl plus the last days pnl
                    max_trade_pnl[row] = max(
                        cumulative_trade_pnl[row], max_trade_pnl[row-1])

                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = (mtm_equity[row-1]
                                          + last_day_trade_pnl.iat[row])


                # If it is the trade exit date and not a reversal
                elif (trade_last_row - row == 0
                      and trade_number.iat[row] == trade_number.iat[row - 1]):

                    # Set cumulative trade pnl to the previous days cumulative
                    # pnl plus the days pnl
                    cumulative_trade_pnl[row] = (cumulative_trade_pnl[row-1]
                                                 + daily_pnl.iat[row])

                    # The maximum of the current trade equity and the maximum
                    # trade equity of the previous day
                    max_trade_pnl[row] = max(
                        cumulative_trade_pnl[row], max_trade_pnl[row-1])

                    # Set the closed equity to the mtm equity
                    closed_equity[row] = mtm_equity[row]


                # If it is the second day of a reversal trade
                elif (trade_row_num == 1
                      and trade_number.iat[row - 1] != trade_number.iloc[row-2]):

                    # Set cumulative trade pnl to the previous days current
                    # pnl plus the days pnl
                    cumulative_trade_pnl[row] = (
                        current_trade_pnl.iat[row - 1] + daily_pnl.iat[row]
                        )

                    # The maximum of the first and second days pnl and zero
                    max_trade_pnl[row] = max(
                        cumulative_trade_pnl[row],
                        current_trade_pnl.iat[row - 1]
                        )

                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = closed_equity[row-1]


                # For every other day in the trade
                else:
                    # Set cumulative trade pnl to the previous days cumulative
                    # pnl plus the days pnl
                    cumulative_trade_pnl[row] = (
                        cumulative_trade_pnl[row-1] + daily_pnl.iat[row]
                        )

                    # The maximum of the current trade equity and the maximum
                    # trade equity of the previous day
                    max_trade_pnl[row] = max(
                        cumulative_trade_pnl[row],
                        max_trade_pnl[row-1]
                        )

                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = closed_equity[row-1]


            # If there is no current trade
            else:
                # Set cumulative trade pnl to zero
                cumulative_trade_pnl[row] = 0

                # Set the closed equity to the previous days closed equity
                closed_equity[row] = closed_equity[row-1]

            # Current open equity
            open_equity[row] = mtm_equity[row] - closed_equity[row]

        prices['cumulative_trade_pnl'] = cumulative_trade_pnl
        prices['max_trade_pnl'] = max_trade_pnl
        prices['mtm_equity'] = mtm_equity
        prices['closed_equity'] = closed_equity
        prices['open_equity'] = open_equity

        return prices


    @staticmethod
    def _min_max_equity(
        prices: pd.DataFrame,
        equity: float) -> pd.DataFrame:
        """
        Calculate min and max mtm equity, closed equity, max retracement and
        ulcer index input series

        Parameters
        ----------
        prices : DataFrame
            The OHLC data and trades signals.
        equity : Float
            The initial account equity level.

        Returns
        -------
        prices : DataFrame
            The input data with additional columns.

        """
        # Extract various series from prices
        daily_pnl = prices['daily_pnl']
        mtm_equity = prices['mtm_equity']
        closed_equity = prices['closed_equity']

        # Create arrays of zeros
        max_mtm_equity = np.array([0.0]*len(daily_pnl))
        min_mtm_equity = np.array([0.0]*len(daily_pnl))
        max_closed_equity = np.array([0.0]*len(daily_pnl))
        max_retracement = np.array([0.0]*len(daily_pnl))
        ulcer_index_d_sq = np.array([0.0]*len(daily_pnl))

        # Set the initial equity values
        max_mtm_equity[0] = equity
        min_mtm_equity[0] = equity
        max_closed_equity[0] = equity

        # For each row of data
        for row in range(1, len(daily_pnl)):

            # Maximum mtm equity to this point
            max_mtm_equity[row] = np.max(mtm_equity[0:row+1])

            # Minimum mtm equity to this point
            min_mtm_equity[row] = np.min(mtm_equity[0:row+1])

            # Maximum closed equity to this point
            max_closed_equity[row] = np.max(closed_equity[:row+1])

            # Maximum of max closed equity and current mtm equity, used in
            # calculating Average Max Retracement
            max_retracement[row] = max(
                (max_closed_equity[row] - mtm_equity.iat[row]), 0)

            # Squared difference between max mtm equity and current mtm equity,
            # used in calculating Ulcer Index
            ulcer_index_d_sq[row] = (
                (((max_mtm_equity[row] - mtm_equity.iat[row])
                 / max_mtm_equity[row]) * 100) ** 2)

        prices['max_closed_equity'] = max_closed_equity
        prices['max_retracement'] = max_retracement
        prices['max_mtm_equity'] = max_mtm_equity
        prices['min_mtm_equity'] = min_mtm_equity
        prices['ulcer_index_d_sq'] = ulcer_index_d_sq

        return prices


    @staticmethod
    def _max_dd_gain(
        prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate max drawdown and max equity gain

        Parameters
        ----------
        prices : DataFrame
            The OHLC data and trades signals.

        Returns
        -------
        prices : DataFrame
            The input data with additional columns.

        """
        # Extract various series from prices
        daily_pnl = prices['daily_pnl']
        mtm_equity = prices['mtm_equity']
        max_mtm_equity = prices['max_mtm_equity']
        min_mtm_equity = prices['min_mtm_equity']

        # Create max drawdown and max gain numpy arrays of zeros
        max_drawdown = np.array([0.0]*len(daily_pnl))
        max_drawdown_perc = np.array([0.0]*len(daily_pnl))
        max_gain = np.array([0.0]*len(daily_pnl))
        max_gain_perc = np.array([0.0]*len(daily_pnl))

        # For each row of data
        for row in range(1, len(daily_pnl)):

            # Maximum drawdown is the smallest value of the current cumulative
            # pnl less the max of all previous rows cumulative pnl and zero
            max_drawdown[row] = mtm_equity.iat[row] - max_mtm_equity.iat[row]

            # Percentage Maximum drawdown
            max_drawdown_perc[row] = (
                (mtm_equity.iat[row] - max_mtm_equity.iat[row]) /
                max_mtm_equity.iat[row]
                )

            # Maximum gain is the largest value of the current cumulative
            # pnl less the min of all previous rows and zero
            max_gain[row] = mtm_equity.iat[row] - min_mtm_equity.iat[row]

            # Percentage Maximum gain
            max_gain_perc[row] = (
                (mtm_equity.iat[row] - min_mtm_equity.iat[row]) /
                min_mtm_equity.iat[row]
                )

        prices['max_dd'] = max_drawdown
        prices['max_dd_perc'] = max_drawdown_perc
        prices['max_gain'] = max_gain
        prices['max_gain_perc'] = max_gain_perc

        return prices


    @staticmethod
    def _trade_drawback(
        prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate max trade pnl drawback

        Parameters
        ----------
        prices : DataFrame
            The OHLC data and trades signals.

        Returns
        -------
        prices : DataFrame
            The input data with additional columns.

        """
        # Extract various series from prices
        daily_pnl = prices['daily_pnl']
        cumulative_trade_pnl = prices['cumulative_trade_pnl']
        max_trade_pnl = prices['max_trade_pnl']

        # Create arrays of zeros
        trade_pnl_drawback = np.array([0.0]*len(daily_pnl))
        trade_pnl_drawback_perc = np.array([0.0]*len(daily_pnl))

        # For each row of data
        for row in range(1, len(daily_pnl)):

            # The difference between the highest equity peak of the trade and
            # the current trade open equity
            trade_pnl_drawback[row] = (max_trade_pnl.iat[row]
                                       - cumulative_trade_pnl.iat[row])

            # The percentage difference between the highest equity peak of the
            # trade and the current trade open equity
            if max_trade_pnl.iat[row] !=0:
                trade_pnl_drawback_perc[row] = (
                    (max_trade_pnl.iat[row] - cumulative_trade_pnl.iat[row])
                    / max_trade_pnl.iat[row])

        prices['trade_pnl_drawback'] = trade_pnl_drawback
        prices['trade_pnl_drawback_perc'] = trade_pnl_drawback_perc

        return prices


    @staticmethod
    def _perfect_profit(
        prices: pd.DataFrame,
        params: dict) -> pd.DataFrame:
        """
        Theoretical optimal of buying every low and selling every high

        Parameters
        ----------
        prices : DataFrame
            The OHLC data, trades signals and pnl.
        position_size : Int
            Number of units traded.

        Returns
        -------
        prices : DataFrame
            The input data with additional columns.

        """

        dpp = np.array([0.0]*len(prices))

        for row in range(params['first_trade_start'], len(dpp)):

            # Calculate Daily Perfect Profit
            dpp[row] = (
                abs(prices['High'].iat[row] - prices['Low'].iat[row])
                * prices['position_size_pp'].iat[row])

            # If the High and Low are the same
            if dpp[row] == 0:

                # Use the previous close
                dpp[row] = (
                    abs(prices['High'].iat[row] - prices['Close'].iat[row - 1])
                    * prices['position_size_pp'].iat[row])

        # Set this to the daily perfect profit
        prices['daily_perfect_profit'] = dpp * params['contract_point_value']

        # Create array of zeros
        prices['total_perfect_profit'] = np.array(
            [0.0]*len(prices['Close']), dtype=float)

        # Cumulative sum of daily perfect profit column
        prices['total_perfect_profit'] = prices[
            'daily_perfect_profit'].cumsum()

        return prices


    @staticmethod
    def _margin_calc(
        prices: pd.DataFrame,
        params: dict) -> pd.DataFrame:

        prices['total_margin'] = np.array([0.0]*len(prices))

        if params['ticker'][0] == '&':
            prices['initial_margin'] = (
                prices['position_size'] * params['per_contract_margin'])

        else:
            prices['initial_margin'] = (
                prices['Close'] * prices['position_size'] * params['margin_%'])

        for row in range(params['first_trade_start'], len(prices)):
            prices['total_margin'].iat[row] = (
                prices['initial_margin'].iat[row]
                + max(0, -prices['cumulative_trade_pnl'].iat[row]))

        return prices


    @staticmethod
    def time_to_recover(
        prices: pd.DataFrame) -> int | str:
        """
        Calculate the time taken for largest loss from peak to trough

        Parameters
        ----------
        prices : DataFrame
            The core OHLC DataFrame.

        Returns
        -------
        dd_length : Int
            The number of days from peak to trough.

        """
        # Extract the largest drawdown
        max_dd_val = min(prices['max_dd'])

        # Find the start location of the max drawdown
        dd_start = prices.index.get_loc(
            prices[prices['max_dd']==max_dd_val].index[0])

        # Calculate the point where equity has returned to the pre drawdown
        # level
        if max(prices['max_dd'][dd_start:]) == 0:
            dd_length = (prices['max_dd'][dd_start:].values == 0).argmax()

        # Otherwise return N/A
        else:
            dd_length = 'N/A'

        return dd_length


    @staticmethod
    def time_max_gain(prices: pd.DataFrame) -> int:
        """
        Calculate the time taken for largest gain from trough to peak

        Parameters
        ----------
        prices : DataFrame
            The core OHLC DataFrame.

        Returns
        -------
        gain_length : Int
            The number of days from trough to peak.

        """
        # Extract the largest pnl gain
        max_gain_val = max(prices['max_gain'])

        # Reverse the max gain column in time
        gain_rev = prices['max_gain'][::-1]

        # Find the location of the max gain
        max_gain_loc = gain_rev.index.get_loc(
            gain_rev[gain_rev==max_gain_val].index[0])

        # Calculate the time taken from zero to this max gain
        gain_length = (gain_rev[max_gain_loc:].values == 0).argmax()

        return gain_length


    @classmethod
    def create_monthly_data(
        cls,
        prices: pd.DataFrame,
        equity: int) -> pd.DataFrame:
        """
        Create monthly summary data

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        equity : Int
            The initial equity level.

        Returns
        -------
        monthly_data : DataFrame
            The monthly summary data.

        """
        # Set up monthly data DataFrame
        monthly_data = cls._initialise_monthly_data(
            prices=prices, equity=equity)

        # For each month
        for row in range(1, len(monthly_data)):

            # Beginning equity is the closing equity from the prior period
            # Raw data keeps all the profits invested whereas the other resets
            # the equity balance each year
            monthly_data['beginning_equity'].iat[row] = monthly_data[
                'end_equity'].iat[row - 1]
            monthly_data['beginning_equity_raw'].iat[row] = monthly_data[
                'end_equity_raw'].iat[row - 1]

            monthly_data.index = pd.to_datetime(monthly_data.index)
            # For each change in year
            if monthly_data.index.year[row] != monthly_data.index.year[row-1]:

                # If the end of year equity level is less than the initial
                # level
                if monthly_data['end_equity'].iat[row - 1] < equity:

                    # Add back the difference to additions
                    monthly_data['additions'].iat[row] = (
                        equity - monthly_data['end_equity'].iat[row - 1])
                else:
                    # Otherwise subtract from withdrawals
                    monthly_data['withdrawals'].iat[row] = (
                        equity - monthly_data['end_equity'].iat[row - 1])

            # Ending equity is the beginning equity plus the sum of additions,
            # withdrawals and the net pnl over the period
            monthly_data['end_equity'].iat[row] = (
                monthly_data['beginning_equity'].iat[row]
                + monthly_data['total_net_profit'].iat[row]
                + monthly_data['additions'].iat[row]
                + monthly_data['withdrawals'].iat[row])

            # Ending equity raw is the beginning equity plus the net pnl over
            # the period
            monthly_data['end_equity_raw'].iat[row] = (
                monthly_data['beginning_equity_raw'].iat[row]
                + monthly_data['total_net_profit'].iat[row])

            # Monthly return is the net pnl over the period divided by the
            # beginning equity plus the sum of additions and withdrawals
            monthly_data['return'].iat[row] = (
                (monthly_data['total_net_profit'].iat[row])
                / (monthly_data['beginning_equity'].iat[row]
                   + monthly_data['additions'].iat[row]
                   + monthly_data['withdrawals'].iat[row]))

            # Monthly return raw is the net pnl over the period divided by the
            # beginning equity raw
            monthly_data['return_raw'].iat[row] = (
                (monthly_data['total_net_profit'].iat[row])
                / (monthly_data['beginning_equity_raw'].iat[row]))

            # For use in Gain to Pain Ratio, absolute loss is the positive
            # value of the negative monthly returns
            if monthly_data['return'].iat[row] < 0:
                monthly_data['abs_loss'].iat[row] = -monthly_data['return'].iat[row]

            # For use in Gain to Pain Ratio, absolute loss is the positive
            # value of the negative monthly returns
            if monthly_data['return_raw'].iat[row] < 0:
                monthly_data['abs_loss_raw'].iat[row] = -monthly_data[
                    'return_raw'].iat[row]

        return monthly_data


    @staticmethod
    def _initialise_monthly_data(
        prices: pd.DataFrame,
        equity: int) -> pd.DataFrame:
        # Create empty DataFrame
        monthly_data = pd.DataFrame()

        # Summarize daily pnl data by resampling to monthly
        monthly_data['total_net_profit'] = prices[
            'daily_pnl'].resample('1ME').sum()
        monthly_data['average_net_profit'] = prices[
            'daily_pnl'].resample('1ME').mean()
        monthly_data['max_net_profit'] = prices[
            'daily_pnl'].resample('1ME').max()
        monthly_data['min_net_profit'] = prices[
            'daily_pnl'].resample('1ME').min()

        # Create arrays of zeros to hold data
        monthly_data['beginning_equity'] = np.array([0.0]*len(monthly_data))
        monthly_data['additions'] = np.array([0.0]*len(monthly_data))
        monthly_data['withdrawals'] = np.array([0.0]*len(monthly_data))
        monthly_data['end_equity'] = np.array([0.0]*len(monthly_data))
        monthly_data['return'] = np.array([0.0]*len(monthly_data))
        monthly_data['beginning_equity_raw'] = np.array(
            [0.0]*len(monthly_data))
        monthly_data['end_equity_raw'] = np.array([0.0]*len(monthly_data))
        monthly_data['return_raw'] = np.array([0.0]*len(monthly_data))
        monthly_data['abs_loss'] = np.array([0.0]*len(monthly_data))
        monthly_data['abs_loss_raw'] = np.array([0.0]*len(monthly_data))

        # Set initial values
        monthly_data['additions'].iat[0] = equity
        monthly_data['end_equity'].iat[0] = (
            monthly_data['beginning_equity'].iat[0]
            + monthly_data['additions'].iat[0]
            + monthly_data['withdrawals'].iat[0]
            + monthly_data['total_net_profit'].iat[0])
        monthly_data['return'].iat[0] = (
            (monthly_data['total_net_profit'].iat[0])
            / (monthly_data['beginning_equity'].iat[0]
               + monthly_data['additions'].iat[0]
               + monthly_data['withdrawals'].iat[0]))
        monthly_data['beginning_equity_raw'].iat[0] = equity
        monthly_data['end_equity_raw'].iat[0] = (
            monthly_data['beginning_equity_raw'].iat[0]
            + monthly_data['total_net_profit'].iat[0])
        monthly_data['return_raw'].iat[0] = (
            (monthly_data['total_net_profit'].iat[0])
            / (monthly_data['beginning_equity_raw'].iat[0]))

        return monthly_data
