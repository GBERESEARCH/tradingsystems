"""
Winning and Losing trades and runs

"""

import numpy as np
import pandas as pd

class Runs():
    """
    Calculate data on winning and losing trades and win/loss runs

    """

    @staticmethod
    def trade_data(prices: pd.DataFrame) -> dict:
        """
        Create dictionary of trades, count of the number of trades and lists /
        dictionaries of winning and losing trades

        Parameters
        ----------
        prices : DataFrame
            The OHLC data with trade signals.

        Returns
        -------
        trades : Dict
            Dictionary containing all the trades.
        num_trades : Int
            Number of trades.
        trades_win_dict : Dict
            Dictionary of winning trades.
        trades_win_list : TYPE
            List of winning trades.
        trades_loss_dict : TYPE
            Dictionary of losing trades.
        trades_loss_list : TYPE
            List of losing trades.

        """
        # profit per trade
        # Create empty trades dictionary
        trades = {}

        # Count the number of unique trade numbers (less 1 for the 0 start)
        num_trades = len(pd.unique(prices['trade_number'])) - 1

        # For each trade number
        for trade_number in range(1, num_trades+1):

            # Calculate profit as the sum of daily pnl for that trade number
            profit = prices[
                prices['trade_number']==trade_number]['daily_pnl'].sum()

            # Assign this number (rounded to 2dp) to the trades dictionary
            trades[trade_number] = np.round(profit, 2)

        # Split winning and losing trades
        # Create empty win/loss dictionaries and lists
        trades_win_dict = {}
        trades_win_list = []
        trades_loss_dict = {}
        trades_loss_list = []

        # for each trade and profit in the trades dictionary
        for key, value in trades.items():

            # If the profit is negative
            if value < 0:

                # Add the trade to the loss dictionary and list (as a tuple)
                trades_loss_dict[key] = value
                trades_loss_list.append((key, value))

            # If the profit is positive
            else:

                # Otherwise add to the win dictionary and list (as a tuple)
                trades_win_dict[key] = value
                trades_win_list.append((key, value))

        trade_data_dict = {
            'trades':trades,
            'num_trades':num_trades,
            'trades_win_dict':trades_win_dict,
            'trades_win_list':trades_win_list,
            'trades_loss_dict':trades_loss_dict,
            'trades_loss_list':trades_loss_list
            }

        return trade_data_dict


    @classmethod
    def trade_runs(
        cls,
        input_trades_list: list,
        run_type: str) -> dict:
        """
        Produce data for winning or losing runs of trades

        Parameters
        ----------
        input_trades_list : List
            List of winning or losing trades.
        run_type : Str, optional
            Whether winning or losing run data is being calculated. The
            default is 'win'.

        Returns
        -------
        max_run_pnl : Float
            Largest profit or loss.
        max_run_count : Int
            Number of trades in largest profit or loss.
        min_run_pnl : Float
            Smallest profit or loss.
        min_run_count : Int
            Number of trades in smallest profit or loss.
        num_runs : Int
            Count of number of winning or losing runs.
        av_run_count : Int
            Count of average number of trades in winning or losing runs.
        av_run_pnl : Float
            Average profit or loss.
        pnl : Tuple
            PNL for each run and number of trades.

        """

        pnl = cls._calc_trade_runs(input_trades_list)

        min_max_run_dict = cls._calc_min_max_runs(pnl, run_type)

        # Count number of runs as the length of the pnl list
        num_runs = len(pnl)

        if pnl:

            # Take the average number of runs as the sum of run lengths in pnl
            # tuple divided by the number of runs
            av_run_count = int(np.round(sum(j for i, j in pnl) / len(pnl), 0))

            # Take the average run pnl as the sum of run pnls in pnl tuple
            # divided by the number of runs
            av_run_pnl = np.round(sum(i for i, j in pnl) / len(pnl), 2)

        else:
            av_run_count = 0
            av_run_pnl = 0

        name_dict = cls._calc_run_names(run_type)

        run_dict = {
            name_dict['max_run_pnl_str']:min_max_run_dict['max_run_pnl'],
            name_dict['max_run_count_str']:min_max_run_dict['max_run_count'],
            name_dict['min_run_pnl_str']:min_max_run_dict['min_run_pnl'],
            name_dict['min_run_count_str']:min_max_run_dict['min_run_count'],
            name_dict['num_runs_str']:num_runs,
            name_dict['av_run_count_str']:av_run_count,
            name_dict['av_run_pnl_str']:av_run_pnl,
            name_dict['pnl_str']:pnl
            }

        return run_dict

    @staticmethod
    def _calc_trade_runs(input_trades_list: list) -> list[tuple[int, int]]:

        # Set initial values
        max_run_count = 1
        run_count = 1
        run_trades_list = []
        total_run_trades_list = []
        last_trade_count = 0

        # For each trade in the winning or losing trades list, sorting by the
        # trade number
        for num, trade in enumerate(sorted(input_trades_list)):
            # For the first trade
            if num == 0:

                # Add the trade pnl to the winning / losing trades run list
                run_trades_list.append(trade[1])

                # If this is the last trade
                if num == len(input_trades_list) - 1:
                    total_run_trades_list.append(run_trades_list)

            # Otherwise, if the trade number is next in sequence after the
            # last stored trade number
            elif trade[0] == last_trade_count + 1:

                # Increase the run count by one
                run_count +=1

                # Update the longest run count
                max_run_count = max(max_run_count, run_count)

                # Add the trade pnl to the winning / losing trades run list
                run_trades_list.append(trade[1])

                # If this is the last trade
                if num == len(input_trades_list) - 1:
                    total_run_trades_list.append(run_trades_list)

            # If the trade is not the next in sequence:
            else:

                # Add the current run to the list of all runs
                total_run_trades_list.append(run_trades_list)

                # If this is not the last trade
                if num != len(input_trades_list) - 1:

                    # Reset the winning / losing trades run list
                    run_trades_list = []

                    # Add the trade pnl to the winning / losing trades run list
                    run_trades_list.append(trade[1])

                    # Increase the run count by one
                    run_count = 1

                # If it is the last trade
                else:

                    # Reset the winning / losing trades run list
                    run_trades_list = []

                    # Add the trade pnl to the winning / losing trades run list
                    run_trades_list.append(trade[1])

                    # Add the current run to the list of all runs
                    total_run_trades_list.append(run_trades_list)

            # Set the last trade count number to the current trade number
            last_trade_count = trade[0]

        # Tuple for each run of PNL and number of trades.
        pnl = sorted([(sum(x), len(x)) for x in total_run_trades_list])

        return pnl


    @staticmethod
    def _calc_min_max_runs(
        pnl: list[tuple[int, int]],
        run_type: str) -> dict[str, int]:

        # Values to select for winning runs
        if run_type == 'win':

            # If there are any winning trades
            if pnl:
                max_run_pnl = pnl[-1][0]
                max_run_count = pnl[-1][-1]
                min_run_pnl = pnl[0][0]
                min_run_count = pnl[0][-1]

            # Otherwise set the values to zero
            else:
                max_run_pnl = 0
                max_run_count = 0
                min_run_pnl = 0
                min_run_count = 0

        # Values to select for losing runs
        else:

            # If there are any losing trades
            if pnl:
                max_run_pnl = pnl[0][0]
                max_run_count = pnl[0][-1]
                min_run_pnl = pnl[-1][0]
                min_run_count = pnl[-1][-1]

            # Otherwise set the values to zero
            else:
                max_run_pnl = 0
                max_run_count = 0
                min_run_pnl = 0
                min_run_count = 0

        min_max_run_dict = {
            'max_run_pnl':max_run_pnl,
            'max_run_count':max_run_count,
            'min_run_pnl':min_run_pnl,
            'min_run_count':min_run_count
            }

        return min_max_run_dict


    @staticmethod
    def _calc_run_names(run_type: str) -> dict:

        max_run_pnl_str = 'max_'+run_type+'_run_pnl'
        max_run_count_str = 'max_'+run_type+'_run_count'
        min_run_pnl_str = 'min_'+run_type+'_run_pnl'
        min_run_count_str = 'min_'+run_type+'_run_count'
        num_runs_str = 'num_'+run_type+'_runs'
        av_run_count_str = 'av_'+run_type+'_run_count'
        av_run_pnl_str = 'av_'+run_type+'_run_pnl'
        pnl_str = run_type+'_pnl'


        name_dict = {
            'max_run_pnl_str':max_run_pnl_str,
            'max_run_count_str':max_run_count_str,
            'min_run_pnl_str':min_run_pnl_str,
            'min_run_count_str':min_run_count_str,
            'num_runs_str':num_runs_str,
            'av_run_count_str':av_run_count_str,
            'av_run_pnl_str':av_run_pnl_str,
            'pnl_str':pnl_str
            }

        return name_dict
