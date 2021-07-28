"""
Reporting tools for the trading system data

"""

import datetime as dt
import math
from decimal import Decimal
import numpy as np
from scipy.stats import skew, kurtosis
from tradingsystems.pnl import Profit


class PerfReport():
    """
    Create a dictionary of performance data and produce a text report of the
    system performance.

    """

    @classmethod
    def performance_data(cls, tables, params, labels, norgate_name_dict):
        """
        Create dictionary of performance data.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data, trades signals and pnl.
        monthly_data : DataFrame
            The monthly summary data.
        ticker_source : Str
            The data source to use for the ticker data, either 'norgate',
            'alpha' or 'yahoo'. The default is 'norgate'.
        asset_type : Str
            The alphavantage asset class type. The default is 'fx'.
        ccy_1 : Str
            Primary currency of pair to return. The default 'GBP'.
        ccy_2 : Str
            Secondary currency of pair to return. The default 'USD'.
        ticker : Str
            Underlying to test. The default '$SPX'.
        entry_label : Str
            The longname of the entry strategy.
        exit_label : Str
            The longname of the exit strategy.
        stop_label : Str
            The longname of the stop strategy.
        norgate_name_dict : Dict
            Dictionary lookup of Norgate tickers to long names.
        slippage : Float
            The amount of slippage to apply to traded prices in basis points.
            The default is 5 bps per unit.
        commission : Float
            The amount of commission charge to apply to each trade. The
            default is $0.00.
        position_size : Int
            The number of units to trade. The default is based on equity.
        benchmark : Series
            The closing prices of the benchmark.
        benchmark_position_size : Int
            The number of units of the benchmark to trade. The default is based
            on equity.
        riskfree : Float, optional
            The riskfree interest rate. The default is 25bps.

        Returns
        -------
        perf_dict : Dict
            Dictionary of performance data.

        """

        # Create empty dictionary
        perf_dict = {}

        prices = tables['prices']

        perf_dict = cls._perf_data_init(
            perf_dict=perf_dict, prices=prices, params=params, labels=labels,
            norgate_name_dict=norgate_name_dict)

        # Maximum margin required
        perf_dict['margin'] = math.ceil(
            max(abs(prices['position_mtm'])) / 100) * 100

        # Net Profit
        perf_dict['net_pnl'] = prices['total_pnl'].iloc[-1]

        # Convert the start and end date strings to dates
        perf_dict['start'] = dt.datetime.strptime(
            perf_dict['start_date'], "%d/%m/%y")
        perf_dict['end'] = dt.datetime.strptime(
            perf_dict['end_date'], "%d/%m/%y")

        # Number of days between start and end dates
        perf_dict['period'] = (perf_dict['end']-perf_dict['start']).days

        # Set the return period in years
        perf_dict['return_period'] = perf_dict['period'] / 365

        # Annualized profit
        perf_dict['annualized_profit '] = (
            perf_dict['net_pnl'] / perf_dict['return_period'])

        # Total return
        perf_dict['total_return_rate'] = (
            perf_dict['net_pnl'] / perf_dict['initial_equity'])

        # Annualized total return
        perf_dict['annualized_return'] = (
            (1 + perf_dict['total_return_rate'])
            ** (1 / perf_dict['return_period']) - 1)

        perf_dict = cls._perf_data_trades(perf_dict=perf_dict, tables=tables)

        # Pessimistic Return on Margin
        perf_dict['adjusted_gross_profit'] = perf_dict['av_win'] * (
            perf_dict['num_wins'] - np.round(np.sqrt(perf_dict['num_wins'])))
        perf_dict['adjusted_gross_loss'] = perf_dict['av_loss'] * (
            perf_dict['num_losses'] - np.round(
                np.sqrt(perf_dict['num_losses'])))
        perf_dict['prom'] = np.round(
            (perf_dict['adjusted_gross_profit']
             - perf_dict['adjusted_gross_loss']) /
            perf_dict['margin'], 2)

        # PROM minus biggest win
        perf_dict['prom_minus_max_win'] = np.round(
            (perf_dict['adjusted_gross_profit'] - perf_dict['max_win']
             - perf_dict['adjusted_gross_loss']) / perf_dict['margin'], 2)

        # Perfect profit - Buy every dip & sell every peak
        perf_dict['perfect_profit'] = prices['total_perfect_profit'].iloc[-1]

        # Model Efficiency - ratio of net pnl to perfect profit
        perf_dict['model_efficiency'] = np.round(
            (perf_dict['net_pnl'] / perf_dict['perfect_profit']) * 100, 2)

        # Winning run data
        perf_dict['win_run_dict'] = Profit.trade_runs(
            perf_dict['trade_data_dict']['trades_win_list'], run_type='win')

        # Losing run data
        perf_dict['loss_run_dict'] = Profit.trade_runs(
            perf_dict['trade_data_dict']['trades_loss_list'], run_type='loss')

        # Maximum Equity drawdown
        perf_dict['max_balance_drawback'] = np.round(min(prices['max_dd']), 2)

        # Maximum Equity drawdown in percentage terms
        perf_dict['max_balance_drawback_perc'] = min(
            prices['max_dd_perc']) * 100

        # Time to recover from Max Drawdown
        perf_dict['time_to_recover'] = Profit.time_to_recover(prices)

        # Maximum Equity gain
        perf_dict['max_gain'] = np.round(max(prices['max_gain']), 2)

        # Maximum Equity gain in percentage terms
        perf_dict['max_gain_perc'] = max(prices['max_gain_perc']) * 100

        # Time taken for maximum gain
        perf_dict['max_gain_time'] = Profit.time_max_gain(prices)

        # Reward / Risk ratio
        perf_dict['reward_risk'] = (perf_dict['net_pnl'] /
                                    abs(perf_dict['max_balance_drawback']))

        # Annual Rate of Return
        perf_dict['annual_ror'] = (perf_dict['annualized_profit '] /
                                   perf_dict['initial_equity']) * 100

        # Profit Index
        perf_dict['profit_factor'] = (perf_dict['total_profit'] /
                                     abs(perf_dict['total_loss']))

        # Mathematical Advantage
        perf_dict['mathematical_advantage'] = np.round(
            (perf_dict['num_wins']
             / len(perf_dict['trade_data_dict']['trades']) * 100)
            * (perf_dict['profit_factor'] + 1) - 100, 2)

        # Get the index of the first trade entry
        # Select just the rows of the first trade from the DataFrame
        first_trade = prices[prices['trade_number']==1]

        # Find the date of the trade entry
        target = first_trade.index[0]

        # Find the location of this in the original index
        first_trade_start = prices.index.get_loc(target)

        # Return of a buy and hold strategy since the first trade entry
        perf_dict['long_only_pnl'] = (
            (prices['Close'].iloc[-1]
             - prices['Open'].iloc[first_trade_start]) *
            params['position_size'])

        # Annual Rate of return of buy and hold
        perf_dict['annual_long_only_ror'] = (
            (perf_dict['long_only_pnl'] / perf_dict['return_period']) /
            perf_dict['initial_equity']) * 100

        # Return of a buy and hold strategy of SPX since the first trade entry
        perf_dict['long_only_pnl_spx'] = (
            (tables['benchmark']['Close'].iloc[-1]
             - tables['benchmark']['Open'].iloc[first_trade_start]) *
            params['benchmark_position_size'])

        # Annual Rate of return of buy and hold of SPX
        perf_dict['annual_long_only_spx_ror'] = (
            (perf_dict['long_only_pnl_spx'] / perf_dict['return_period']) /
            perf_dict['initial_equity']) * 100

        perf_dict = cls._perf_data_stat_measures(
            perf_dict=perf_dict, tables=tables, params=params)

        # Maximum MTM Equity Profit
        perf_dict['max_equity_profit'] = (
            prices['max_mtm_equity'].iloc[-1] - perf_dict['initial_equity'])

        # Largest open equity drawdown
        perf_dict['open_equity_dd'] = (prices['trade_pnl_drawback'].max())*(-1)

        # Open equity
        perf_dict['open_equity'] = prices['open_equity'].iloc[-1]

        # Closed equity
        perf_dict['closed_equity'] = prices['closed_equity'].iloc[-1]

        # Largest monthly gain
        perf_dict['month_net_pnl_large'] = (
            tables['monthly_data']['total_net_profit'].max())

        # Largest monthly loss
        perf_dict['month_net_pnl_small'] = (
            tables['monthly_data']['total_net_profit'].min())

        # Average monthly gain/loss
        perf_dict['month_net_pnl_av'] = (
            tables['monthly_data']['total_net_profit'].mean())

        # Values still to be worked out
        placeholder_dict = {
            'pessimistic_margin':0.00,
            'adj_pess_margin':0.00,
            'pess_month_avg':0.00,
            'pess_month_variance':0.00,
            'mod_pess_margin':0.00,
            }

        perf_dict.update(placeholder_dict)

        return perf_dict


    @staticmethod
    def _perf_data_init(perf_dict, prices, params, labels, norgate_name_dict):

        # Contract and strategy details
        if (params['ticker_source'] == 'alpha'
            and params['asset_type'] in ['fx', 'crypto']):
            perf_dict['contract'] = params['ccy_1'] + params['ccy_2']
        else:
            perf_dict['contract'] = params['ticker']

        # Entry, exit and stop labels
        perf_dict['entry_label'] = labels['entry_label']
        perf_dict['exit_label'] = labels['exit_label']
        perf_dict['stop_label'] = labels['stop_label']

        # Initial Equity
        perf_dict['initial_equity'] = prices['mtm_equity'].iloc[0]

        # Set Ticker Longname
        if params['ticker_source'] == 'norgate':
            perf_dict['longname'] = norgate_name_dict[params['ticker']]
        else:
            perf_dict['longname'] = perf_dict['contract']

        # Slippage and commission in dollars
        perf_dict['slippage'] = params['slippage']
        perf_dict['commission'] = params['commission']

        # Start and end dates
        perf_dict['start_date'] = prices.index[0].date().strftime("%d/%m/%y")
        perf_dict['end_date'] = prices.index[-1].date().strftime("%d/%m/%y")

        return perf_dict


    @staticmethod
    def _perf_data_trades(perf_dict, tables):

        prices = tables['prices']

        # Calculate trade data
        perf_dict['trade_data_dict'] = Profit.trade_data(prices)

        perf_dict['total_trades'] = perf_dict['trade_data_dict']['num_trades']

        # number of wins and losses
        perf_dict['num_wins'] = len(
            perf_dict['trade_data_dict']['trades_win_dict'])
        perf_dict['num_losses'] = len(
            perf_dict['trade_data_dict']['trades_loss_dict'])

        # Sum of all profits
        perf_dict['total_profit'] = np.round(
            sum(perf_dict['trade_data_dict']['trades_win_dict'].values()), 2)

        # Sum of all losses
        perf_dict['total_loss'] = np.round(
            sum(perf_dict['trade_data_dict']['trades_loss_dict'].values()), 2)

        # Percentage of winning trades
        perf_dict['win_percent'] = int(
            np.round(perf_dict['num_wins'] / len(
                perf_dict['trade_data_dict']['trades']) * 100, 0))

        # Percentage of losing trades
        perf_dict['loss_percent'] = int(
            np.round(perf_dict['num_losses']
                     / len(perf_dict['trade_data_dict']['trades']) * 100, 0))


        # If there are winning trades
        if perf_dict['trade_data_dict']['trades_win_dict'].values():

            # Max winning trade
            perf_dict['max_win'] = np.round(
                max(perf_dict['trade_data_dict']['trades_win_dict'].values()),
                2)

            # Min winning trade
            perf_dict['min_win'] = np.round(
                min(perf_dict['trade_data_dict']['trades_win_dict'].values()),
                2)

            # Average winning trade
            perf_dict['av_win'] = np.round(
                sum(perf_dict['trade_data_dict']['trades_win_dict'].values())
                / len(perf_dict['trade_data_dict']['trades_win_list']), 2)

            # standard deviation of winning trades
            perf_dict['std_wins'] = np.round(
                np.std(perf_dict['trade_data_dict']['trades_win_list']), 2)

        # Otherwise set these to zero
        else:
            perf_dict['max_win'] = 0
            perf_dict['min_win'] = 0
            perf_dict['av_win'] = 0
            perf_dict['std_wins'] = 0


        # If there are losing trades
        if perf_dict['trade_data_dict']['trades_loss_dict'].values():

            # Max losing trade
            perf_dict['max_loss'] = min(
                perf_dict['trade_data_dict']['trades_loss_dict'].values())

            # Min losing trade
            perf_dict['min_loss'] = max(
                perf_dict['trade_data_dict']['trades_loss_dict'].values())

            # average losing trade
            perf_dict['av_loss'] = np.round(
                sum(perf_dict['trade_data_dict']['trades_loss_dict'].values())
                / len(perf_dict['trade_data_dict']['trades_loss_list']), 2)

            # standard deviation of losing trades
            perf_dict['std_losses'] = np.round(
                np.std(perf_dict['trade_data_dict']['trades_loss_list']), 2)

        # Otherwise set these to zero
        else:
            perf_dict['max_loss'] = 0
            perf_dict['min_loss'] = 0
            perf_dict['av_loss'] = 0
            perf_dict['std_losses'] = 0


        # Maximum win/loss ratio
        perf_dict['max_win_loss_ratio'] = np.round(
            abs(perf_dict['max_win'] / perf_dict['max_loss']), 2)


        # Minimum win/loss ratio
        perf_dict['min_win_loss_ratio'] = np.round(
            abs(perf_dict['min_win'] / perf_dict['min_loss']), 2)


        # avwin / avloss
        perf_dict['av_win_loss_ratio'] = np.round(
            abs(perf_dict['av_win'] / perf_dict['av_loss']), 2)

        # average trade
        perf_dict['av_trade'] = np.round(
            perf_dict['net_pnl']
            / len(perf_dict['trade_data_dict']['trades']), 2)

        return perf_dict


    @staticmethod
    def _perf_data_stat_measures(perf_dict, tables, params):

        prices = tables['prices']

        # Riskfree Rate
        perf_dict['riskfree_rate'] = params['riskfree']

        # Mean Price
        perf_dict['close_price_mean'] = np.round(np.mean(prices['Close']), 2)

        # Variance Price
        perf_dict['close_price_variance'] = np.round(
            np.var(prices['Close']), 2)

        # Standard Deviation Price
        perf_dict['close_price_std_dev'] = np.round(np.std(prices['Close']), 2)

        # Skewness Price
        perf_dict['close_price_skewness'] = np.round(skew(prices['Close']), 2)

        # Kurtosis Price
        perf_dict['close_price_kurtosis'] = np.round(
            kurtosis(prices['Close']), 2)

        # Mean Return
        perf_dict['close_return_mean'] = np.round(
            np.mean(prices['Close'].pct_change()*100), 2)

        # Variance Return
        perf_dict['close_return_variance'] = np.round(
            np.var(prices['Close'].pct_change()*100), 2)

        # Standard Deviation Return
        perf_dict['close_return_std_dev'] = np.round(
            np.std(prices['Close'].pct_change()*100), 2)

        # Annualized Volatility
        perf_dict['close_return_ann_vol'] = np.round(
            np.std(prices['Close'].pct_change()*100)*np.sqrt(252), 2)

        # Skewness Return
        perf_dict['close_return_skewness'] = np.round(
            skew(prices['Close'].pct_change()*100, nan_policy='omit'), 2)

        # Kurtosis Return
        perf_dict['close_return_kurtosis'] = np.round(kurtosis(
            prices['Close'].pct_change()*100, nan_policy='omit'), 2)

        # Efficiency Ratio
        perf_dict['efficiency_ratio'] = np.round(
            (abs(prices['Close'][-1] - prices['Close'][0])
            / np.nansum(abs(prices['Close'] - prices['Close'].shift())))
            * 100, 2)

        # MTM Equity Annualized Standard Deviation
        perf_dict['equity_std_dev'] = np.round(
            np.std(prices['mtm_equity'].pct_change()*100)*np.sqrt(252), 2)

        # Sharpe Ratio
        perf_dict['sharpe_ratio'] = ((
            perf_dict['annual_ror'] - perf_dict['riskfree_rate'])
            / perf_dict['equity_std_dev'])

        # Information Ratio
        perf_dict['information_ratio'] = (
            perf_dict['annual_ror'] / perf_dict['equity_std_dev'])

        # Treynor Ratio
        perf_dict['treynor_inv_correl'] = tables[
            'benchmark'].Close.pct_change().corr(
                prices.mtm_equity.pct_change())
        #stock_correl = benchmark.Close.pct_change().corr(
        #    prices.Close.pct_change())
        perf_dict['treynor_sd_inv'] = np.std(
            prices.mtm_equity.pct_change())*np.sqrt(252)
        #sd_stock = np.std(prices.Close.pct_change())
        perf_dict['treynor_sd_index'] = np.std(
            tables['benchmark'].Close.pct_change())*np.sqrt(252)
        #beta = stock_correl * (sd_stock / sd_index)
        perf_dict['treynor_beta'] = (perf_dict['treynor_inv_correl']
                                     * (perf_dict['treynor_sd_inv']
                                        / perf_dict['treynor_sd_index']))
        treynor = np.round(
            (perf_dict['annual_ror'] - perf_dict['riskfree_rate'])
            / perf_dict['treynor_beta'], 2)
        if treynor > 0:
            perf_dict['treynor_ratio'] = treynor
        else:
            perf_dict['treynor_ratio'] = 'N/A'

        # Sortino Ratio
        equity_return = prices['mtm_equity'].pct_change()*100
        downside_deviation = np.std(
            equity_return[equity_return < 0])*np.sqrt(252)
        perf_dict['sortino_ratio'] = np.round((
            perf_dict['annual_ror'] - perf_dict['riskfree_rate'])
            / downside_deviation, 2)

        # Calmar Ratio
        perf_dict['calmar_ratio'] = (
            perf_dict['annual_ror']
            / abs(perf_dict['max_balance_drawback_perc']))

        # Average Maximum Retracement
        perf_dict['average_max_retracement'] = (
            prices['max_retracement'].mean())

        # Ulcer Index
        perf_dict['ulcer_index'] = np.sqrt(
            prices['ulcer_index_d_sq'][prices['ulcer_index_d_sq']!=0].mean())

        # Gain to Pain
        perf_dict['gain_to_pain'] = np.round(
            tables['monthly_data']['return'].sum()
            / tables['monthly_data']['abs_loss'].sum(), 2)

        return perf_dict


    @classmethod
    def report_table(cls, input_dict):
        """
        Print out backtesting performance results.

        Modelled on performance results presented in Robert Pardo, 1992,
        Design, Testing and Optimization of Trading Systems

        Parameters
        ----------
        input_dict : Dict
            Dictionary of performance data.

        Returns
        -------
        Prints results to the console.

        """

        win_run_dict = input_dict['win_run_dict']
        loss_run_dict = input_dict['loss_run_dict']

        # Format the performance dictionary so that the financial data is
        # rounded to 2 decimal places and set values as strings
        input_dict = cls._dict_format(input_dict)
        win_run_dict = cls._dict_format(win_run_dict)
        loss_run_dict = cls._dict_format(loss_run_dict)

        # Core strategy details
        cls._report_header(input_dict)

        # Trade Win / Loss statistics
        cls._report_trades(input_dict, win_run_dict, loss_run_dict)

        # Equity return / drawdown statistics
        cls._report_equity(input_dict)

        # Key performance measures
        cls._report_key_perf(input_dict)

        # Data distribution statistics
        cls._report_data_dist(input_dict)


    @staticmethod
    def _report_header(input_dict):

        # Format header - centred and with lines above and below
        print('='*78)
        print('{:^78}'.format('Performance Analysis Report'))
        print('-'*78)

        # Contract traded on left and period covered on right
        print('Contract Ticker : {:<41}{} - {}'.format(
            input_dict['contract'],
            input_dict['start_date'],
            input_dict['end_date']))

        # Contract Longname
        print('Contract Name   : {:>10}'.format(input_dict['longname']))

        # Entry Method
        print('Entry           : {:>10}'.format(input_dict['entry_label']))

        # Exit Method
        print('Exit            : {:>10}'.format(input_dict['exit_label']))

        # Stop Method
        print('Stop            : {:>10}'.format(input_dict['stop_label']))

        # Beginning balance on left
        print('Initial Equity  :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['initial_equity'],
            '',
            'Margin           : $',
            input_dict['margin']))

        # Commission, slippage and margin
        print('Commission      :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['commission'],
            '',
            'Slippage         :  ',
            input_dict['slippage']+'bps'))

        # Empty line
        print()

        # Profit stats
        print('Net P/L         :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['net_pnl'],
            '',
            'Perfect Profit   : $',
            input_dict['perfect_profit']))
        print('Total Profit    :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['total_profit'],
            '',
            'Total Loss       : $',
            input_dict['total_loss']))
        print('Av. Trade P/L   :     ${:>10}{:>9}{}{:>11}'.format(
            input_dict['av_trade'],
            '',
            'Model Efficiency : %',
            input_dict['model_efficiency']))


    @staticmethod
    def _report_trades(input_dict, win_run_dict, loss_run_dict):

        # Headers for # trades, Max, Min and Average with lines above and below
        print('-'*78)
        print('{:>16}:{:^10}:{:^17}:{:^17}:{:^14}'.format(
            '',
            '# Trades',
            'Maximum',
            'Minimum',
            'Average'))
        print('-'*78)

        # Total trades summary
        print('Total...........:{:^10}:{:^17}:{:^17}:'.format(
            input_dict['total_trades'],
            'Win {} / {}%'.format(
                input_dict['num_wins'], input_dict['win_percent']),
            'Loss {} / {}%'.format(
                input_dict['num_losses'], input_dict['loss_percent'])))

        # Winning trades and winning runs
        print('Win.............:{:^10}:{:>11}{:>6}:{:>10}{:>7}:{:>9}'.format(
            input_dict['num_wins'],
            input_dict['max_win'],
            '',
            input_dict['min_win'],
            '',
            input_dict['av_win']))

        print('Win runs........:{:^10}:{:>11}{:<6}:{:>10}{:<7}:{:>9}{}'.format(
            win_run_dict['num_win_runs'],
            win_run_dict['max_win_run_pnl'],
            ' / {}'.format(win_run_dict['max_win_run_count']),
            win_run_dict['min_win_run_pnl'],
            ' / {}'.format(win_run_dict['min_win_run_count']),
            win_run_dict['av_win_run_pnl'],
            ' / {}'.format(win_run_dict['av_win_run_count'])))

        # Losing trades and losing runs
        print('Loss............:{:^10}:{:>11}{:>6}:{:>10}{:>7}:{:>9}'.format(
            input_dict['num_losses'],
            input_dict['max_loss'],
            '',
            input_dict['min_loss'],
            '',
            input_dict['av_loss']))

        print('Loss runs.......:{:^10}:{:>11}{:<6}:{:>10}{:<7}:{:>9}{}'.format(
            loss_run_dict['num_loss_runs'],
            loss_run_dict['max_loss_run_pnl'],
            ' / {}'.format(loss_run_dict['max_loss_run_count']),
            loss_run_dict['min_loss_run_pnl'],
            ' / {}'.format(loss_run_dict['min_loss_run_count']),
            loss_run_dict['av_loss_run_pnl'],
            ' / {}'.format(loss_run_dict['av_loss_run_count'])))

        # Win loss ratio
        print('Win/Loss Ratio..:{:>10}:{:>11}{:>6}:{:>10}{:>7}:{:>9}'.format(
            '',
            input_dict['max_win_loss_ratio'],
            '',
            input_dict['min_win_loss_ratio'],
            '',
            input_dict['av_win_loss_ratio']))

        # Separating line
        print('-'*78)


    @staticmethod
    def _report_equity(input_dict):

        # Open and Closed equity
        print('Open Equity.......... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['open_equity'],
            '',
            'Closed Equity......... $',
            input_dict['closed_equity']))

        # Max Open Equity DD / equity profit
        print('Max Open Eq. DD...... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['open_equity_dd'],
            '',
            'Max Equity Profit..... $',
            input_dict['max_equity_profit']))

        # Max Closed Equity DD drawback / Max Equity Gain
        print('Peak to Trough DD.... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['max_balance_drawback'],
            '',
            'Trough to Peak Gain... $',
            input_dict['max_gain']))

        # Max Percent drawdown / gain
        print('Max Drawdown......... %{:>10}{:<6}{}{:>10}'.format(
            input_dict['max_balance_drawback_perc'],
            '',
            'Max Gain.............. %',
            input_dict['max_gain_perc']))

        # Time to recover and time for max gain
        print('Time to Recover (Days) {:>10}{:<6}{}{:>10}'.format(
            input_dict['time_to_recover'],
            '',
            'Time for Max Gain (Days)',
            input_dict['max_gain_time']))

        # Reward / Risk
        print('Reward / Risk........  {:^16}{}{:>10}'.format(
            '{} TO 1'.format(input_dict['reward_risk']),
            'Annual Rate of Return. %',
            input_dict['annual_ror']))

        # Profit Index and Mathematical Advantage
        print('Profit Factor........  {:>10}{:<6}{}{:>10}'.format(
            input_dict['profit_factor'],
            '',
            'Mathematical Advantage %',
            input_dict['mathematical_advantage']))

        # Pessimistic Margin
        print('Pessimistic Margin...  {:>10}{:<6}{}{:>10}'.format(
            input_dict['pessimistic_margin'],
            '',
            'Adjusted Pess. Margin.  ',
            input_dict['adj_pess_margin']))

        print('Pess. Month Avg......  {:>10}{:<6}{}{:>10}'.format(
            input_dict['pess_month_avg'],
            '',
            'Pess. Month Variance..  ',
            input_dict['pess_month_variance']))

        # Monthly P/L
        print('Monthly Net PL Large. ${:>10}{:<6}{}{:>10}'.format(
            input_dict['month_net_pnl_large'],
            '',
            'Monthly Net PL Small.. $',
            input_dict['month_net_pnl_small']))

        print('Monthly Net PL Ave... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['month_net_pnl_av'],
            '',
            'Modified Pess. Margin.  ',
            input_dict['mod_pess_margin']))

        print('Long Only Net PL..... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['long_only_pnl'],
            '',
            'Long Only Annual RoR.. %',
            input_dict['annual_long_only_ror']))

        print('Long Only SPX Net PL. ${:>10}{:<6}{}{:>10}'.format(
            input_dict['long_only_pnl_spx'],
            '',
            'Long Only SPX Ann RoR. %',
            input_dict['annual_long_only_spx_ror']))


    @staticmethod
    def _report_key_perf(input_dict):

        # Key Performance Measures
        print('-'*78)
        print('{:^78}'.format('Key Performance Measures'))
        print('-'*78)

        # Sharpe Ratio & Information Ratio
        print('Sharpe Ratio.........  {:>10}{:<6}{}{:>10}'.format(
            input_dict['sharpe_ratio'],
            '',
            'Information Ratio.....  ',
            input_dict['information_ratio']))

        # Sortino Ratio & Treynor Ratio
        print('Sortino Ratio........  {:>10}{:<6}{}{:>10}'.format(
            input_dict['sortino_ratio'],
            '',
            'Treynor Ratio.........  ',
            input_dict['treynor_ratio']))

        # Calmar Ratio & Average Max Retracement
        print('Calmar Ratio.........  {:>10}{:<6}{}{:>10}'.format(
            input_dict['calmar_ratio'],
            '',
            'Av. Max Retracement... $',
            input_dict['average_max_retracement']))

        # Ulcer Index & Gain to Pain
        print('Ulcer Index..........  {:>10}{:<6}{}{:>10}'.format(
            input_dict['ulcer_index'],
            '',
            'Gain to Pain..........  ',
            input_dict['gain_to_pain']))

        # Price Distribution statistics
        print('-'*78)
        print('{:^78}'.format('Data Distribution Statistics'))
        print('-'*78)

        # Mean & Standard Deviation of Prices
        print('Mean Price........... ${:>10}{:<6}{}{:>10}'.format(
            input_dict['close_price_mean'],
            '',
            'St. Dev. Price........ $',
            input_dict['close_price_std_dev']))

        # Skewness & Kurtosis of Prices
        print('Skewness Price.......  {:>10}{:<6}{}{:>10}'.format(
            input_dict['close_price_skewness'],
            '',
            'Kurtosis Price........  ',
            input_dict['close_price_kurtosis']))


    @staticmethod
    def _report_data_dist(input_dict):

        # Return Distribution statistics

        # Mean & Standard Deviation of Returns
        print('Mean Return.......... %{:>10}{:<6}{}{:>10}'.format(
            input_dict['close_return_mean'],
            '',
            'St. Dev. Return....... %',
            input_dict['close_return_std_dev']))

        # Skewness & Kurtosis of Returns
        print('Skewness Return......  {:>10}{:<6}{}{:>10}'.format(
            input_dict['close_return_skewness'],
            '',
            'Kurtosis Return.......  ',
            input_dict['close_return_kurtosis']))

        # Efficiency Ratio & Annualized Volatility
        print('Efficiency Ratio..... %{:>10}{:<6}{}{:>10}'.format(
            input_dict['efficiency_ratio'],
            '',
            'Annualized Vol........ %',
            input_dict['close_return_ann_vol']))

        # Closing line
        print('='*78)


    @staticmethod
    def _dict_format(input_dict):
        """
        Format the performance dictionary so that the financial data is
        rounded to 2 decimal places and set values as strings

        Parameters
        ----------
        input_dict : Dict
            The performance data dictionary.

        Returns
        -------
        str_input_dict : Dict
            The formatted performance data dictionary.

        """

        # Create empty dictionary
        str_input_dict = {}

        # Set decimal format
        dp2 = Decimal(10) ** -2  # (equivalent to Decimal '0.01')

        # For each entry in the dictionary
        for key, value in input_dict.items():

            # If the value is a floating point number
            if type(value) in (float, np.float64):

                # Apply the decimal formatting and convert to string
                str_input_dict[key] = str(Decimal(value).quantize(dp2))
            else:

                # Otherwise just convert to string
                str_input_dict[key] = str(value)

        return str_input_dict
