"""
Calculate performance for various technical trading systems and display the
results in table and graph form.

"""

# Imports
import copy
import pandas as pd
from tradingsystems.graphs import PerformanceGraph as perfgraph
from tradingsystems.marketdata import Markets
from tradingsystems.pnl import Profit
from tradingsystems.reports import PerfReport
from tradingsystems.signals import Entry, Exit
from tradingsystems.trades import Positions
from tradingsystems.systems_params import system_params_dict
from tradingsystems.utils import Labels, Dates


class TestStrategy():
    """
    Run a backtest over the chosen strategy

    Parameters
    ----------
    ticker : Str, optional
        Underlying to test. The default '$SPX'.
    ccy_1 : Str, optional
        Primary currency of pair to return. The default 'GBP'.
    ccy_2 : Str, optional
        Secondary currency of pair to return. The default 'USD'.
    asset_type : Str
        The alphavantage asset class type. The default is 'fx'.
    start_date : Str, optional
        Date to begin backtest. Format is YYYY-MM-DD.
    end_date : Str, optional
        Date to end backtest. Format is YYYY-MM-DD.
    lookback : Int, optional
        Number of business days to use for the backtest. The default is 750
        business days (circa 3 years).
    ma1 : Int, optional
        The first moving average period.
    ma2 : Int, optional
        The second moving average period.
    ma3 : Int, optional
        The third moving average period.
    ma4 : Int, optional
        The fourth moving average period.
    simple_ma : Bool, optional
        Whether to calculate a simple or exponential moving average. The
        default is True.
    position_size : Int, optional
        The number of units to trade. The default is based on equity.
    pos_size_fixed : Bool
        Whether to used a fixed position size for all trades. The default
        is True.
    ticker_source : Str, optional
        The data source to use for the ticker data, either 'norgate',
        'alpha' or 'yahoo'. The default is 'norgate'.
    bench_ticker : Str, optional
        Underlying to use as benchmark. The default '$SPX'.
    bench_source : Str, optional
        The data source to use for the benchmark data, either 'norgate',
        'alpha' or 'yahoo'. The default is 'norgate'.
    slippage : Float, optional
        The amount of slippage to apply to traded prices in basis points.
        The default is 5 bps per unit.
    commission : Float, optional
        The amount of commission charge to apply to each trade. The
        default is $0.00.
    riskfree : Float, optional
        The riskfree interest rate. The default is 25bps.
    entry_type : Str, optional
        The entry strategy. The default is '2ma'.
    exit_type : Str, optional
        The exit strategy. The default is 'trailing_stop'.
    stop_type : Str, optional
        The stop strategy. The default is 'initial_dollar'.
    entry_period : Int, optional
        The number of days to use in the entry strategy. The default is 14.
    exit_period : Int, optional
        The number of days to use in the exit strategy. The default is 5.
    stop_period : Int, optional
        The number of days to use in the stop strategy. The default is 5.
    entry_oversold : Int, optional
        The oversold level to use in the entry strategy.
    entry_overbought : Int, optional
        The overbought level to use in the entry strategy.
    exit_oversold : Int, optional
        The oversold level to use in the exit strategy.
    exit_overbought : Int, optional
        The overbought level to use in the exit strategy.
    entry_threshold : Float
        The entry threshold used for momentum / volatility strategies.
        The default is 0 for momentum and 1.5 for volatility.
    exit_threshold : Float
        The exit threshold used for the volatility strategy.
        The default is 1.
    entry_acceleration_factor : Float
        The acceleration factor used in the Parabolic SAR entry signal.
        The default is 0.02.
    exit_acceleration_factor : Float
        The acceleration factor used in the Parabolic SAR exit signal.
        The default is 0.02.
    sip_price : Bool
        Whether to set the SIP of the Parabolic SAR exit to n-day
        high / low or to the high of the previous trade. The default is
        False.
    equity : Float
        The initial account equity level. The default is $100,000.00.
    exit_amount : Float
        The dollar exit amount. The default is $1000.00.
    stop_amount : Float
        The dollar stop amount. The default is $500.00.
    api_key : Str
        AlphaVantage API key. If not provided will look for
        'ALPHAVANTAGE_API_KEY' in the environment variables.

    Returns
    -------
    Results
        Prints out performance data for the strategy.

    """

    def __init__(self, **kwargs):

        # Import dictionary of default parameters
        self.default_dict = copy.deepcopy(system_params_dict)

        # Longnames for Norgate Tickers
        self.norgate_name_dict = Markets.norgate_name_dict()

        # Store initial inputs
        inputs = {}
        for key, value in kwargs.items():
            inputs[key] = value

        # Initialise system parameters
        params = self._init_params(inputs)

        # Set the start and end dates if not provided
        params['start_date'], params['end_date'] = Dates.date_set(
            start_date=params['start_date'], end_date=params['end_date'],
            lookback=params['lookback'])

        # Create DataFrame of OHLC prices from NorgateData or Yahoo Finance
        tables = {}

        tables['prices'] = Markets.create_base_data(
            ticker=params['ticker'], source=params['ticker_source'],
            params=params)

        # Extract benchmark data for Beta calculation
        tables['benchmark'] = Markets.create_base_data(
            ticker=params['bench_ticker'], source=params['bench_source'],
            params=params)

        # Set the strategy labels
        labels  = {}
        labels['entry_label'], labels['exit_label'], \
            labels['stop_label'] = Labels.strategy_labels(
                params=params, default_dict=self.default_dict)

        # Generate initial trade data
        tables['prices'], params['start'], params['position_size'], \
            params['benchmark_position_size'], \
                raw_trade_price_dict = self._raw_entry_signals(
                    tables=tables, params=params)

        # Create exit and stop targets
        tables['prices'] = self._exit_stop_targets(
            prices=tables['prices'], params=params,
            position_size=params['position_size'],
            trade_price_dict=raw_trade_price_dict)

        # Create exit and stop signals
        tables['prices'] = Exit.exit_and_stop_signals(
            prices=tables['prices'],
            trade_number=tables['prices']['raw_trade_number'],
            end_of_day_position=tables['prices']['raw_end_of_day_position'],
            params=params)

        # Concatenate the Entry, Exit and Stop signals in a single DataFrame
        trade_signals = pd.concat(
            [tables['prices']['raw_trade_signal'],
             tables['prices']['exit_signal'],
             tables['prices']['stop_signal']], axis=1)

        # Generate single combined trade signal
        tables['prices']['combined_signal'] = Positions.signal_combine(
            prices=tables['prices'], start=params['start'],
            end_of_day_position=tables['prices']['raw_end_of_day_position'],
            trade_signals=trade_signals)

        # Create trade and position data
        pos_dict = Positions.positions_and_trade_actions(
            prices=tables['prices'],
            signal=tables['prices']['combined_signal'],
            start=params['start'],
            position_size=params['position_size'])

        for key, value in pos_dict.items():
            tables['prices'][key] = value

        tables['prices']['trade_number'] = Positions.trade_numbers(
            prices=tables['prices'],
            end_of_day_position=tables['prices']['end_of_day_position'],
            start=params['start'])

        # Calculate the trades and pnl for the strategy
        tables['prices'] = Profit.profit_data(
            prices=tables['prices'],
            params=params)

        # Create monthly summary data
        tables['monthly_data'] = Profit.create_monthly_data(
            prices=tables['prices'], equity=params['equity'])

        # Create dictionary of performance data
        tables['perf_dict'] = PerfReport.performance_data(
            tables=tables, params=params, labels=labels,
            norgate_name_dict=self.norgate_name_dict,)

        # Print out results
        PerfReport.report_table(input_dict=tables['perf_dict'])

        self.params = params
        self.tables = tables
        self.labels = labels
        self.inputs = inputs


    @staticmethod
    def _init_params(inputs):

        params = copy.deepcopy(system_params_dict['df_params'])

        entry_signal_dict = system_params_dict['df_entry_signal_dict']
        exit_signal_dict = system_params_dict['df_exit_signal_dict']
        stop_signal_dict = system_params_dict['df_stop_signal_dict']

        # For all the supplied arguments
        for key, value in inputs.items():

            # Replace the default parameter with that provided
            params[key] = value

        # Set the start and end dates to None if not supplied
        if 'start_date' not in inputs.keys():
            params['start_date'] = None

        if 'end_date' not in inputs.keys():
            params['end_date'] = None

        # Create a list of the entry, exit and stop types
        types = [
            params['entry_type'], params['exit_type'], params['stop_type']]

        # For each parameter in params
        for param in params.keys():

            # If the parameter has not been supplied as an input and it is not
            # the entry exit or stop type
            if (param not in inputs.keys()
                and param not in types):

                # If the parameter takes a specific value for the particular
                # entry type then replace the default with this value
                if param in entry_signal_dict[types[0]].keys():
                    params[param] = entry_signal_dict[types[0]][str(param)]

                # If the parameter takes a specific value for the particular
                # exit type then replace the default with this value
                if param in exit_signal_dict[types[1]].keys():
                    params[param] = exit_signal_dict[types[1]][str(param)]

                # If the parameter takes a specific value for the particular
                # stop type then replace the default with this value
                if param in stop_signal_dict[types[2]].keys():
                    params[param] = stop_signal_dict[types[2]][str(param)]

        return params


    @classmethod
    def _raw_entry_signals(cls, tables, params):
        """
        Generate the initial raw entry signals, positions and trades

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        entry_type : Str, optional
            The entry strategy. The default is '2ma'.
        ma1 : Int, optional
            The first moving average period.
        ma2 : Int, optional
            The second moving average period.
        ma3 : Int, optional
            The third moving average period.
        ma4 : Int, optional
            The fourth moving average period.
        entry_period : Int
            The number of days to use in the entry strategy. The default is 14.
        entry_oversold : Int
            The oversold level to use in the entry strategy.
        entry_overbought : Int
            The overbought level to use in the entry strategy.
        entry_threshold : Float
            The entry threshold used for momentum / volatility strategies.
            The default is 0 for momentum and 1.5 for volatility.
        simple_ma : Bool
            Whether to calculate a simple or exponential moving average. The
            default is True.
        benchmark : Series
            The series of closing prices of the benchmark underlying.
        entry_acceleration_factor : Float
            The acceleration factor used in the Parabolic SAR entry signal.
            The default is 0.02.

        Returns
        -------
        prices : DataFrame
            The OHLC data.
        start : Int
            The first valid row to start calculating trade information from.

        """
        # Generate entry signals
        prices, start, \
            tables['prices']['raw_trade_signal'] = Entry.entry_signal(
                tables=tables, params=params)

        # Set the position size
        position_size, benchmark_position_size = Positions.position_size(
                prices=prices, benchmark=tables['benchmark'],
                equity=params['equity'])

        # Calculate initial position info
        raw_pos_dict = Positions.positions_and_trade_actions(
            prices=prices,
            signal=prices['raw_trade_signal'],
            start=start,
            position_size=position_size)

        # Generate trade numbers
        prices['raw_trade_number'] = Positions.trade_numbers(
            prices=prices,
            end_of_day_position=raw_pos_dict['end_of_day_position'],
            start=start)

        # Generate initial trade prices
        raw_trade_price_dict = Positions.trade_prices(
            prices=prices,
            trade_number=prices['raw_trade_number'])

        prices = cls._map_to_prices(
            prices=prices,
            pos_dict=raw_pos_dict,
            trade_price_dict=raw_trade_price_dict,
            title_modifier='raw_')

        return prices, start, position_size, benchmark_position_size, \
            raw_trade_price_dict


    @staticmethod
    def _map_to_prices(prices, pos_dict, trade_price_dict, title_modifier):

        for key, value in pos_dict.items():
            prices[title_modifier+key] = value

        for key, value in trade_price_dict.items():
            prices[title_modifier+key] = value

        return prices


    @classmethod
    def _exit_stop_targets(
            cls, prices, position_size, params, trade_price_dict):
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

        prices = cls._exit_targets(
            prices=prices, exit_amount=params['exit_amount'],
            position_size=position_size,
            trade_price_dict=trade_price_dict)

        prices = cls._stop_targets(
            prices=prices, stop_amount=params['stop_amount'],
            position_size=position_size,
            trade_price_dict=trade_price_dict)

        return prices


    @classmethod
    def _exit_targets(
            cls, prices, exit_amount, position_size, trade_price_dict):
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
                prices['exit_trailing_high_low'] = Positions.pnl_targets(
                    prices=prices, dollar_amount=exit_amount,
                    position_size=position_size,
                    end_of_day_position=prices['raw_end_of_day_position'],
                    trade_price_dict=trade_price_dict)

        return prices


    @classmethod
    def _stop_targets(
            cls, prices, stop_amount, position_size, trade_price_dict):
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
                prices['stop_trailing_high_low'] = Positions.pnl_targets(
                    prices=prices, dollar_amount=stop_amount,
                    position_size=position_size,
                    end_of_day_position=prices['raw_end_of_day_position'],
                    trade_price_dict=trade_price_dict)

        return prices


    def performance_graph(self, signals=None):
        """
        Graph the performance of the strategy

        Parameters
        ----------
        signals : Bool
            Whether to plot the Buy/Sell signals on the price chart.

        Returns
        -------
        Matplotlib Chart
            Displays the chart.

        """

        # If the signals flag is not provided, take the default value
        if signals is None:
            signals = self.default_dict['df_signals']

        # Dictionary to store entry signal data
        es_dict = {}

        # Entry labels
        es_dict['entry_signal_labels'] = self.default_dict[
            'df_entry_signal_labels']

        # Entry signal indicator column names
        es_dict['entry_signal_indicators'] = self.default_dict[
            'df_entry_signal_indicators']

        params = self.params
        #labels = self.labels
        tables = self.tables

        # For these entry strategies just plot the price chart and equity curve
        if params['entry_type'] in [
                '2ma', '3ma', '4ma', 'sar', 'channel_breakout']:
            graph = perfgraph.two_panel_graph(
                signals=signals, tables=tables, params=params,
                es_dict=es_dict)

        # Otherwise also plot the relevant indicator
        else:
            graph = perfgraph.three_panel_graph(
                signals=signals, tables=tables, params=params,
                es_dict=es_dict)

        return graph
