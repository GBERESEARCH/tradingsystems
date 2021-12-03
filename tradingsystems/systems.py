"""
Calculate performance for various technical trading systems and display the
results in table and graph form.

"""

# Imports
import copy
from tradingsystems.graphs import PerformanceGraph as perfgraph
from tradingsystems.marketdata import Markets
from tradingsystems.positions import Positions
from tradingsystems.pnl import Profit
from tradingsystems.reports import PerfReport
from tradingsystems.signals import Signals
from tradingsystems.systems_params import system_params_dict
from tradingsystems.targets import TradeTargets
from tradingsystems.trades import Trades
from tradingsystems.utils import Labels, Dates, Reformat


class TestStrategy():
    """
    Run a backtest over the chosen strategy

    Parameters
    ----------

    api_key : Str
        AlphaVantage API key. If not provided will look for
        'ALPHAVANTAGE_API_KEY' in the environment variables.
    asset_type : Str
        The alphavantage asset class type. The default is 'fx'.
    bench_source : Str, optional
        The data source to use for the benchmark data, either 'norgate',
        'alpha' or 'yahoo'. The default is 'norgate'.
    bench_ticker : Str, optional
        Underlying to use as benchmark. The default '$SPX'.
    commission : Float, optional
        The amount of commission charge to apply to each trade. The
        default is $0.00.
    ccy_1 : Str, optional
        Primary currency of pair to return. The default 'GBP'.
    ccy_2 : Str, optional
        Secondary currency of pair to return. The default 'USD'.
    end_date : Str, optional
        Date to end backtest. Format is YYYY-MM-DD.
    entry_acceleration_factor : Float
        The acceleration factor used in the Parabolic SAR entry signal.
        The default is 0.02.
    entry_overbought : Int, optional
        The overbought level to use in the entry strategy.
    entry_oversold : Int, optional
        The oversold level to use in the entry strategy.
    entry_period : Int, optional
        The number of days to use in the entry strategy. The default is 14.
    entry_threshold : Float
        The entry threshold used for momentum / volatility strategies.
        The default is 0 for momentum and 1.5 for volatility.
    entry_type : Str, optional
        The entry strategy. The default is '2ma'.
    equity : Float
        The initial account equity level. The default is $100,000.00.
    exit_acceleration_factor : Float
        The acceleration factor used in the Parabolic SAR exit signal.
        The default is 0.02.
    exit_amount : Float
        The dollar exit amount. The default is $1000.00.
    exit_oversold : Int, optional
        The oversold level to use in the exit strategy.
    exit_overbought : Int, optional
        The overbought level to use in the exit strategy.
    exit_period : Int, optional
        The number of days to use in the exit strategy. The default is 5.
    exit_threshold : Float
        The exit threshold used for the volatility strategy.
        The default is 1.
    exit_type : Str, optional
        The exit strategy. The default is 'trailing_stop'.
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
    position_size : Int, optional
        The number of units to trade. The default is based on equity.
    pos_size_fixed : Bool
        Whether to used a fixed position size for all trades. The default
        is True.
    riskfree : Float, optional
        The riskfree interest rate. The default is 25bps.
    simple_ma : Bool, optional
        Whether to calculate a simple or exponential moving average. The
        default is True.
    sip_price : Bool
        Whether to set the SIP of the Parabolic SAR exit to n-day
        high / low or to the high of the previous trade. The default is
        False.
    slippage : Float, optional
        The amount of slippage to apply to traded prices in basis points.
        The default is 5 bps per unit.
    start_date : Str, optional
        Date to begin backtest. Format is YYYY-MM-DD.
    stop_amount : Float
        The dollar stop amount. The default is $500.00.
    stop_period : Int, optional
        The number of days to use in the stop strategy. The default is 5.
    stop_type : Str, optional
        The stop strategy. The default is 'initial_dollar'.
    ticker : Str, optional
        Underlying to test. The default '$SPX'.
    ticker_source : Str, optional
        The data source to use for the ticker data, either 'norgate',
        'alpha' or 'yahoo'. The default is 'norgate'.

    Returns
    -------
    Results
        Prints out performance data for the strategy and plots performance
        graph.

    """

    def __init__(self, **kwargs):

        # Import dictionary of default parameters
        self.default_dict = copy.deepcopy(system_params_dict)

        # Define storage containers
        self.params = {}
        self.tables = {}
        self.labels = {}
        self.inputs = {}

        # Longnames for Norgate Tickers
        self.norgate_name_dict = Markets.norgate_name_dict()

        # Carry out the backtest, print out results report and graph
        # performance
        self.backtest_report_graph(**kwargs)


    def backtest_report_graph(self, **kwargs):
        """
        Generate backtest and print results report and performance graph.

        Parameters
        ----------
        params : Dict
            Dictionary of parameters.
        signals : Bool
            Whether to plot the Buy/Sell signals on the price chart.

        Returns
        -------
        Updates params, tables and labels.

        """

        # Generate backtest
        self.run_backtest(**kwargs)

        # Print out results report
        self.performance_report()

        # Graph performance
        self.performance_graph(signals=self.params['signals'])


    def run_backtest(self, **kwargs):
        """
        Generate strategy backtest

        Parameters
        ----------
        params : Dict
            Dictionary of parameters.

        Returns
        -------
        params : Dict
            Dictionary of parameters.
        tables : Dict
            Dictionary of tables.

        """

        # Store initial inputs
        for key, value in kwargs.items():
            self.inputs[key] = value

        # Initialise system parameters
        params = self._init_params(self.inputs)

        # Set the start and end dates if not provided
        params['start_date'], params['end_date'] = Dates.date_set(
            start_date=params['start_date'], end_date=params['end_date'],
            lookback=params['lookback'])

        # Create DataFrame of OHLC prices from NorgateData or Yahoo Finance
        tables = {}

        tables['prices'], params = Markets.create_base_data(
            ticker=params['ticker'], source=params['ticker_source'],
            params=params, bench_flag=False)

        # Extract benchmark data for Beta calculation
        tables['benchmark'], params = Markets.create_base_data(
            ticker=params['bench_ticker'], source=params['bench_source'],
            params=params, bench_flag=True)

        # Set the strategy labels
        labels  = {}
        labels['entry_label'], labels['exit_label'], \
            labels['stop_label'] = Labels.strategy_labels(
                params=params, default_dict=self.default_dict)

        # Generate initial trade data
        tables, params, raw_trade_price_dict = Signals.raw_entry_signals(
            tables=tables, params=params)

        # Create exit and stop targets
        tables['prices'] = TradeTargets.exit_and_stop_targets(
            prices=tables['prices'], params=params,
            trade_price_dict=raw_trade_price_dict)

        # Create exit and stop signals
        tables['prices'] = Signals.exit_and_stop_signals(
            prices=tables['prices'], params=params)

        # Prepare final signals
        tables = Signals.final_signals(params, tables)

        # Create trade and position data
        pos_dict = Positions.calc_positions(
            prices=tables['prices'],
            signal=tables['prices']['combined_signal'],
            start=params['start'])

        # Scale the position info by the position size
        pos_dict = Reformat.position_scale(
            pos_dict=pos_dict, position_size=tables['prices']['position_size'])

        # Map the raw positions to the OHLC data
        tables['prices'] = Reformat.map_to_prices(
            prices=tables['prices'],
            input_dict=pos_dict,
            title_modifier='')

        tables['prices']['trade_number'] = Trades.trade_numbers(
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

        self.params = params
        self.tables = tables
        self.labels = labels


    @staticmethod
    def _init_params(inputs):
        """
        Initialise parameter dictionary

        Parameters
        ----------
        inputs : Dict
            Dictionary of parameters supplied to the function.

        Returns
        -------
        params : Dict
            Dictionary of parameters.

        """
        # Copy the default parameters
        params = copy.deepcopy(system_params_dict['df_params'])

        # Extract the entry, exit and stop signal dictionaries
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

                if params['exit_type'] is not None:
                    # If the parameter takes a specific value for the
                    # particular exit type then replace the default with this
                    # value
                    if param in exit_signal_dict[types[1]].keys():
                        params[param] = exit_signal_dict[types[1]][str(param)]

                if params['stop_type'] is not None:
                    # If the parameter takes a specific value for the
                    # particular stop type then replace the default with this
                    # value
                    if param in stop_signal_dict[types[2]].keys():
                        params[param] = stop_signal_dict[types[2]][str(param)]

        return params


    def performance_report(self):
        """
        Display the performance report

        Parameters
        ----------
        tables : Dict
            Dictionary containing performance dict.

        Returns
        -------
        Prints the performance report to the console.

        """
        # Print out results
        PerfReport.report_table(input_dict=self.tables['perf_dict'])


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
            signals = self.params['signals']

        # Dictionary to store entry signal data
        es_dict = {}

        # Entry labels
        es_dict['entry_signal_labels'] = self.default_dict[
            'df_entry_signal_labels']

        # Entry signal indicator column names
        es_dict['entry_signal_indicators'] = self.default_dict[
            'df_entry_signal_indicators']

        # For these entry strategies just plot the price chart and equity curve
        if self.params['entry_type'] in [
                '2ma', '3ma', '4ma', 'sar', 'channel_breakout']:
            graph = perfgraph.two_panel_graph(
                signals=signals, tables=self.tables, params=self.params,
                es_dict=es_dict)

        # Otherwise also plot the relevant indicator
        else:
            graph = perfgraph.three_panel_graph(
                signals=signals, tables=self.tables, params=self.params,
                es_dict=es_dict)

        return graph
