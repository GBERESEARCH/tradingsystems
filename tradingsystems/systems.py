"""
Calculate performance for various technical trading systems and display the
results in table and graph form.

"""

# Imports
import copy
from tradingsystems.graphs import PerformanceGraph as perfgraph
from tradingsystemsdata.signals import CalculateSignalData as calcsig
from tradingsystemsdata.systems_params import system_params_dict
from tradingsystemsdata.systems import TestStrategy as TSData


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

        if kwargs.get('return_data', False):
            # Generate backtest
            params, tables, labels, norgate_name_dict = TSData.run_backtest(**kwargs)
                

        else:
            # Carry out the backtest, print out results report and graph
            # performance
            params, tables, labels, norgate_name_dict = self.backtest_report_graph(**kwargs)

        # Generate signals when graph isn't drawn.    
        params = calcsig.generate_signals(
            default_dict=self.default_dict, 
            params=params, 
            tables=tables
            )
        
        self.params = params
        self.tables = tables
        self.labels = labels
        self.norgate_name_dict = norgate_name_dict


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
        params, tables, labels, norgate_name_dict = TSData.run_backtest(**kwargs)

        # Print out results report
        TSData.performance_report(input_dict=tables['perf_dict'])

        # Graph performance
        params = self.performance_graph(
            params=params, 
            tables=tables, 
            default_dict=self.default_dict, 
            signals=params['signals']
            )

        return params, tables, labels, norgate_name_dict


    def performance_graph(
        self, 
        params: dict,
        tables: dict,
        default_dict: dict,
        signals: bool | None = None, 
        ):
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
            signals = params['signals']

        # Dictionary to store entry signal data
        es_dict = {}

        # Entry labels
        es_dict['entry_signal_labels'] = default_dict[
            'df_entry_signal_labels']

        # Entry signal indicator column names
        es_dict['entry_signal_indicators'] = default_dict[
            'df_entry_signal_indicators']

        # For these entry strategies just plot the price chart and equity curve
        if params['entry_type'] in [
                '2ma', '3ma', '4ma', 'sar', 'channel_breakout']:
            params = perfgraph.two_panel_graph(
                signals=signals, tables=tables, params=params,
                es_dict=es_dict)

        # Otherwise also plot the relevant indicator
        else:
            params = perfgraph.three_panel_graph(
            signals=signals, tables=tables, params=params,
            es_dict=es_dict)

        params['es_dict'] = es_dict

        return params
        