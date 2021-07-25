# Imports
from tradingsystems.graphs import PerformanceGraph as perfgraph
from tradingsystems.marketdata import Markets
from operator import itemgetter
from tradingsystems.pnl import Profit
from tradingsystems.reports import PerfReport
from tradingsystems.signals import Entry, Exit
from tradingsystems.trades import Positions
from tradingsystems.systems_params import system_params_dict
from tradingsystems.utils import Labels, Dates


class Data():
    
    def __init__(self):
        
        # Import dictionary of default parameters 
        self.df_dict = system_params_dict
        
        # Longnames for Norgate Tickers
        self.norgate_name_dict = Markets._norgate_name_dict()
        
        # Entry, Exit and Stop labels
        self.entry_signal_labels = self.df_dict['df_entry_signal_labels']
        self.exit_signal_labels = self.df_dict['df_exit_signal_labels']
        self.stop_signal_labels = self.df_dict['df_stop_signal_labels']
        
        # Entry, Exit and Stop dictionaries of defaults
        self.entry_signal_dict = self.df_dict['df_entry_signal_dict']
        self.exit_signal_dict = self.df_dict['df_exit_signal_dict']
        self.stop_signal_dict = self.df_dict['df_stop_signal_dict']   
        
        # Entry signal indicator column names
        self.entry_signal_indicators = self.df_dict[
            'df_entry_signal_indicators']
  
    
    def _refresh_params_default(self, **kwargs):
        """
        Set parameters in various functions to the default values.
        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.
        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults if 
            no data provided
        """
        
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Set it to the default value and assign to the object 
                # and to input dictionary
                v = self.df_dict['df_'+str(k)]
                self.__dict__[k] = v
                kwargs[k] = v 
            
            # If the value has been provided as an input, assign this 
            # to the object
            else:
                self.__dict__[k] = v
                      
        return kwargs    
   

    def _refresh_signals_default(self, **kwargs):
        """
        Set parameters for use in various pricing functions
        Parameters
        ----------
        **kwargs : Various
                   Takes any of the arguments of the various methods 
                   that use it to refresh data.
        Returns
        -------
        Various
            Runs methods to fix input parameters and reset defaults 
            if no data provided
        """
 
        # For all the supplied arguments
        for k, v in kwargs.items():
            
            # If a value for a parameter has not been provided
            if v is None:
                
                # Check if the key is in the entry signal dict 
                try:                    
                    # Extract these from the df_combo_dict
                    v = self.entry_signal_dict[str(
                        self.entry_type)][str(k)]
                    
                    # Assign to input dictionary
                    kwargs[k] = v

                except:
                    pass
                        
                # Check if the key is in the exit signal dict    
                try:                    
                    # Extract these from the exit signal dict
                    v = self.exit_signal_dict[str(
                        self.exit_type)][str(k)]
                    
                    # Assign to input dictionary
                    kwargs[k] = v

                except:
                    pass    
                    
                # Check if the key is in the entry signal dict                
                try:                    
                    # Extract these from the df_combo_dict
                    v = self.stop_signal_dict[str(
                        self.stop_type)][str(k)]
                    
                    # Assign to input dictionary
                    kwargs[k] = v

                except:
                   
                    if v is None:
                        # Otherwise set to the standard default 
                        # value
                        v = self.df_dict['df_'+str(k)]
                        kwargs[k] = v
            
                # Now assign this to the object and input dictionary
                self.__dict__[k] = v
            
            # If the parameter has been provided as an input, 
            # assign this to the object
            else:
                self.__dict__[k] = v
                               
        return kwargs            
    
    
    def test_strategy(
            self, ticker=None, ccy_1=None, ccy_2=None, asset_type=None, 
            start_date=None, end_date=None, lookback=None, 
            ma1=None, ma2=None, ma3=None, ma4=None, simple_ma=None, 
            position_size=None, pos_size_fixed=None, ticker_source=None, 
            bench_ticker=None, bench_source=None, slippage=None, 
            commission=None, riskfree=None, strategy=None, entry_type=None, 
            exit_type=None, stop_type=None, entry_period=None, 
            exit_period=None, stop_period=None, entry_oversold=None, 
            entry_overbought=None, exit_oversold=None, exit_overbought=None, 
            entry_threshold=None, exit_threshold=None,
            entry_acceleration_factor=None, exit_acceleration_factor=None, 
            sip_price=None, equity=None, exit_amount=None, stop_amount=None,
            api_key=None):
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
        
        # Set Entry, Exit and Stop types
        (entry_type, exit_type, stop_type) = itemgetter(
            'entry_type', 'exit_type', 
            'stop_type')(self._refresh_params_default(
                entry_type=entry_type, exit_type=exit_type, 
                stop_type=stop_type))
                
        # Basic parameters
        # If data is not supplied as an input, take default values 
        (ticker, ccy_1, ccy_2, asset_type, lookback, simple_ma, position_size, 
         pos_size_fixed, ticker_source, bench_ticker, bench_source, slippage, 
         commission, riskfree, equity) = itemgetter(
             'ticker', 'ccy_1', 'ccy_2', 'asset_type', 'lookback', 'simple_ma', 
             'position_size', 'pos_size_fixed', 'ticker_source', 
             'bench_ticker', 'bench_source', 'slippage', 'commission', 
             'riskfree', 'equity')(self._refresh_params_default(
                 ticker=ticker, ccy_1=ccy_1, ccy_2=ccy_2, 
                 asset_type=asset_type, lookback=lookback, simple_ma=simple_ma, 
                 position_size=position_size, pos_size_fixed=pos_size_fixed, 
                 ticker_source=ticker_source, bench_ticker=bench_ticker, 
                 bench_source=bench_source, slippage=slippage, 
                 commission=commission, riskfree=riskfree, equity=equity))
                
        # Strategy specific parameters         
        # If data is not supplied as an input, take default values 
        (ma1, ma2, ma3, ma4, entry_period, exit_period, stop_period,
         entry_oversold, entry_overbought, exit_oversold,
         exit_overbought, entry_threshold, exit_threshold,
         entry_acceleration_factor, exit_acceleration_factor, sip_price, 
         exit_amount, stop_amount) = itemgetter(
             'ma1', 'ma2', 'ma3', 'ma4', 'entry_period', 'exit_period', 
             'stop_period', 'entry_oversold', 'entry_overbought', 
             'exit_oversold', 'exit_overbought', 'entry_threshold', 
             'exit_threshold', 'entry_acceleration_factor', 
             'exit_acceleration_factor', 'sip_price', 'exit_amount', \
                 'stop_amount')(self._refresh_signals_default(
                 ma1=ma1, ma2=ma2, ma3=ma3, ma4=ma4, entry_period=entry_period, 
                 exit_period=exit_period, stop_period=stop_period, 
                 entry_oversold=entry_oversold, 
                 entry_overbought=entry_overbought, 
                 exit_oversold=exit_oversold, 
                 exit_overbought=exit_overbought, 
                 entry_threshold=entry_threshold, 
                 exit_threshold=exit_threshold, 
                 entry_acceleration_factor=entry_acceleration_factor, 
                 exit_acceleration_factor=exit_acceleration_factor,
                 sip_price=sip_price, 
                 exit_amount=exit_amount, stop_amount=stop_amount))        
                             
        # Set the start and end dates if not provided
        self.start_date, self.end_date = Dates._date_set(
            start_date=start_date, end_date=end_date, lookback=self.lookback)
        
        # Create DataFrame of OHLC prices from NorgateData or Yahoo Finance
        df = Markets.create_base_data(
            ticker=self.ticker, ccy_1=self.ccy_1, ccy_2=self.ccy_2, 
            start_date=self.start_date, end_date=self.end_date, 
            source=self.ticker_source, asset_type=self.asset_type, 
            api_key=api_key)

        # Extract benchmark data for Beta calculation
        self.benchmark = Markets.create_base_data(
            ticker=self.bench_ticker, start_date=self.start_date, 
            end_date=self.end_date, source=self.bench_source, 
            asset_type=self.asset_type, api_key=api_key)
 
        # Set the strategy labels
        self.entry_label, self.exit_label, \
            self.stop_label = Labels._strategy_labels(
            df=df, ma1=self.ma1, ma2=self.ma2, ma3=self.ma3, ma4=self.ma4, 
            entry_period=self.entry_period, exit_period=self.exit_period, 
            stop_period=self.stop_period, entry_oversold=self.entry_oversold, 
            exit_oversold=self.exit_oversold, 
            entry_overbought=self.entry_overbought, 
            exit_overbought=self.exit_overbought, 
            entry_threshold=self.entry_threshold, 
            exit_threshold=self.exit_threshold, simple_ma=self.simple_ma, 
            entry_type=self.entry_type, exit_type=self.exit_type, 
            stop_type=self.stop_type, exit_amount=self.exit_amount,
            stop_amount=self.stop_amount, 
            entry_acceleration_factor=self.entry_acceleration_factor, 
            exit_acceleration_factor=self.exit_acceleration_factor,
            sip_price=self.sip_price, df_dict=self.df_dict)
        
        # Generate initial trade data
        df, self.start, self.position_size, \
            self.benchmark_position_size = self._raw_entry_signals(
                df=df, entry_type=self.entry_type, ma1=self.ma1, ma2=self.ma2, 
                ma3=self.ma3, ma4=self.ma4, entry_period=self.entry_period, 
                entry_oversold=self.entry_oversold, 
                entry_overbought=self.entry_overbought, 
                entry_threshold=self.entry_threshold, equity=self.equity,
                simple_ma=self.simple_ma, benchmark=self.benchmark,
                entry_acceleration_factor=self.entry_acceleration_factor)
        
        # Create exit and stop targets
        df = self._exit_targets(df=df, exit_amount=self.exit_amount, 
                                position_size=self.position_size)
        
        df = self._stop_targets(df=df, stop_amount=self.stop_amount, 
                                position_size=self.position_size)
        
        # Create exit and stop signals
        df = Exit._exit_and_stop_signals(
            df=df, position_size=position_size, exit_type=self.exit_type, 
            stop_type=self.stop_type, exit_amount=self.exit_amount, 
            exit_period=self.exit_period, stop_amount=self.stop_amount, 
            stop_period=self.stop_period, exit_threshold=self.exit_threshold, 
            exit_oversold=self.exit_oversold, 
            exit_overbought=self.exit_overbought,
            exit_acceleration_factor=self.exit_acceleration_factor, 
            sip_price=self.sip_price, trade_number=df['raw_td_number'], 
            end_of_day_position=df['raw_eod_position'],
            exit_trailing_close=df['exit_trailing_close'], 
            exit_profit_target=df['exit_profit_target'], 
            stop_initial_dollar_loss=df['stop_initial_dollar_loss'], 
            stop_profit_target=df['stop_profit_target'], 
            stop_trailing_close=df['stop_trailing_close'], 
            stop_trailing_high_low=df['stop_trailing_high_low'])
       
        # Combine signals
        df['combined_signal'] = Positions._signal_combine(
            df=df, start=self.start, raw_trade_signal=df['raw_td_signal'], 
            end_of_day_position=df['raw_eod_position'], 
            trade_number=df['raw_td_number'], 
            exit_signal=df['exit_signal'], 
            stop_signal=df['stop_signal'])
        
        # Create trade and position data
        df['start_of_day_position'], df['trade_signal'], \
            df['position'] = Positions._positions_and_trade_actions(
                df=df, signal=df['combined_signal'], start=self.start, 
                position_size=self.position_size)
            
        df['trade_number'] = Positions._trade_numbers(
            df=df, end_of_day_position=df['position'], 
            start=self.start)    
                
        # Calculate the trades and pnl for the strategy
        self.df = Profit._profit_data(
            df=df, position_size=self.position_size, slippage=self.slippage, 
            commission=self.commission, equity=self.equity)
        
        # Create monthly summary data
        self.monthly_data = Profit._create_monthly_data(
            df=self.df, equity=self.equity)
        
        # Create dictionary of performance data
        self.perf_dict = PerfReport._performance_data(
            df=self.df, monthly_data=self.monthly_data, 
            ticker_source=self.ticker_source, asset_type=self.asset_type, 
            ccy_1=self.ccy_1, ccy_2=self.ccy_2, ticker=self.ticker, 
            entry_label=self.entry_label, exit_label=self.exit_label, 
            stop_label=self.stop_label,
            norgate_name_dict=self.norgate_name_dict, slippage=self.slippage, 
            commission=self.commission, position_size=self.position_size, 
            benchmark=self.benchmark, 
            benchmark_position_size=self.benchmark_position_size, 
            riskfree=self.riskfree)

        # Print out results
        PerfReport.report_table(input_dict=self.perf_dict)

        return self


    @staticmethod
    def _raw_entry_signals(
            df, entry_type, ma1, ma2, ma3, ma4, entry_period, entry_oversold, 
            entry_overbought, entry_threshold, simple_ma, benchmark,
            equity, entry_acceleration_factor):
        """
        Generate the initial raw entry signals, positions and trades

        Parameters
        ----------
        df : DataFrame
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
        df : DataFrame
            The OHLC data.
        start : Int
            The first valid row to start calculating trade information from.

        """
        # Generate entry signals
        df, start, df['raw_td_signal'] = Entry._entry_signal(
            df=df, entry_type=entry_type, ma1=ma1, ma2=ma2, ma3=ma3, ma4=ma4,
            entry_period=entry_period, entry_oversold=entry_oversold, 
            entry_overbought=entry_overbought, entry_threshold=entry_threshold, 
            simple_ma=simple_ma, 
            entry_acceleration_factor=entry_acceleration_factor)

        # Set the position size
        position_size, benchmark_position_size = Positions._position_size(
                df=df, benchmark=benchmark, equity=equity)

        # Calculate initial position info        
        df['raw_start_of_day_position'], df['raw_td_action'], \
            df['raw_eod_position'] = Positions._positions_and_trade_actions(
                df=df, signal=df['raw_td_signal'], start=start, 
                position_size=position_size)
        
        # Generate trade numbers    
        df['raw_td_number'] = Positions._trade_numbers(
            df=df, end_of_day_position=df['raw_eod_position'], 
            start=start)

        # Generate initial trade prices
        df['raw_td_entry_price'], df['raw_td_exit_price'], \
            df['raw_td_high_price'], df['raw_td_low_price'], \
                df['raw_td_close_high_price'], \
                    df['raw_td_close_low_price'] = Positions._trade_prices(
                        df=df, trade_number=df['raw_td_number'])
            
        return df, start, position_size, benchmark_position_size    


    @classmethod
    def _exit_targets(cls, df, exit_amount, position_size):
        """
        Create 4 series of exit targets

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        exit_amount : Float
            The dollar exit amount. The default is $1000.00.
        position_size : Int, optional
            The number of units to trade. The default is based on equity.

        Returns
        -------
        df : DataFrame
            The OHLC data..

        """
        # Generate profit targets / trailing stops        
        df['exit_profit_target'], df['exit_initial_dollar_loss'], \
            df['exit_trailing_close'], \
                df['exit_trailing_high_low'] = Positions._pnl_targets(
                    df=df, dollar_amount=exit_amount, 
                    position_size=position_size,
                    trade_number=df['raw_td_number'], 
                    end_of_day_position=df['raw_eod_position'], 
                    trade_entry_price=df['raw_td_entry_price'], 
                    trade_high_price=df['raw_td_high_price'], 
                    trade_close_high_price=df['raw_td_close_high_price'], 
                    trade_low_price=df['raw_td_low_price'], 
                    trade_close_low_price=df['raw_td_close_low_price']) 

        return df
    
    
    @classmethod
    def _stop_targets(cls, df, stop_amount, position_size):
        """
        Create 4 series of stop targets 

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        stop_amount : Float
            The dollar stop amount. The default is $500.00.
        position_size : Int, optional
            The number of units to trade. The default is based on equity.

        Returns
        -------
        df : DataFrame
            The OHLC data.

        """
        # Generate profit targets / trailing stops        
        df['stop_profit_target'], df['stop_initial_dollar_loss'], \
            df['stop_trailing_close'], \
                df['stop_trailing_high_low'] = Positions._pnl_targets(
                    df=df, dollar_amount=stop_amount, 
                    position_size=position_size,
                    trade_number=df['raw_td_number'], 
                    end_of_day_position=df['raw_eod_position'], 
                    trade_entry_price=df['raw_td_entry_price'], 
                    trade_high_price=df['raw_td_high_price'], 
                    trade_close_high_price=df['raw_td_close_high_price'], 
                    trade_low_price=df['raw_td_low_price'], 
                    trade_close_low_price=df['raw_td_close_low_price'])
                
        return df
 

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
            signals = self.df_dict['df_signals']
        
        # For these entry strategies just plot the price chart and equity curve
        if self.entry_type in ['2ma', '3ma', '4ma', 'sar', 'channel_breakout']:
            return perfgraph._two_panel_graph(
                signals=signals, df=self.df, entry_type=self.entry_type, 
                perf_dict=self.perf_dict, entry_period=self.entry_period,
                entry_signal_labels=self.entry_signal_labels, 
                entry_signal_indicators=self.entry_signal_indicators, 
                ma1=self.ma1, ma2=self.ma2, ma3=self.ma3, ma4=self.ma4)
        
        # Otherwise also plot the relevant indicator
        else:
            return perfgraph._three_panel_graph(
                signals=signals, df=self.df, entry_type=self.entry_type, 
                perf_dict=self.perf_dict, entry_period=self.entry_period,
                entry_signal_labels=self.entry_signal_labels, 
                entry_signal_indicators=self.entry_signal_indicators,
                entry_overbought=self.entry_overbought, 
                entry_oversold=self.entry_oversold)
        
