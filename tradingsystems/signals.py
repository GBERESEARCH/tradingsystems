"""
Entry and Exit signals

"""
import numpy as np
import pandas as pd
from dollar_exits import DollarExit
from indicator_entries import IndicatorEntry
from indicator_exits import IndicatorExit
from ma_entries import MovingAverageEntry
from positions import Positions
from trades import Trades
from utils import Labels, Reformat

class Signals():
    """
    Calculate entry signals

    """

    @classmethod
    def raw_entry_signals(cls, tables, params):
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
        tables['prices'], params['start'], \
            tables['prices']['raw_trade_signal'] = cls.entry_signal(
                tables=tables, params=params)

        # Calculate initial position info
        raw_pos_dict = Positions.calc_positions(
            prices=tables['prices'],
            signal=tables['prices']['raw_trade_signal'],
            start=params['start'])

        # Generate trade numbers
        tables['prices']['raw_trade_number'] = Trades.trade_numbers(
            prices=tables['prices'],
            end_of_day_position=raw_pos_dict['end_of_day_position'],
            start=params['start'])

        # Set the position size
        tables['prices'], tables['benchmark'], \
            params = Positions.position_size(
                prices=tables['prices'], benchmark=tables['benchmark'],
                params=params)

        # Set the position size label
        params = Labels.position_size_label(params)

        # Scale the position info by the position size
        scaled_pos_dict = Reformat.position_scale(
            pos_dict=raw_pos_dict,
            position_size=tables['prices']['position_size'])

        # Generate initial trade prices
        raw_trade_price_dict = Trades.trade_prices(
            prices=tables['prices'],
            trade_number=tables['prices']['raw_trade_number'])

        # Map the raw positions to the OHLC data
        tables['prices'] = Reformat.map_to_prices(
            prices=tables['prices'],
            input_dict=scaled_pos_dict,
            title_modifier='raw_')

        # Map the raw trade prices to the OHLC data
        tables['prices'] = Reformat.map_to_prices(
            prices=tables['prices'],
            input_dict=raw_trade_price_dict,
            title_modifier='raw_')

        return tables, params, raw_trade_price_dict


    @classmethod
    def entry_signal(cls, tables, params):
        """
        Calculate trade entry signals

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
        simple_ma : Bool
            Whether to calculate a simple or exponential moving average. The
            default is True.
        entry_period : Int
            The number of days to use in the entry strategy. The default is 14.
        entry_oversold : Int
            The oversold level to use in the entry strategy.
        entry_overbought : Int
            The overbought level to use in the entry strategy.
        entry_threshold : Float
            The entry threshold used for momentum / volatility strategies.
            The default is 0 for momentum and 1.5 for volatility.
        entry_acceleration_factor : Float
            The acceleration factor used in the Parabolic SAR entry signal.
            The default is 0.02.

        Returns
        -------
        prices : DataFrame
            The OHLC data.
        start : Int
            The first valid row to start calculating trade information from.
        signal : Series
            The series of Buy / Sell signals.

        """

        # Double Moving Average Crossover
        if params['entry_type'] == '2ma':
            tables['prices'], start, \
                signal = MovingAverageEntry.entry_double_ma_crossover(
                    prices=tables['prices'],
                    params=params)

        # Triple Moving Average Crossover
        elif params['entry_type'] == '3ma':
            tables['prices'], start, \
                signal = MovingAverageEntry.entry_triple_ma_crossover(
                    prices=tables['prices'],
                    params=params)

        # Quad Moving Average Crossover
        elif params['entry_type'] == '4ma':
            tables['prices'], start, \
                signal = MovingAverageEntry.entry_quad_ma_crossover(
                    prices=tables['prices'],
                    params=params)

        # Parabolic SAR
        elif params['entry_type'] == 'sar':
            tables['prices'], start, \
                signal = IndicatorEntry.entry_parabolic_sar(
                    prices=tables['prices'],
                    acceleration_factor=params['entry_acceleration_factor'])

        # Channel Breakout
        elif params['entry_type'] == 'channel_breakout':
            tables['prices'], start, \
                signal = IndicatorEntry.entry_channel_breakout(
                    prices=tables['prices'],
                    time_period=params['entry_period'])

        # Stochastic Crossover
        elif params['entry_type'] == 'stoch_cross':
            tables['prices'], start, \
                signal = IndicatorEntry.entry_stochastic_crossover(
                    tables['prices'],
                    time_period=params['entry_period'],
                    oversold=params['entry_oversold'],
                    overbought=params['entry_overbought'])

        # Stochastic Over Under
        elif params['entry_type'] == 'stoch_over_under':
            tables['prices'], start, \
                signal = IndicatorEntry.entry_stochastic_over_under(
                    tables['prices'],
                    time_period=params['entry_period'],
                    oversold=params['entry_oversold'],
                    overbought=params['entry_overbought'])

        # Stochastic Pop
        elif params['entry_type'] == 'stoch_pop':
            tables['prices'], start, \
                signal = IndicatorEntry.entry_stochastic_pop(
                    tables['prices'],
                    time_period=params['entry_period'],
                    oversold=params['entry_oversold'],
                    overbought=params['entry_overbought'])

        # Relative Strength Index
        elif params['entry_type'] == 'rsi':
            tables['prices'], start, \
                signal = IndicatorEntry.entry_rsi(
                    tables['prices'],
                    time_period=params['entry_period'],
                    oversold=params['entry_oversold'],
                    overbought=params['entry_overbought'])

        # ADX
        elif params['entry_type'] == 'adx':
            tables['prices'], start, \
                signal = IndicatorEntry.entry_adx(
                    tables['prices'],
                    time_period=params['entry_period'],
                    threshold=params['adx_threshold'])

        # Commodity Channel Index
        elif params['entry_type'] == 'cci':
            tables['prices'], start, \
                signal = IndicatorEntry.entry_commodity_channel_index(
                    tables['prices'],
                    time_period=params['entry_period'],
                    threshold=params['entry_threshold'])

        # Momentum
        elif params['entry_type'] == 'momentum':
            tables['prices'], start, \
                signal = IndicatorEntry.entry_momentum(
                    tables['prices'],
                    time_period=params['entry_period'],
                    threshold=params['entry_threshold'])

        # Volatility
        elif params['entry_type'] == 'volatility':
            tables['prices'], start, \
                signal = IndicatorEntry.entry_volatility(
                    tables['prices'],
                    time_period=params['entry_period'],
                    threshold=params['entry_threshold'])

        return tables['prices'], start, signal


    @classmethod
    def exit_and_stop_signals(cls, prices, params):
        """
        Calculate trade exit and stop signals.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        sip_price : Bool
            Whether to set the SIP of the Parabolic SAR exit to n-day
            high / low or to the high of the previous trade. The default is
            False.
        exit_type : Str, optional
            The exit strategy. The default is 'trailing_stop'.
        exit_period : Int, optional
            The number of days to use in the exit strategy. The default is 5.
        stop_type : Str, optional
            The stop strategy. The default is 'initial_dollar'.
        stop_period : Int, optional
            The number of days to use in the stop strategy. The default is 5.
        exit_threshold : Float
            The exit threshold used for the volatility strategy.
            The default is 1.
        exit_oversold : Int, optional
            The oversold level to use in the exit strategy.
        exit_overbought : Int, optional
            The overbought level to use in the exit strategy.
        exit_acceleration_factor : Float
            The acceleration factor used in the Parabolic SAR exit signal.
            The default is 0.02.
        exit_trailing_close : Series, optional
            The exit levels for each trade based on the trailing close
        exit_profit_target : Series, optional
            The exit levels for each trade based on a profit target
        stop_initial_dollar_loss : Series, optional
            The stop levels for each trade based on a dollar loss from the
            entry level.
        stop_profit_target : Series, optional
            The stop levels for each trade based on a profit target
        stop_trailing_close : Series, optional
            The stop levels for each trade based on the trailing close
        stop_trailing_high_low : Series, optional
            The stop levels for each trade based on the trailing high / low.

        Returns
        -------
        prices : DataFrame
            The OHLC data

        """
        if params['exit_type'] is not None:
            # Generate the exit signals
            prices, prices['exit_signal'] = cls._exit_signal(
                prices=prices, params=params)
        else:
            prices['exit_signal'] = np.array([0]*len(prices))

        if params['stop_type'] is not None:
            # Generate the stop signals
            prices, prices['stop_signal'] = cls._stop_signal(
                prices=prices, params=params)
        else:
            prices['stop_signal'] = np.array([0]*len(prices))

        return prices


    @classmethod
    def _exit_signal(cls, prices, params):
        """
        Calculate trade exit signals.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        exit_type : Str, optional
            The exit strategy. The default is 'trailing_stop'.
        exit_period : Int, optional
            The number of days to use in the exit strategy. The default is 5.
        exit_threshold : Float
            The exit threshold used for the volatility strategy.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        exit_oversold : Int, optional
            The oversold level to use in the exit strategy.
        exit_overbought : Int, optional
            The overbought level to use in the exit strategy.
        exit_acceleration_factor : Float
            The acceleration factor used in the Parabolic SAR exit signal.
            The default is 0.02.
        sip_price : Bool
            Whether to set the SIP of the Parabolic SAR exit to n-day
            high / low or to the high of the previous trade. The default is
            False.

        Returns
        -------
        prices : DataFrame
            The OHLC data
        exit : Series
            The exit signals.


        prices : DataFrame
            The OHLC data

        """

        # Parabolic SAR Exit
        if params['exit_type'] == 'sar':
            prices, exit_ = IndicatorExit.exit_parabolic_sar(
                prices=prices,
                time_period=params['exit_period'],
                acceleration_factor=params['exit_acceleration_factor'],
                sip_price=params['sip_price'])

        # Support / Resistance Exit
        elif params['exit_type'] == 'sup_res':
            prices, exit_ = IndicatorExit.exit_support_resistance(
                prices=prices,
                time_period=params['exit_period'])

        # Trailing RSI Exit
        elif params['exit_type'] == 'rsi_trail':
            prices, exit_ = IndicatorExit.exit_rsi_trail(
                prices=prices,
                time_period=params['exit_period'],
                oversold=params['exit_oversold'],
                overbought=params['exit_overbought'])

        # Key Reversal Day Exit
        elif params['exit_type'] == 'key_reversal':
            prices, exit_ = IndicatorExit.exit_key_reversal(
                prices=prices,
                time_period=params['exit_period'])

        # Volatility Breakout Exit
        elif params['exit_type'] == 'volatility':
            prices, exit_ = IndicatorExit.exit_volatility(
                prices=prices,
                time_period=params['exit_period'],
                threshold=params['exit_threshold'])

        # Stochastic Crossover Exit
        elif params['exit_type'] == 'stoch_cross':
            prices, exit_ = IndicatorExit.exit_stochastic_crossover(
                prices=prices,
                time_period=params['exit_period'])

        # N-day Range Exit
        elif params['exit_type'] == 'nday_range':
            prices, exit_ = IndicatorExit.exit_nday_range(
                prices=prices,
                time_period=params['exit_period'])

        # Random Exit
        elif params['exit_type'] == 'random':
            prices, exit_ = IndicatorExit.exit_random(
                prices=prices)

        # Trailing Stop Exit
        elif params['exit_type'] == 'trailing_stop':
            prices, exit_ = DollarExit.exit_dollar(
                exit_level='trail_close',
                prices=prices,
                trigger_value=prices['exit_trailing_close'])

        # Profit Target Exit
        elif params['exit_type'] == 'profit_target':
            prices, exit_ = DollarExit.exit_dollar(
                exit_level='profit_target',
                prices=prices,
                trigger_value=prices['exit_profit_target'])

        return prices, exit_


    @classmethod
    def _stop_signal(cls, prices, params):
        """
        Calculate trade stop signals

        Parameters
        ----------
        prices : DataFrame
            The OHLC data
        stop_type : Str
            The type of stop to use.
        stop_period : Int
            The length of time for the indicator calculation.
        trade_number : Series
            Array of trade numbers.
        end_of_day_position : Series
            The close of day, long/short/flat position.
        stop_initial_dollar_loss : Series, optional
            The stop levels for each trade based on a dollar loss from the
            entry level.
        stop_profit_target : Series, optional
            The stop levels for each trade based on a profit target
        stop_trailing_close : Series, optional
            The stop levels for each trade based on the trailing close
        stop_trailing_high_low : Series, optional
            The stop levels for each trade based on the trailing high / low.

        Returns
        -------
        prices : DataFrame
            The OHLC data
        stop : Series
            The stop signals.

        """

        # Support / Resistance Stop
        if params['stop_type'] == 'sup_res':
            prices, stop = IndicatorExit.exit_support_resistance(
                prices=prices,
                time_period=params['stop_period'])

        # Immediate Profit Stop
        elif params['stop_type'] == 'immediate_profit':
            prices, stop = IndicatorExit.exit_immediate_profit(
                prices=prices,
                time_period=params['stop_period'])

        # Initial Dollar Loss Stop
        elif params['stop_type'] == 'initial_dollar':
            prices, stop = DollarExit.exit_dollar(
                exit_level='initial',
                prices=prices,
                trigger_value=prices['stop_initial_dollar_loss'])

        # Breakeven Stop
        elif params['stop_type'] == 'breakeven':
            prices, stop = DollarExit.exit_dollar(
                exit_level='breakeven',
                prices=prices,
                trigger_value=prices['stop_profit_target'])

        # Trailing Stop (Closing Price)
        elif params['stop_type'] == 'trail_close':
            prices, stop = DollarExit.exit_dollar(
                exit_level='trail_close',
                prices=prices,
                trigger_value=prices['stop_trailing_close'])

        # Trailing Stop (High / Low Price)
        elif params['stop_type'] == 'trail_high_low':
            prices, stop = DollarExit.exit_dollar(
                exit_level='trail_high_low',
                prices=prices,
                trigger_value=prices['stop_trailing_high_low'])

        return prices, stop


    @staticmethod
    def final_signals(params, tables):
        """
        Concatenate entry, exit and stop signals into a combined signal.

        Parameters
        ----------
        prices : DataFrame
            The OHLC data.
        start : Int
            The first valid row to start calculating trade information from.

        Returns
        -------
        prices : DataFrame
            The OHLC data.

        """
        # Concatenate the Entry, Exit and Stop signals in a single DataFrame
        trade_signals = pd.concat(
            [tables['prices']['raw_trade_signal'],
             tables['prices']['exit_signal'],
             tables['prices']['stop_signal']], axis=1)

        # Generate single combined trade signal
        tables['prices']['combined_signal'] = Trades.signal_combine(
            prices=tables['prices'], start=params['start'],
            end_of_day_position=tables['prices']['raw_end_of_day_position'],
            trade_signals=trade_signals)

        return tables
