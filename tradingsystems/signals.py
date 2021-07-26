"""
Entry and Exit signals

"""

from tradingsystems.dollar_exits import DollarExit
from tradingsystems.indicator_entries import IndicatorEntry
from tradingsystems.indicator_exits import IndicatorExit
from tradingsystems.ma_entries import MovingAverageEntry

class Entry():
    """
    Calculate entry signals

    """

    @classmethod
    def entry_signal(
            cls, prices=None, entry_type=None, ma1=None, ma2=None, ma3=None,
            ma4=None, simple_ma=None, entry_period=None, entry_oversold=None,
            entry_overbought=None, entry_threshold=None,
            entry_acceleration_factor=None):
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
        if entry_type == '2ma':
            prices, start, \
                signal = MovingAverageEntry.entry_double_ma_crossover(
                    prices=prices, ma1=ma1, ma2=ma2, simple_ma=simple_ma)

        # Triple Moving Average Crossover
        elif entry_type == '3ma':
            prices, start, \
                signal = MovingAverageEntry.entry_triple_ma_crossover(
                    prices, ma1=ma1, ma2=ma2, ma3=ma3, simple_ma=simple_ma)

        # Quad Moving Average Crossover
        elif entry_type == '4ma':
            prices, start, \
                signal = MovingAverageEntry.entry_quad_ma_crossover(
                    prices, ma1=ma1, ma2=ma2, ma3=ma3, ma4=ma4,
                    simple_ma=simple_ma)

        # Parabolic SAR
        elif entry_type == 'sar':
            prices, start, \
                signal = IndicatorEntry.entry_parabolic_sar(
                    prices=prices,
                    acceleration_factor=entry_acceleration_factor)

        # Channel Breakout
        elif entry_type == 'channel_breakout':
            prices, start, \
                signal = IndicatorEntry.entry_channel_breakout(
                    prices, time_period=entry_period)

        # Stochastic Crossover
        elif entry_type == 'stoch_cross':
            prices, start, \
                signal = IndicatorEntry.entry_stochastic_crossover(
                    prices, time_period=entry_period, oversold=entry_oversold,
                    overbought=entry_overbought)

        # Stochastic Over Under
        elif entry_type == 'stoch_over_under':
            prices, start, \
                signal = IndicatorEntry.entry_stochastic_over_under(
                    prices, time_period=entry_period, oversold=entry_oversold,
                    overbought=entry_overbought)

        # Stochastic Pop
        elif entry_type == 'stoch_pop':
            prices, start, \
                signal = IndicatorEntry.entry_stochastic_pop(
                    prices, time_period=entry_period, oversold=entry_oversold,
                    overbought=entry_overbought)

        # Relative Strength Index
        elif entry_type == 'rsi':
            prices, start, \
                signal = IndicatorEntry.entry_rsi(
                    prices, time_period=entry_period, oversold=entry_oversold,
                    overbought=entry_overbought)

        # Commodity Channel Index
        elif entry_type == 'cci':
            prices, start, \
                signal = IndicatorEntry.entry_commodity_channel_index(
                    prices, time_period=entry_period,
                    threshold=entry_threshold)

        # Momentum
        elif entry_type == 'momentum':
            prices, start, \
                signal = IndicatorEntry.entry_momentum(
                    prices, time_period=entry_period,
                    threshold=entry_threshold)

        # Volatility
        elif entry_type == 'volatility':
            prices, start, \
                signal = IndicatorEntry.entry_volatility(
                    prices, time_period=entry_period,
                    threshold=entry_threshold)

        return prices, start, signal


class Exit():
    """
    Calculate Exit and Stop signals

    """

    @classmethod
    def exit_and_stop_signals(
            cls, prices, trade_number, end_of_day_position,
            sip_price=None ,exit_type=None, exit_period=None, stop_type=None,
            stop_period=None, exit_threshold=None, exit_oversold=None,
            exit_overbought=None, exit_acceleration_factor=None,
            exit_trailing_close=None, exit_profit_target=None,
            stop_initial_dollar_loss=None, stop_profit_target=None,
            stop_trailing_close=None, stop_trailing_high_low=None):
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
        # Generate the exit signals
        prices, prices['exit_signal'] = cls._exit_signal(
            prices=prices, exit_type=exit_type, exit_period=exit_period,
            exit_threshold=exit_threshold, trade_number=trade_number,
            end_of_day_position=end_of_day_position,
            exit_oversold=exit_oversold, exit_overbought=exit_overbought,
            exit_acceleration_factor=exit_acceleration_factor,
            sip_price=sip_price, exit_trailing_close=exit_trailing_close,
            exit_profit_target=exit_profit_target)

        # Generate the stop signals
        prices, prices['stop_signal'] = cls._stop_signal(
            prices=prices, stop_type=stop_type, stop_period=stop_period,
            trade_number=trade_number,
            end_of_day_position=end_of_day_position,
            stop_initial_dollar_loss=stop_initial_dollar_loss,
            stop_profit_target=stop_profit_target,
            stop_trailing_close=stop_trailing_close,
            stop_trailing_high_low=stop_trailing_high_low)

        return prices


    @classmethod
    def _exit_signal(cls, prices, exit_type=None, exit_period=None,
                     exit_threshold=None, trade_number=None,
                     end_of_day_position=None, exit_oversold=None,
                     exit_overbought=None, exit_acceleration_factor=None,
                     sip_price=None, exit_trailing_close=None,
                     exit_profit_target=None):
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
        if exit_type == 'sar':
            prices, exit_ = IndicatorExit.exit_parabolic_sar(
                prices=prices, trade_number=trade_number,
                end_of_day_position=end_of_day_position,
                time_period=exit_period,
                acceleration_factor=exit_acceleration_factor,
                sip_price=sip_price)

        # Support / Resistance Exit
        elif exit_type == 'sup_res':
            prices, exit_ = IndicatorExit.exit_support_resistance(
                prices=prices, trade_number=trade_number,
                end_of_day_position=end_of_day_position,
                time_period=exit_period)

        # Trailing RSI Exit
        elif exit_type == 'rsi_trail':
            prices, exit_ = IndicatorExit.exit_rsi_trail(
                prices=prices, trade_number=trade_number,
                end_of_day_position=end_of_day_position,
                time_period=exit_period, oversold=exit_oversold,
                overbought=exit_overbought)

        # Key Reversal Day Exit
        elif exit_type == 'key_reversal':
            prices, exit_ = IndicatorExit.exit_key_reversal(
                prices=prices, trade_number=trade_number,
                end_of_day_position=end_of_day_position,
                time_period=exit_period)

        # Volatility Breakout Exit
        elif exit_type == 'volatility':
            prices, exit_ = IndicatorExit.exit_volatility(
                prices=prices, trade_number=trade_number,
                end_of_day_position=end_of_day_position,
                time_period=exit_period, threshold=exit_threshold)

        # Stochastic Crossover Exit
        elif exit_type == 'stoch_cross':
            prices, exit_ = IndicatorExit.exit_stochastic_crossover(
                prices=prices, time_period=exit_period,
                trade_number=trade_number,
                end_of_day_position=end_of_day_position)

        # N-day Range Exit
        elif exit_type == 'nday_range':
            prices, exit_ = IndicatorExit.exit_nday_range(
                prices=prices, trade_number=trade_number,
                end_of_day_position=end_of_day_position,
                time_period=exit_period)

        # Random Exit
        elif exit_type == 'random':
            prices, exit_ = IndicatorExit.exit_random(
                prices=prices, trade_number=trade_number,
                end_of_day_position=end_of_day_position)

        # Trailing Stop Exit
        elif exit_type == 'trailing_stop':
            prices, exit_ = DollarExit.exit_dollar(
                prices=prices, trigger_value=exit_trailing_close,
                exit_level='trail_close', trade_number=trade_number,
                end_of_day_position=end_of_day_position)

        # Profit Target Exit
        elif exit_type == 'profit_target':
            prices, exit_ = DollarExit.exit_dollar(
                prices=prices, trigger_value=exit_profit_target,
                exit_level='profit_target', trade_number=trade_number,
                end_of_day_position=end_of_day_position)

        return prices, exit_


    @classmethod
    def _stop_signal(
            cls, prices, stop_type=None, stop_period=None, trade_number=None,
            end_of_day_position=None, stop_initial_dollar_loss=None,
            stop_profit_target=None, stop_trailing_close=None,
            stop_trailing_high_low=None):
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
        if stop_type == 'sup_res':
            prices, stop = IndicatorExit.exit_support_resistance(
                prices=prices, trade_number=trade_number,
                end_of_day_position=end_of_day_position,
                time_period=stop_period)

        # Immediate Profit Stop
        elif stop_type == 'immediate_profit':
            prices, stop = IndicatorExit.exit_immediate_profit(
                prices=prices, trade_number=trade_number,
                end_of_day_position=end_of_day_position,
                time_period=stop_period)

        # Initial Dollar Loss Stop
        elif stop_type == 'initial_dollar':
            prices, stop = DollarExit.exit_dollar(
                prices=prices, trigger_value=stop_initial_dollar_loss,
                exit_level='initial', trade_number=trade_number,
                end_of_day_position=end_of_day_position)

        # Breakeven Stop
        elif stop_type == 'breakeven':
            prices, stop = DollarExit.exit_dollar(
                prices=prices, trigger_value=stop_profit_target,
                exit_level='breakeven', trade_number=trade_number,
                end_of_day_position=end_of_day_position,
                trade_high_price=prices['raw_td_high_price'],
                trade_low_price=prices['raw_td_low_price'])

        # Trailing Stop (Closing Price)
        elif stop_type == 'trail_close':
            prices, stop = DollarExit.exit_dollar(
                prices=prices, trigger_value=stop_trailing_close,
                exit_level='trail_close', trade_number=trade_number,
                end_of_day_position=end_of_day_position)

        # Trailing Stop (High / Low Price)
        elif stop_type == 'trail_high_low':
            prices, stop = DollarExit.exit_dollar(
                prices=prices, trigger_value=stop_trailing_high_low,
                exit_level='trail_high_low', trade_number=trade_number,
                end_of_day_position=end_of_day_position)

        return prices, stop
