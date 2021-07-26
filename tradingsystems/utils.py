"""
Utility functions

"""

import datetime as dt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

class Labels():
    """
    Create labels for the Entry, Exit and Stop Strategies.

    """

    @staticmethod
    def strategy_labels(**kwargs):
        """
        Create label and price signal for chosen strategy

        Parameters
        ----------
        **kwargs : Various
            The input parameters necessary for the chosen strategy.

        Returns
        -------
        entry_label : Str
            The longname of the entry strategy.
        exit_label : Str
            The longname of the exit strategy.
        stop_label : Str
            The longname of the stop strategy.

        """

        # Simple or Exponential Moving Average label
        if kwargs['simple_ma']:
            ma_type_label = 'S'
        else:
            ma_type_label = 'E'


        # Entry labels

        # Double Moving Average Crossover
        if kwargs['entry_type'] == '2ma':

            # Set the entry label
            entry_label = (str(kwargs['ma1'])
                           +'-, '
                           +str(kwargs['ma2'])
                           +'-day : '
                           +kwargs['default_dict']['df_entry_signal_labels'][
                               kwargs['entry_type']][0]
                           +ma_type_label
                           +kwargs['default_dict']['df_entry_signal_labels'][
                               kwargs['entry_type']][1])


        # Triple Moving Average Crossover
        elif kwargs['entry_type'] == '3ma':

            # Set the entry label
            entry_label = (str(kwargs['ma1'])
                           +'-, '
                           +str(kwargs['ma2'])
                           +'-, '
                           +str(kwargs['ma3'])
                           +'-day : '
                           +kwargs['default_dict']['df_entry_signal_labels'][
                               kwargs['entry_type']][0]
                           +ma_type_label
                           +kwargs['default_dict']['df_entry_signal_labels'][
                               kwargs['entry_type']][1])


        # Quad Moving Average Crossover
        elif kwargs['entry_type'] == '4ma':

            # Set the entry label
            entry_label = (str(kwargs['ma1'])
                           +'-, '
                           +str(kwargs['ma2'])
                           +'-, '
                           +str(kwargs['ma3'])
                           +'-, '
                           +str(kwargs['ma4'])
                           +'-day : '
                           +kwargs['default_dict']['df_entry_signal_labels'][
                               kwargs['entry_type']][0]
                           +ma_type_label
                           +kwargs['default_dict']['df_entry_signal_labels'][
                               kwargs['entry_type']][1])


        # Parabolic SAR
        elif kwargs['entry_type'] == 'sar':

            # Set the entry label
            entry_label = (str(kwargs['entry_period'])
                          +'-day '
                          +str(np.round(kwargs[
                              'entry_acceleration_factor'] * 100, 1))
                          +'% AF '
                          +kwargs['default_dict']['df_entry_signal_labels'][
                              kwargs['entry_type']])


        # Channel Breakout
        elif kwargs['entry_type'] == 'channel_breakout':

            # Set the entry label
            entry_label = (str(kwargs['entry_period'])
                           +'-day : '
                           +kwargs['default_dict']['df_entry_signal_labels'][
                               kwargs['entry_type']])


        # Stochastic Crossover, Stochastic Pop, Stochastic Over Under and
        # Relative Strength Index
        elif kwargs['entry_type'] in ['stoch_cross', 'stoch_over_under',
                                      'stoch_pop', 'rsi']:

            # Set the entry label
            entry_label = (str(kwargs['entry_period'])
                           +'-day '
                           +str(kwargs['entry_overbought'])
                           +'-'
                           +str(kwargs['entry_oversold'])
                           +' : '
                           +kwargs['default_dict']['df_entry_signal_labels'][
                               kwargs['entry_type']])


        # Commodity Channel Index, Momentum and Volatility
        elif kwargs['entry_type'] in ['cci', 'momentum', 'volatility']:

            # Set the entry label
            entry_label = (str(kwargs['entry_period'])
                           +'-day '
                           +str(int(kwargs['entry_threshold']*100))
                           +'% : '
                           +kwargs['default_dict']['df_entry_signal_labels'][
                               kwargs['entry_type']])

        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid entry type")


        # Exit labels

        # Parabolic SAR
        if kwargs['exit_type'] == 'sar':

            # Set the exit label
            exit_label = (str(kwargs['exit_period'])
                          +'-day '
                          +str(np.round(kwargs[
                              'exit_acceleration_factor'] * 100, 1))
                          +'% AF '
                          +kwargs['default_dict']['df_exit_signal_labels'][
                              kwargs['exit_type']])


        # Stochastic Crossover and Trailing Relative Strength Index
        elif kwargs['exit_type'] in ['stoch_cross', 'rsi_trail']:

            # Set the exit label
            exit_label = (str(kwargs['exit_period'])
                           +'-day '
                           +str(kwargs['exit_overbought'])
                           +'-'
                           +str(kwargs['exit_oversold'])
                           +' : '
                           +kwargs['default_dict']['df_exit_signal_labels'][
                               kwargs['exit_type']])


        # Volatility
        elif kwargs['exit_type'] in ['volatility']:

            # Set the exit label
            exit_label = (str(kwargs['exit_period'])
                          +'-day '
                          +str(int(kwargs['exit_threshold']*100))
                          +'% : '
                          +kwargs['default_dict']['df_exit_signal_labels'][
                              kwargs['exit_type']])


        # Trailing Stop and Profit Target
        elif kwargs['exit_type'] in ['trailing_stop', 'profit_target']:

            # Set the exit label
            exit_label = ('$'
                          +str(int(kwargs['exit_amount']))
                          +' '
                          +kwargs['default_dict']['df_exit_signal_labels'][
                               kwargs['exit_type']])


        # Support/Resistance, Key Reversal Day and n-Day Range
        elif kwargs['exit_type'] in ['sup_res', 'key_reversal', 'nday_range']:

            # Set the exit label
            exit_label = (str(kwargs['exit_period'])
                          +'-day '
                          +kwargs['default_dict']['df_exit_signal_labels'][
                              kwargs['exit_type']])


        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid exit type")


        # Stop labels

        # Initial Dollar, Breakeven, Trailing Close and Trailing High Low
        if kwargs['stop_type'] in ['initial_dollar', 'breakeven',
                                   'trail_close', 'trail_high_low']:

            # Set the stop label
            stop_label = ('$'
                          +str(int(kwargs['stop_amount']))
                          +' '
                          +kwargs['default_dict']['df_stop_signal_labels'][
                               kwargs['stop_type']])


        # Support / Resistance and Immediate Profit
        elif kwargs['stop_type'] in ['sup_res', 'immediate_profit']:

            # Set the stop label
            stop_label = (str(kwargs['stop_period'])
                          +'-day '
                          +kwargs['default_dict']['df_stop_signal_labels'][
                              kwargs['stop_type']])


        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid stop type")

        #self.strategy_label = entry_label

        return entry_label, exit_label, stop_label


class Dates():
    """
    Date calculation and formatting functions.

    """

    @staticmethod
    def date_set(start_date, end_date, lookback):
        """
        Create start and end dates if not supplied

        Parameters
        ----------
        start_date : Str, optional
            Date to begin backtest. Format is YYYY-MM-DD. The default is 750
            business days prior (circa 3 years).
        end_date : Str, optional
            Date to end backtest. Format is YYYY-MM-DD. The default is the
            last business day.
        lookback : Int, optional
            Number of business days to use for the backtest. The default is 750
            business days (circa 3 years).

        Returns
        -------
        start_date : Str
            Date to begin backtest. Format is YYYY-MM-DD.
        end_date : Str
            Date to end backtest. Format is YYYY-MM-DD.

        """

        # If end date is not supplied, set to previous working day
        if end_date is None:
            end_date_as_dt = (dt.datetime.today() - BDay(1)).date()
            end_date = str(end_date_as_dt)

        # If start date is not supplied, set to today minus lookback period
        if start_date is None:
            start_date_as_dt = (dt.datetime.today() -
                                pd.Timedelta(days=lookback*(365/250))).date()
            start_date = str(start_date_as_dt)

        return start_date, end_date
