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

    @classmethod
    def strategy_labels(cls, params=None, default_dict=None):
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

        entry_label = cls._entry_label(
            params=params, default_dict=default_dict)

        exit_label = cls._exit_label(
            params=params, default_dict=default_dict)

        stop_label = cls._stop_label(
            params=params, default_dict=default_dict)

        return entry_label, exit_label, stop_label


    @staticmethod
    def _entry_label(params=None, default_dict=None):

        # Simple or Exponential Moving Average label
        if params['simple_ma']:
            ma_type_label = 'S'
        else:
            ma_type_label = 'E'


        # Entry labels

        # Double Moving Average Crossover
        if params['entry_type'] == '2ma':

            # Set the entry label
            entry_label = (str(params['ma1'])
                           +'-, '
                           +str(params['ma2'])
                           +'-day : '
                           +default_dict['df_entry_signal_labels'][
                               params['entry_type']][0]
                           +ma_type_label
                           +default_dict['df_entry_signal_labels'][
                               params['entry_type']][1])


        # Triple Moving Average Crossover
        elif params['entry_type'] == '3ma':

            # Set the entry label
            entry_label = (str(params['ma1'])
                           +'-, '
                           +str(params['ma2'])
                           +'-, '
                           +str(params['ma3'])
                           +'-day : '
                           +default_dict['df_entry_signal_labels'][
                               params['entry_type']][0]
                           +ma_type_label
                           +default_dict['df_entry_signal_labels'][
                               params['entry_type']][1])


        # Quad Moving Average Crossover
        elif params['entry_type'] == '4ma':

            # Set the entry label
            entry_label = (str(params['ma1'])
                           +'-, '
                           +str(params['ma2'])
                           +'-, '
                           +str(params['ma3'])
                           +'-, '
                           +str(params['ma4'])
                           +'-day : '
                           +default_dict['df_entry_signal_labels'][
                               params['entry_type']][0]
                           +ma_type_label
                           +default_dict['df_entry_signal_labels'][
                               params['entry_type']][1])


        # Parabolic SAR
        elif params['entry_type'] == 'sar':

            # Set the entry label
            entry_label = (str(params['entry_period'])
                          +'-day '
                          +str(np.round(params[
                              'entry_acceleration_factor'] * 100, 1))
                          +'% AF '
                          +default_dict['df_entry_signal_labels'][
                              params['entry_type']])


        # Channel Breakout
        elif params['entry_type'] == 'channel_breakout':

            # Set the entry label
            entry_label = (str(params['entry_period'])
                           +'-day : '
                           +default_dict['df_entry_signal_labels'][
                               params['entry_type']])


        # Stochastic Crossover, Stochastic Pop, Stochastic Over Under and
        # Relative Strength Index
        elif params['entry_type'] in ['stoch_cross', 'stoch_over_under',
                                      'stoch_pop', 'rsi']:

            # Set the entry label
            entry_label = (str(params['entry_period'])
                           +'-day '
                           +str(params['entry_overbought'])
                           +'-'
                           +str(params['entry_oversold'])
                           +' : '
                           +default_dict['df_entry_signal_labels'][
                               params['entry_type']])


        # Commodity Channel Index, Momentum and Volatility
        elif params['entry_type'] in ['cci', 'momentum', 'volatility']:

            # Set the entry label
            entry_label = (str(params['entry_period'])
                           +'-day '
                           +str(int(params['entry_threshold']*100))
                           +'% : '
                           +default_dict['df_entry_signal_labels'][
                               params['entry_type']])

        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid entry type")


        return entry_label


    @staticmethod
    def _exit_label(params=None, default_dict=None):

        # Exit labels

        # Parabolic SAR
        if params['exit_type'] == 'sar':

            # Set the exit label
            exit_label = (str(params['exit_period'])
                          +'-day '
                          +str(np.round(params[
                              'exit_acceleration_factor'] * 100, 1))
                          +'% AF '
                          +default_dict['df_exit_signal_labels'][
                              params['exit_type']])


        # Stochastic Crossover and Trailing Relative Strength Index
        elif params['exit_type'] in ['stoch_cross', 'rsi_trail']:

            # Set the exit label
            exit_label = (str(params['exit_period'])
                           +'-day '
                           +str(params['exit_overbought'])
                           +'-'
                           +str(params['exit_oversold'])
                           +' : '
                           +default_dict['df_exit_signal_labels'][
                               params['exit_type']])


        # Volatility
        elif params['exit_type'] in ['volatility']:

            # Set the exit label
            exit_label = (str(params['exit_period'])
                          +'-day '
                          +str(int(params['exit_threshold']*100))
                          +'% : '
                          +default_dict['df_exit_signal_labels'][
                              params['exit_type']])


        # Trailing Stop and Profit Target
        elif params['exit_type'] in ['trailing_stop', 'profit_target']:

            # Set the exit label
            exit_label = ('$'
                          +str(int(params['exit_amount']))
                          +' '
                          +default_dict['df_exit_signal_labels'][
                               params['exit_type']])


        # Support/Resistance, Key Reversal Day and n-Day Range
        elif params['exit_type'] in ['sup_res', 'key_reversal', 'nday_range']:

            # Set the exit label
            exit_label = (str(params['exit_period'])
                          +'-day '
                          +default_dict['df_exit_signal_labels'][
                              params['exit_type']])

        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid exit type")

        return exit_label


    @staticmethod
    def _stop_label(params=None, default_dict=None):

        # Stop labels

        # Initial Dollar, Breakeven, Trailing Close and Trailing High Low
        if params['stop_type'] in ['initial_dollar', 'breakeven',
                                   'trail_close', 'trail_high_low']:

            # Set the stop label
            stop_label = ('$'
                          +str(int(params['stop_amount']))
                          +' '
                          +default_dict['df_stop_signal_labels'][
                               params['stop_type']])


        # Support / Resistance and Immediate Profit
        elif params['stop_type'] in ['sup_res', 'immediate_profit']:

            # Set the stop label
            stop_label = (str(params['stop_period'])
                          +'-day '
                          +default_dict['df_stop_signal_labels'][
                              params['stop_type']])


        # Otherwise raise an error
        else:
            raise ValueError("Please enter a valid stop type")


        return stop_label


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

        # If end date is not provided, set to previous working day
        if end_date is None:
            end_date_as_dt = (dt.datetime.today() - BDay(1)).date()
            end_date = str(end_date_as_dt)

        # If start date is not provided, set to today minus lookback period
        if start_date is None:
            start_date_as_dt = (dt.datetime.today() -
                                pd.Timedelta(days=lookback*(365/250))).date()
            start_date = str(start_date_as_dt)

        return start_date, end_date
