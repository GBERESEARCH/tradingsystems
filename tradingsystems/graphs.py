"""
Graph the performance of the trading strategy

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import axes
from matplotlib.ticker import FormatStrFormatter
from tradingsystemsdata.graph_data import GraphData
# pylint: disable=unbalanced-tuple-unpacking
# pylint: disable=no-else-return

class PerformanceGraph():
    """
    Class of functions used to graph trading system performance

    """

    @classmethod
    def two_panel_graph(
        cls,
        es_dict: dict,
        params: dict,
        tables: dict,
        signals: bool | None = None) -> dict:
        """
        Create the 2 panel graph

        Parameters
        ----------
        signals : Bool
            Whether to plot the Buy/Sell signals on the price chart.

        Returns
        -------
        Matplotlib Chart
            Displays the chart.

        """

        # Setup initialization variables
        graph_params = GraphData.graph_variables(
                prices=tables['prices'], entry_type=params['entry_type'],
                entry_signal_indicators=es_dict['entry_signal_indicators'])

        # Set the matplotlib style to use for the charts
        with plt.style.context('fivethirtyeight'):

            # Set up the 2 graphs
            ax1, ax2 = cls._two_panel_setup(
                prices=tables['prices'], graph_params=graph_params,
                params=params, es_dict=es_dict)

            # If signals are to be plotted
            if signals:

                # Create the trade signal points
                signal_dict = GraphData.create_signals(
                    prices=tables['prices'], graph_params=graph_params)

                # Add these to the price chart
                ax1 = cls._plot_signals(axis=ax1, signal_dict=signal_dict)

            # Add legend, labels and titles to the graphs
            ax1, ax2 = cls._two_panel_legend(
                ax1=ax1, ax2=ax2, perf_dict=tables['perf_dict'])

        params['graph_params'] = graph_params
        params['signal_dict'] = signal_dict

        # Plot the graphs
        plt.show()

        return params


    @classmethod
    def three_panel_graph(
        cls,
        es_dict: dict,
        params: dict,
        tables: dict,
        signals: bool | None = None) -> dict:
        """
        Create the 3 panel graph

        Parameters
        ----------
        signals : Bool
            Whether to plot the Buy/Sell signals on the price chart.

        Returns
        -------
        Matplotlib Chart
            Displays the chart.

        """

        # Setup initialization variables
        graph_params = GraphData.graph_variables(
                prices=tables['prices'], entry_type=params['entry_type'],
                entry_signal_indicators=es_dict['entry_signal_indicators'])

        # All but the Stochastic, ADX and MACD Entry methods use a single
        # indicator column
        if params['entry_type'] not in ['stoch_cross', 'stoch_pop',
                                        'stoch_over_under', 'adx', 'macd']:
            indicator = tables['prices'][
                es_dict['entry_signal_indicators'][params['entry_type']]]
        else:
            indicator = None

        # Set the matplotlib style to use for the charts
        with plt.style.context('fivethirtyeight'):

            if params['entry_type'] == 'macd':
                # Set up the 3 graphs
                ax1, ax2, ax3, axt = cls._three_panel_setup(
                    prices=tables['prices'], graph_params=graph_params,
                    params=params, es_dict=es_dict)
            else:
                # Set up the 3 graphs
                ax1, ax2, ax3 = cls._three_panel_setup(
                    prices=tables['prices'], graph_params=graph_params,
                    params=params, es_dict=es_dict)

            # If signals are to be plotted
            if signals:

                # Create the trade signal points
                signal_dict = GraphData.create_signals(
                    prices=tables['prices'], graph_params=graph_params)

                # Add these to the price chart
                ax1 = cls._plot_signals(axis=ax1, signal_dict=signal_dict)

            # Set the overbought / oversold lines on the indicator chart
            ax2 = cls._indicator_format(
                axis=ax2, dates=graph_params['dates'], indicator=indicator,
                params=params)

            if params['entry_type'] == 'macd':
                axt = cls._indicator_format(
                    axis=axt, dates=graph_params['dates'], indicator=indicator,
                    params=params)

            # Add legend, labels and titles to the graphs
            if params['entry_type'] == 'macd':
                axes_dict = {'ax1':ax1, 'ax2':ax2, 'ax3':ax3, 'axt':axt}
                ax1, ax2, ax3, axt = cls._three_panel_legend(
                    axes_dict=axes_dict, perf_dict=tables['perf_dict'],
                    params=params,
                    entry_signal_labels=es_dict['entry_signal_labels'])
            else:
                axes_dict = {'ax1':ax1, 'ax2':ax2, 'ax3':ax3}
                ax1, ax2, ax3 = cls._three_panel_legend(
                    axes_dict=axes_dict, perf_dict=tables['perf_dict'],
                    params=params,
                    entry_signal_labels=es_dict['entry_signal_labels'])

        params['graph_params'] = graph_params
        params['signal_dict'] = signal_dict

        # Plot the graphs
        plt.show()

        return params


    @classmethod
    def _two_panel_setup(
        cls,
        prices: pd.DataFrame,
        graph_params: dict,
        params: dict,
        es_dict: dict) -> tuple[axes.Axes,axes.Axes]:
        """
        Set up the 2 panel chart

        Parameters
        ----------
        dates : Pandas Series
            The dates to plot on the x-axis.
        price : Pandas Series
            Closing Prices.
        equity : Pandas Series
            Daily Mark to Market Equity level.

        Returns
        -------
        ax1 : Matplotlib Axes object
            The main price chart.
        ax2 : Matplotlib Axes object
            The MTM Equity chart.

        """

        # If the entry is a moving average
        if params['entry_type'] in ['2ma', '3ma', '4ma']:

            # The indicator label has 2 parts to combine
            indicator_label = (
                es_dict['entry_signal_labels'][params['entry_type']][0]
                + es_dict['entry_signal_labels'][params['entry_type']][1])
        # Otherwise
        else:
            # Assign the label from the default dictionary
            indicator_label = es_dict[
                'entry_signal_labels'][params['entry_type']]

        # Set the figure size to 25 wide and 10 high
        plt.rcParams['figure.figsize'] = (25, 10)

        # Plot the 2 charts as subplots, the first 5 rows high, starting at
        # zero, the second 3 rows high, starting at 6
        ax1 = plt.subplot2grid((9,1), (0,0), rowspan = 5, colspan = 1)
        ax2 = plt.subplot2grid((9,1), (6,0), rowspan = 3, colspan = 1)

        ax1 = cls._axis_scale(
            ax1=ax1, graph_params=graph_params, params=params)

        # Plot price against time in the first graph
        ax1.plot(graph_params['dates'],
                 graph_params['price'],
                 linewidth=1.5,
                 label='Close Price')

        # If the entry is Parabolic SAR
        if params['entry_type'] == 'sar':

            # Extract the SAR series from the core DataFrame
            indicator = prices[
                es_dict['entry_signal_indicators'][params['entry_type']]]

            # Add these as a scatter to the price-time chart
            ax1.scatter(graph_params['dates'],
                        indicator,
                        marker='.',
                        color='red',
                        s=10,
                        label=str(params['entry_period'])
                        +'-day '
                        +indicator_label)

        # If the entry is a channel breakout
        elif params['entry_type'] == 'channel_breakout':

            # Extract Rolling Low and Rolling High Closes
            lower_channel = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][0]]
            upper_channel = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][1]]

            # Add these to the price-time chart
            ax1.plot(graph_params['dates'],
                     lower_channel,
                     linewidth=2,
                     label=str(params['entry_period'])+'-day Low')
            ax1.plot(graph_params['dates'],
                     upper_channel,
                     linewidth=2,
                     label=str(params['entry_period'])+'-day High')

        # If the entry is a moving average
        elif params['entry_type'] in ['2ma', '3ma', '4ma']:

            # Extract and plot the first 2 moving averages
            ma_1 = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][0]]
            ax1.plot(graph_params['dates'],
                     ma_1,
                     linewidth=2,
                     label=str(params['ma1'])+'-day MA')

            ma_2 = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][1]]
            ax1.plot(graph_params['dates'],
                     ma_2,
                     linewidth=2,
                     label=str(params['ma2'])+'-day MA')

            # If the entry type is not a double moving average
            if params['entry_type'] != '2ma':

                # Plot the third moving average
                ma_3 = prices[
                    es_dict['entry_signal_indicators'][
                        params['entry_type']][2]]
                ax1.plot(graph_params['dates'],
                         ma_3,
                         linewidth=2,
                         label=str(params['ma3'])+'-day MA')

                # If the entry type is not the triple moving average
                if params['entry_type'] != '3ma':

                    # Plot the fourth moving average
                    ma_4 = prices[
                        es_dict['entry_signal_indicators'][
                            params['entry_type']][3]]
                    ax1.plot(graph_params['dates'],
                             ma_4,
                             linewidth=2,
                             label=str(params['ma4'])+'-day MA')

        # Plot the MTM Equity curve
        ax2.plot(graph_params['dates'],
                 graph_params['equity'],
                 linewidth=1.5,
                 label='MTM Equity')

        return ax1, ax2


    @classmethod
    def _three_panel_setup(
        cls,
        prices: pd.DataFrame,
        graph_params: dict,
        params: dict,
        es_dict: dict) -> tuple[
            axes.Axes, axes.Axes, axes.Axes, axes.Axes] | tuple[
                axes.Axes, axes.Axes, axes.Axes]:
        """
        Set up the 3 panel chart

        Parameters
        ----------
        dates : Pandas Series
            The dates to plot on the x-axis.
        prices : Pandas Series
            Closing Prices.
        equity : Pandas Series
            Daily Mark to Market Equity level.

        Returns
        -------
        ax1 : Matplotlib Axes object
            The main price chart.
        ax2 : Matplotlib Axes object
            The Indicator chart.
        ax3 : Matplotlib Axes object
            The MTM Equity chart.

        """

         # Assign the label from the default dictionary
        indicator_label = es_dict[
                'entry_signal_labels'][params['entry_type']]

        # Set the figure size to 25 wide and 14 high
        plt.rcParams['figure.figsize'] = (25, 14)

        # Plot the 3 charts as subplots, the first 5 rows high, starting at
        # zero, the second 3 rows high, starting at 6, the third 3 rows high
        # starting at 10
        ax1 = plt.subplot2grid((13,1), (0,0), rowspan = 5, colspan = 1)
        ax2 = plt.subplot2grid((13,1), (6,0), rowspan = 3, colspan = 1)
        ax3 = plt.subplot2grid((13,1), (10,0), rowspan = 3, colspan = 1)

        ax1 = cls._axis_scale(
            ax1=ax1, graph_params=graph_params, params=params)

        # Plot price against time in the first graph
        ax1.plot(graph_params['dates'],
                 graph_params['price'],
                 color='black',
                 linewidth=1.5,
                 label='Close Price')

        # If the trade entry is a Stochastic
        if 'stoch' in params['entry_type']:

            # Take the slow k and d values from the core DataFrame
            slow_k = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][0]]
            slow_d = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][1]]

            # Plot these in the second chart
            ax2.plot(graph_params['dates'],
                     slow_k,
                     color='blue',
                     linewidth=1.5,
                     label='Slow k')
            ax2.plot(graph_params['dates'],
                     slow_d,
                     color='red',
                     linewidth=1.5,
                     label='Slow d')

        elif 'adx' in params['entry_type']:

            # Take the ADX, DI_plus and DI_minus values from the core DataFrame
            adx = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][0]]
            di_plus = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][1]]
            di_minus = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][2]]

            # Plot these in the second chart
            ax2.plot(graph_params['dates'],
                     adx,
                     color='black',
                     linewidth=2,
                     label='ADX')
            ax2.plot(graph_params['dates'],
                     di_plus,
                     color='blue',
                     linewidth=1.5,
                     label='DI+')
            ax2.plot(graph_params['dates'],
                     di_minus,
                     color='red',
                     linewidth=1.5,
                     label='DI-')

        elif 'macd' in params['entry_type']:

            # Take the MACD, MACD Signal and MACD Histogram values from the core DataFrame
            macd = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][0]]
            macd_signal = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][1]]
            macd_hist = prices[
                es_dict['entry_signal_indicators'][params['entry_type']][2]]

            #num_dates = list(range(1, len(graph_params['dates']) + 1))
            #cat_dates = []
            #for date in num_dates:
            #    cat_dates.append(str(date))

            # Plot these in the second chart
            ax2.plot(graph_params['dates'], #cat_dates,
                     macd,
                     color='blue',
                     linewidth=0.5,
                     label='MACD')
            ax2.plot(graph_params['dates'], #cat_dates,
                     macd_signal,
                     color='red',
                     linewidth=0.5,
                     label='Signal')
            #ax2 = macd_hist.plot(
            #    kind='bar',
            #    color=cls._bar_color(macd_hist, 'g', 'r'),
            #    label='Histogram')
            axt = ax2.twinx()
            axt.bar(graph_params['dates'], #cat_dates,
                    macd_hist,
                    color=GraphData.bar_color(macd_hist,'g','r'),
                    width=1,
                    label='MACD_Hist')

            ax2_lim = np.round(max(np.nanmax(np.absolute(macd)),
                                   np.nanmax(np.absolute(macd_signal))))
            axt_lim = np.round(np.nanmax(np.absolute(macd_hist)))
            ax2.set_ylim(-ax2_lim, ax2_lim)
            axt.set_ylim(-axt_lim, axt_lim)

            #ax2.set_xticks([])
            #axt.set_xticks([])

        # Otherwise
        else:
            # Take the indicator column based on entry type and plot in the
            # second chart
            indicator = prices[
                es_dict['entry_signal_indicators'][params['entry_type']]]
            ax2.plot(graph_params['dates'],
                     indicator,
                     color='blue',
                     linewidth=1.5,
                     label=str(params['entry_period'])+'-day '+indicator_label)

        # Plot the MTM Equity curve
        ax3.plot(graph_params['dates'],
                 graph_params['equity'],
                 linewidth=1.5,
                 label='MTM Equity')

        for axis in [ax1, ax2, ax3]:
            axis.set_xlim(graph_params['dates'][0], graph_params['dates'][-1])

        if 'macd' in params['entry_type']:
            axt.set_xlim(graph_params['dates'][0], graph_params['dates'][-1])
            return ax1, ax2, ax3, axt
        else:
            return ax1, ax2, ax3


    @classmethod
    def _axis_scale(
        cls,
        ax1: axes.Axes,
        graph_params: dict,
        params: dict) -> axes.Axes:

        # Set y-axis to 2 decimal places for FX pairs
        if params['asset_type'] == 'fx':
            ax1.yaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))

        else:
            upper, lower = GraphData.set_upper_lower(graph_params=graph_params)

            # Set y-axis to scale decimal places with price.
            if (upper - lower) < 10:
                ax1.yaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))
            elif (upper - lower) < 30:
                ax1.yaxis.set_major_formatter(FormatStrFormatter('% 1.1f'))
            else:
                ax1.yaxis.set_major_formatter(FormatStrFormatter('% 1.0f'))

        return ax1


    @staticmethod
    def _plot_signals(
        axis: axes.Axes,
        signal_dict: dict) -> axes.Axes:
        """
        Plot trade signals on price-time chart

        Parameters
        ----------
        axis : Matplotlib Axes object
            The main price chart.
        signal_dict : Dict
            Dictionary containing the trade signal details.

        Returns
        -------
        axis : Matplotlib Axes object
            The main price chart.

        """
        # Plot the signals to buy and go long
        axis.scatter(signal_dict['buy_long_dates'],
                    signal_dict['buy_long_marker'],
                    marker='^',
                    color='green',
                    s=100,
                    label='Go Long')

        # Plot the signals to buy and flatten the position
        axis.scatter(signal_dict['buy_flat_dates'],
                    signal_dict['buy_flat_marker'],
                    marker='^',
                    color='deepskyblue',
                    s=100,
                    label='Flatten Short')

        # Plot the signals to sell and flatten the position
        axis.scatter(signal_dict['sell_flat_dates'],
                    signal_dict['sell_flat_marker'],
                    marker='v',
                    color='teal',
                    s=100,
                    label='Flatten Long')

        # Plot the signals to sell and go short
        axis.scatter(signal_dict['sell_short_dates'],
                    signal_dict['sell_short_marker'],
                    marker='v',
                    color='red',
                    s=100,
                    label='Go Short')

        return axis


    @staticmethod
    def _indicator_format(
        axis: axes.Axes,
        params: dict,
        indicator: pd.Series | None = None,
        dates: pd.Series | None = None) -> axes.Axes:
        """
        Apply Overbought / Oversold formatting to the indicator chart

        Parameters
        ----------
        ax : Matplotlib Axes object
            The Indicator chart.
        dates : Pandas Series
            The dates to plot on the x-axis.
        indicator : Pandas Series
            The values of the indicator for these dates.

        Returns
        -------
        ax : Matplotlib Axes object
            The Indicator chart.

        """

        # For all the indicators other than momentum, volatility, adx and macd
        if params['entry_type'] not in [
            'momentum', 'volatility', 'adx', 'macd']:

            # Plot horizontal overbought and oversold lines
            axis.axhline(
                y=params['entry_oversold'],
                color='black',
                linewidth=1)
            axis.axhline(
                y=params['entry_overbought'],
                color='black',
                linewidth=1)

            # For the RSI and CCI indicators
            if params['entry_type'] in ['rsi', 'cci']:

                # Fill the area over the overbought line to the indicator
                axis.fill_between(
                    dates, indicator, params['entry_overbought'],
                    where=indicator>=params['entry_overbought'],
                    interpolate=True, color='red')

                # Fill the area under the oversold line to the indicator
                axis.fill_between(
                    dates, indicator, params['entry_oversold'],
                    where=indicator<=params['entry_oversold'],
                    interpolate=True, color='red')

        if params['entry_type'] == 'adx':
            # Plot horizontal trend threshold
            axis.axhline(
                y=params['adx_threshold'],
                color='black',
                linewidth=2)

        if params['entry_type'] == 'macd':
            # Plot horizontal zero line
            axis.axhline(
                y=0,
                color='black',
                linewidth=1)

        return axis


    @staticmethod
    def _two_panel_legend(
        ax1: axes.Axes,
        ax2: axes.Axes,
        perf_dict: dict) -> tuple[axes.Axes, axes.Axes]:
        """
        Create the legend, axis labels and titles for the 2 panel graph

        Parameters
        ----------
        ax1 : Matplotlib Axes object
            The main price chart.
        ax2 : Matplotlib Axes object
            The MTM Equity chart.

        Returns
        -------
        ax1 : Matplotlib Axes object
            The main price chart.
        ax2 : Matplotlib Axes object
            The MTM Equity chart.

        """

        # Title of main price chart is the contract longname plus the entry
        # strategy
        ax1.set_title(perf_dict['longname']+' - '+perf_dict['entry_label'])
        ax2.set_title("Equity Curve")
        ax1.set_ylabel('Price')
        ax2.set_ylabel('Equity')
        ax1.legend()

        return ax1, ax2


    @staticmethod
    def _three_panel_legend(
        axes_dict: dict[str, axes.Axes],
        perf_dict: dict,
        entry_signal_labels: dict,
        params: dict) -> tuple[
            axes.Axes, axes.Axes, axes.Axes, axes.Axes] | tuple[
                axes.Axes, axes.Axes, axes.Axes]:
        """
        Create the legend, axis labels and titles for the 3 panel graph

        Parameters
        ----------
        ax1 : Matplotlib Axes object
            The main price chart.
        ax2 : Matplotlib Axes object
            The Indicator chart.
        ax3 : Matplotlib Axes object
            The MTM Equity chart.

        Returns
        -------
        ax1 : Matplotlib Axes object
            The main price chart.
        ax2 : Matplotlib Axes object
            The Indicator chart.
        ax3 : Matplotlib Axes object
            The MTM Equity chart.

        """
        ax1 = axes_dict['ax1']
        ax2 = axes_dict['ax2']
        ax3 = axes_dict['ax3']

        # Title of main price chart is the contract longname plus the entry
        # strategy
        ax1.set_title(perf_dict['longname']+' - '+perf_dict['entry_label'])
        ax1.set_ylabel('Price')

        # Title of the Indicator chart is the time period plus indicator name
        ax2.set_title(str(params['entry_period'])
                      +'-day '
                      +entry_signal_labels[params['entry_type']])
        if params['entry_type'] in ['adx', 'macd', 'stoch_cross', 'stoch_pop',
                                    'stoch_over_under']:
            ax2.legend(loc=2)
        ax3.set_title("Equity Curve")
        ax3.set_ylabel('Equity')
        ax1.legend()

        if params['entry_type'] == 'macd':
            axt = axes_dict['axt']
            ax2.set_ylabel('MACD - Signal')
            axt.set_ylabel('Histogram')
            ax2.set_title(entry_signal_labels[params['entry_type']])
            return ax1, ax2, ax3, axt
        else:
            return ax1, ax2, ax3
