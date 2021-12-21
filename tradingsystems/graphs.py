"""
Graph the performance of the trading strategy

"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

class PerformanceGraph():
    """
    Class of functions used to graph trading system performance

    """

    @classmethod
    def two_panel_graph(
            cls, signals=None, tables=None, params=None, es_dict=None):
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
        graph_params = cls._graph_variables(
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
                signal_dict = cls._create_signals(
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
            cls, signals=None, tables=None, params=None, es_dict=None):
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
        graph_params = cls._graph_variables(
                prices=tables['prices'], entry_type=params['entry_type'],
                entry_signal_indicators=es_dict['entry_signal_indicators'])

        # All but the Stochastic & ADX Entry methods use a single indicator
        # column
        if ('stoch' and 'adx') not in params['entry_type']:
            indicator = tables['prices'][
                es_dict['entry_signal_indicators'][params['entry_type']]]
        else:
            indicator = None

        # Set the matplotlib style to use for the charts
        with plt.style.context('fivethirtyeight'):

            # Set up the 3 graphs
            ax1, ax2, ax3 = cls._three_panel_setup(
                prices=tables['prices'], graph_params=graph_params,
                params=params, es_dict=es_dict)

            # If signals are to be plotted
            if signals:

                # Create the trade signal points
                signal_dict = cls._create_signals(
                    prices=tables['prices'], graph_params=graph_params)

                # Add these to the price chart
                ax1 = cls._plot_signals(axis=ax1, signal_dict=signal_dict)

            # Set the overbought / oversold lines on the indicator chart
            ax2 = cls._indicator_format(
                axis=ax2, dates=graph_params['dates'], indicator=indicator,
                params=params)

            # Add legend, labels and titles to the graphs
            axes = {'ax1':ax1, 'ax2':ax2, 'ax3':ax3}
            ax1, ax2, ax3 = cls._three_panel_legend(
                axes=axes, perf_dict=tables['perf_dict'],
                params=params,
                entry_signal_labels=es_dict['entry_signal_labels'])

        params['graph_params'] = graph_params
        params['signal_dict'] = signal_dict

        # Plot the graphs
        plt.show()

        return params


    @staticmethod
    def _graph_variables(
            prices=None, entry_type=None, entry_signal_indicators=None):
        """
        Create graph initialization variables

        Returns
        -------
        dates : Pandas Series
            The dates to plot on the x-axis.
        price : Pandas Series
            Closing Prices.
        equity : Pandas Series
            Daily Mark to Market Equity level.
        cumsig : Pandas Series
            The cumulative buy / sell signal.
        lower_bound : Pandas Series
            Lower point to set where trading signals are plotted from.
        upper_bound : Pandas Series
            Upper point to set where trading signals are plotted from.

        """

        # Dictionary to store default params
        graph_params = {}

        # Set the dates to the index of the main DataFrame
        graph_params['dates'] = prices.index

        # Closing Prices
        graph_params['price'] = prices['Close']

        # MTM Equity
        graph_params['equity'] = prices['mtm_equity']

        # Cumulative sum of the combined entry, exit and stop signal
        graph_params['cumsig'] = prices['combined_signal'].cumsum()

        # The lower and upper bounds are used in setting where the trade
        # signals are plotted on the price chart
        # If the entry is a channel breakout
        if entry_type == 'channel_breakout':

            # Set the lower bound as rolling low close prices
            graph_params['lower_bound'] = prices[
                entry_signal_indicators[entry_type][0]]

            # Set the upper bound as rolling high close prices
            graph_params['upper_bound'] = prices[
                entry_signal_indicators[entry_type][1]]

        elif entry_type in ['2ma', '3ma', '4ma']:

            # Set the lower bound as the lowest of the moving average values
            # and the price
            graph_params['lower_bound'] = prices['min_ma']

            # Set the upper bound as the highest of the moving average values
            # and the price
            graph_params['upper_bound'] = prices['max_ma']

        # Otherwise
        else:
            # Set the upper and lower bounds to the closing price
            graph_params['lower_bound'] = graph_params['price']
            graph_params['upper_bound'] = graph_params['price']

        return graph_params


    @classmethod
    def _two_panel_setup(
            cls, prices=None, graph_params=None, params=None, es_dict=None):
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
            cls, prices=None, graph_params=None, params=None, es_dict=None):
        """
        Set up the 3 panel chart

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

        return ax1, ax2, ax3


    @classmethod
    def _axis_scale(cls, ax1, graph_params, params):

        # Set y-axis to 2 decimal places for FX pairs
        if params['asset_type'] == 'fx':
            ax1.yaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))

        else:
            upper, lower = cls._set_upper_lower(graph_params=graph_params)

            # Set y-axis to scale decimal places with price.
            if (upper - lower) < 10:
                ax1.yaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))
            elif (upper - lower) < 30:
                ax1.yaxis.set_major_formatter(FormatStrFormatter('% 1.1f'))
            else:
                ax1.yaxis.set_major_formatter(FormatStrFormatter('% 1.0f'))

        return ax1


    @classmethod
    def _create_signals(
            cls, prices=None, graph_params=None):
        """
        Create trade signals to be plotted on main price chart

        Parameters
        ----------
        cumsig : Pandas Series
            The cumulative buy / sell signal.
        lower_bound : Pandas Series
            Lower point to set where trading signals are plotted from.
        upper_bound : Pandas Series
            Upper point to set where trading signals are plotted from.

        Returns
        -------
        signal_dict : Dict
            Dictionary containing the trade signal details.

        """
        # Create empty dictionary
        signal_dict = {}
        upper, lower = cls._set_upper_lower(graph_params=graph_params)

        buy_sell_distance = 0.10 * (upper - lower) # 0.07
        flat_distance = 0.15 * (upper - lower) # 0.1

        # Buy signal to go long is where the current cumulative signal is to be
        # long when yesterday it was flat
        signal_dict['buy_long_signals'] = (
            (graph_params['cumsig'] == 1)
            & (graph_params['cumsig'].shift() != 1))

        # Place the marker at the specified distance below the lower bound
        signal_dict['buy_long_marker'] = (
            graph_params['lower_bound']
            * signal_dict['buy_long_signals']
            - buy_sell_distance)
            #- graph_params['lower_bound'].max()*buy_sell_distance)

        signal_dict['buy_long_marker'] = signal_dict[
            'buy_long_marker'][signal_dict['buy_long_signals']]

        # Set the dates for the buy long signals
        signal_dict['buy_long_dates'] = prices.index[
            signal_dict['buy_long_signals']]

        # Buy signal to go flat is where the current cumulative signal is to be
        # flat when yesterday it was short
        signal_dict['buy_flat_signals'] = (
            (graph_params['cumsig'] == 0)
            & (graph_params['cumsig'].shift() == -1))

        # Place the marker at the specified distance below the lower bound
        signal_dict['buy_flat_marker'] = (
            graph_params['lower_bound']
            * signal_dict['buy_flat_signals']
            - flat_distance)
            #- graph_params['lower_bound'].max()*flat_distance)

        signal_dict['buy_flat_marker'] = signal_dict[
            'buy_flat_marker'][signal_dict['buy_flat_signals']]

        # Set the dates for the buy flat signals
        signal_dict['buy_flat_dates'] = prices.index[
            signal_dict['buy_flat_signals']]

        # Sell signal to go flat is where the current cumulative signal is to
        # be flat when yesterday it was long
        signal_dict['sell_flat_signals'] = (
            (graph_params['cumsig'] == 0)
            & (graph_params['cumsig'].shift() == 1))

        # Place the marker at the specified distance above the upper bound
        signal_dict['sell_flat_marker'] = (
            graph_params['upper_bound']
            * signal_dict['sell_flat_signals']
            + flat_distance)
            #+ graph_params['upper_bound'].max()*flat_distance)

        signal_dict['sell_flat_marker'] = signal_dict[
            'sell_flat_marker'][signal_dict['sell_flat_signals']]

        # Set the dates for the sell flat signals
        signal_dict['sell_flat_dates'] = prices.index[
            signal_dict['sell_flat_signals']]

        # Set the dates for the sell short signals
        signal_dict['sell_short_signals'] = (
            (graph_params['cumsig'] == -1)
            & (graph_params['cumsig'].shift() != -1))

        # Place the marker at the specified distance above the upper bound
        signal_dict['sell_short_marker'] = (
            graph_params['upper_bound']
            * signal_dict['sell_short_signals']
            + buy_sell_distance)
            #+ graph_params['upper_bound'].max()*buy_sell_distance)

        signal_dict['sell_short_marker'] = signal_dict[
            'sell_short_marker'][signal_dict['sell_short_signals']]

        # Set the dates for the sell short signals
        signal_dict['sell_short_dates'] = prices.index[
            signal_dict['sell_short_signals']]

        return signal_dict


    @staticmethod
    def _set_upper_lower(graph_params):
        # Set upper to the max of the upper bound and lower to the lowest
        # non-zero value of the lower bound, stripping zeros and nan values

        upper = graph_params['upper_bound'][
            graph_params['upper_bound'] != 0].dropna().max()
        lower = graph_params['lower_bound'][
            graph_params['lower_bound'] != 0].dropna().min()

        return upper, lower


    @staticmethod
    def _plot_signals(axis=None, signal_dict=None):
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
            axis=None, indicator=None, dates=None,  params=None):
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

        # For all the indicators other than momentum and volatility
        if params['entry_type'] not in ['momentum', 'volatility', 'adx']:

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

        return axis


    @staticmethod
    def _two_panel_legend(ax1, ax2, perf_dict):
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
            axes, perf_dict, entry_signal_labels, params):
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
        ax1 = axes['ax1']
        ax2 = axes['ax2']
        ax3 = axes['ax3']

        # Title of main price chart is the contract longname plus the entry
        # strategy
        ax1.set_title(perf_dict['longname']+' - '+perf_dict['entry_label'])
        ax1.set_ylabel('Price')

        # Title of the Indicator chart is the time period plus indicator name
        ax2.set_title(str(params['entry_period'])
                      +'-day '
                      +entry_signal_labels[params['entry_type']])
        if 'adx' in params['entry_type']:
            ax2.legend(loc=2)
        ax3.set_title("Equity Curve")
        ax3.set_ylabel('Equity')
        ax1.legend()

        return ax1, ax2, ax3
