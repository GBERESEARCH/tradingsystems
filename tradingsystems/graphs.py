import matplotlib.pyplot as plt

class PerformanceGraph():
        
    @classmethod
    def _two_panel_graph(
            cls, signals=None, df=None, entry_type=None, perf_dict=None, 
            entry_signal_labels=None, entry_signal_indicators=None, 
            entry_period=None, ma1=None, ma2=None, ma3=None, ma4=None):
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
        dates, price, equity, cumsig, lower_bound, \
            upper_bound = cls._graph_variables(df=df, entry_type=entry_type, 
                entry_signal_indicators=entry_signal_indicators)
        
        # Set the matplotlib style to use for the charts    
        with plt.style.context('fivethirtyeight'):
            
            # Set up the 2 graphs
            ax1, ax2 = cls._two_panel_setup(
                df=df, dates=dates, price=price, equity=equity, 
                entry_type=entry_type, entry_signal_labels=entry_signal_labels, 
                entry_signal_indicators=entry_signal_indicators, 
                entry_period=entry_period, ma1=ma1, ma2=ma2, ma3=ma3, ma4=ma4)
            
            # If signals are to be plotted
            if signals:
                
                # Create the trade signal points
                signal_dict = cls._create_signals(
                    df=df, cumsig=cumsig, lower_bound=lower_bound, 
                    upper_bound=upper_bound)
                
                # Add these to the price chart
                ax1 = cls._plot_signals(ax=ax1, signal_dict=signal_dict)
       
            # Add legend, labels and titles to the graphs
            ax1, ax2 = cls._two_panel_legend(
                ax1=ax1, ax2=ax2, perf_dict=perf_dict)
        
        # Plot the graphs
        plt.show()
        
    
    @classmethod
    def _three_panel_graph(
            cls, signals=None, df=None, entry_type=None, 
            entry_signal_labels=None, entry_signal_indicators=None, 
            entry_period=None, entry_overbought=None, entry_oversold=None,
            perf_dict=None):
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
        dates, price, equity, cumsig, lower_bound, \
            upper_bound = cls._graph_variables(
                df=df, entry_type=entry_type, 
                entry_signal_indicators=entry_signal_indicators)
        
        # All but the Stochastic Entry methods use a single indicator column
        if 'stoch' not in entry_type:
            indicator = df[entry_signal_indicators[entry_type]]
        else:
            indicator = None
       
        # Set the matplotlib style to use for the charts
        with plt.style.context('fivethirtyeight'):
            
            # Set up the 3 graphs
            ax1, ax2, ax3 = cls._three_panel_setup(
                df=df, dates=dates, price=price, equity=equity, 
                entry_type=entry_type, entry_signal_labels=entry_signal_labels, 
                entry_signal_indicators=entry_signal_indicators, 
                entry_period=entry_period)
            
            # If signals are to be plotted
            if signals:
                
                # Create the trade signal points
                signal_dict = cls._create_signals(
                    df=df, cumsig=cumsig, lower_bound=lower_bound, 
                    upper_bound=upper_bound)
                
                # Add these to the price chart
                ax1 = cls._plot_signals(ax=ax1, signal_dict=signal_dict)
        
            # Set the overbought / oversold lines on the indicator chart
            ax2 = cls._indicator_format(
                ax=ax2, dates=dates, indicator=indicator, 
                entry_type=entry_type, entry_overbought=entry_overbought, 
                entry_oversold=entry_oversold)
            
            # Add legend, labels and titles to the graphs
            ax1, ax2, ax3 = cls._three_panel_legend(
                ax1=ax1, ax2=ax2, ax3=ax3, perf_dict=perf_dict, 
                entry_period=entry_period, entry_type=entry_type, 
                entry_signal_labels=entry_signal_labels)
        
        # Plot the graphs            
        plt.show()


    @staticmethod
    def _graph_variables(
            df=None, entry_type=None, entry_signal_indicators=None):
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
        
        # Set the dates to the index of the main DataFrame
        dates = df.index
        
        # Closing Prices
        price = df['Close']
        
        # MTM Equity
        equity = df['mtm_equity']
        
        # Cumulative sum of the combined entry, exit and stop signal
        cumsig = df['combined_signal'].cumsum()
        
        # The lower and upper bounds are used in setting where the trade 
        # signals are plotted on the price chart 
        # If the entry is a channel breakout
        if entry_type == 'channel_breakout':
            
            # Set the lower bound as rolling low close prices
            lower_bound = df[entry_signal_indicators[entry_type][0]]
            
            # Set the upper bound as rolling high close prices
            upper_bound = df[entry_signal_indicators[entry_type][1]]
        
        # Otherwise
        else:
            # Set the upper and lower bounds to the closing price
            lower_bound = price
            upper_bound = price        
        
        return dates, price, equity, cumsig, lower_bound, upper_bound

    
    @staticmethod    
    def _two_panel_setup(
            df=None, dates=None, price=None, equity=None, entry_type=None, 
            entry_signal_labels=None, entry_signal_indicators=None, 
            entry_period=None, ma1=None, ma2=None, ma3=None, ma4=None):
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
        if entry_type in ['2ma', '3ma', '4ma']:
            
            # The indicator label has 2 parts to combine
            indicator_label = (entry_signal_labels[entry_type][0] 
                               + entry_signal_labels[entry_type][1])
        # Otherwise
        else:
            # Assign the label from the default dictionary
            indicator_label = entry_signal_labels[entry_type]
        
        # Set the figure size to 25 wide and 10 high
        plt.rcParams['figure.figsize'] = (25, 10)
        
        # Plot the 2 charts as subplots, the first 5 rows high, starting at 
        # zero, the second 3 rows high, starting at 6 
        ax1 = plt.subplot2grid((9,1), (0,0), rowspan = 5, colspan = 1)
        ax2 = plt.subplot2grid((9,1), (6,0), rowspan = 3, colspan = 1)
        
        # Plot price against time in the first graph
        ax1.plot(dates, price, linewidth=1.5, label='Close Price')
        
        # If the entry is Parabolic SAR
        if entry_type == 'sar':
            
            # Extract the SAR series from the core DataFrame
            indicator = df[entry_signal_indicators[entry_type]]
            
            # Add these as a scatter to the price-time chart
            ax1.scatter(dates, indicator, marker='.', color='red', s=10, 
                        label=str(entry_period)+'-day '+indicator_label)
        
        # If the entry is a channel breakout
        elif entry_type == 'channel_breakout':
            
            # Extract Rolling Low and Rolling High Closes 
            lower_channel = df[entry_signal_indicators[entry_type][0]]
            upper_channel = df[entry_signal_indicators[entry_type][1]]
            
            # Add these to the price-time chart
            ax1.plot(dates, lower_channel, linewidth=2, 
                     label=str(entry_period)+'-day Low')
            ax1.plot(dates, upper_channel, linewidth=2, 
                     label=str(entry_period)+'-day High')
        
        # If the entry is a moving average
        elif entry_type in ['2ma', '3ma', '4ma']:
            
            # Extract and plot the first 2 moving averages
            ma_1 = df[entry_signal_indicators[entry_type][0]]
            ax1.plot(dates, ma_1, linewidth=2, label=str(ma1)+'-day MA')
            
            ma_2 = df[entry_signal_indicators[entry_type][1]]
            ax1.plot(dates, ma_2, linewidth=2, label=str(ma2)+'-day MA')
            
            # If the entry type is not a double moving average
            if entry_type != '2ma':
                
                # Plot the third moving average
                ma_3 = df[entry_signal_indicators[entry_type][2]]
                ax1.plot(dates, ma_3, linewidth=2, label=str(ma3)+'-day MA')
                
                # If the entry type is not the triple moving average
                if entry_type != '3ma':
                    
                    # Plot the fourth moving average
                    ma_4 = df[entry_signal_indicators[entry_type][3]]
                    ax1.plot(dates, ma_4, linewidth=2, 
                             label=str(ma4)+'-day MA')
        
        # Plot the MTM Equity curve
        ax2.plot(dates, equity, linewidth=1.5, label='MTM Equity')
        
        return ax1, ax2


    @staticmethod
    def _three_panel_setup(
            df=None, dates=None, price=None, equity=None, entry_type=None, 
            entry_signal_labels=None, entry_signal_indicators=None, 
            entry_period=None):
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
        indicator_label = entry_signal_labels[entry_type]
        
        # Set the figure size to 25 wide and 14 high
        plt.rcParams['figure.figsize'] = (25, 14)
        
        # Plot the 3 charts as subplots, the first 5 rows high, starting at 
        # zero, the second 3 rows high, starting at 6, the third 3 rows high 
        # starting at 10 
        ax1 = plt.subplot2grid((13,1), (0,0), rowspan = 5, colspan = 1)
        ax2 = plt.subplot2grid((13,1), (6,0), rowspan = 3, colspan = 1)
        ax3 = plt.subplot2grid((13,1), (10,0), rowspan = 3, colspan = 1)
        
        # Plot price against time in the first graph
        ax1.plot(dates, price, color='black', linewidth=1.5, 
                 label='Close Price')
        
        # If the trade entry is a Stochastic
        if 'stoch' in entry_type:
            
                # Take the slow k and d values from the core DataFrame 
                slow_k = df[entry_signal_indicators[entry_type][0]]
                slow_d = df[entry_signal_indicators[entry_type][1]]
                
                # Plot these in the second chart
                ax2.plot(dates, slow_k, color='blue', linewidth=1.5, 
                     label='Slow k')
                ax2.plot(dates, slow_d, color='red', linewidth=1.5, 
                     label='Slow d')
        # Otherwise
        else:
            # Take the indicator column based on entry type and plot in the 
            # second chart
            indicator = df[entry_signal_indicators[entry_type]]
            ax2.plot(dates, indicator, color='blue', linewidth=1.5, 
                     label=str(entry_period)+'-day '+indicator_label)
    
        # Plot the MTM Equity curve
        ax3.plot(dates, equity, linewidth=1.5, label='MTM Equity')
    
        return ax1, ax2, ax3


    @staticmethod
    def _create_signals(
            df=None, cumsig=None, lower_bound=None, upper_bound=None):    
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
        
        # Buy signal to go long is where the current cumulative signal is to be 
        # long when yesterday it was flat
        signal_dict['buy_long_signals'] = (cumsig == 1) & (cumsig.shift() != 1)
        
        # Place the marker 5% below the lower bound
        signal_dict['buy_long_marker'] = (lower_bound 
                                          * signal_dict['buy_long_signals'] 
                                          - lower_bound.max()*.05)
        
        signal_dict['buy_long_marker'] = signal_dict[
            'buy_long_marker'][signal_dict['buy_long_signals']]
        
        # Set the dates for the buy long signals
        signal_dict['buy_long_dates'] = df.index[
            signal_dict['buy_long_signals']]
        
        # Buy signal to go flat is where the current cumulative signal is to be 
        # flat when yesterday it was short
        signal_dict['buy_flat_signals'] = ((cumsig == 0) 
                                           & (cumsig.shift() == -1))
        
        # Place the marker 8% below the lower bound
        signal_dict['buy_flat_marker'] = (lower_bound 
                                          * signal_dict['buy_flat_signals'] 
                                          - lower_bound.max()*.08)
        
        signal_dict['buy_flat_marker'] = signal_dict[
            'buy_flat_marker'][signal_dict['buy_flat_signals']]
        
        # Set the dates for the buy flat signals
        signal_dict['buy_flat_dates'] = df.index[
            signal_dict['buy_flat_signals']]
        
        # Sell signal to go flat is where the current cumulative signal is to 
        # be flat when yesterday it was long
        signal_dict['sell_flat_signals'] = ((cumsig == 0) 
                                            & (cumsig.shift() == 1))
        
        # Place the marker 8% above the upper bound
        signal_dict['sell_flat_marker'] = (upper_bound 
                                           * signal_dict['sell_flat_signals'] 
                                           + upper_bound.max()*.08)
       
        signal_dict['sell_flat_marker'] = signal_dict[
            'sell_flat_marker'][signal_dict['sell_flat_signals']]
                
        # Set the dates for the sell flat signals
        signal_dict['sell_flat_dates'] = df.index[
            signal_dict['sell_flat_signals']]
        
        # Set the dates for the sell short signals
        signal_dict['sell_short_signals'] = ((cumsig == -1) 
                                             & (cumsig.shift() != -1))
        
        # Place the marker 5% above the upper bound
        signal_dict['sell_short_marker'] = (upper_bound 
                                            * signal_dict['sell_short_signals'] 
                                            + upper_bound.max()*.05)
        
        signal_dict['sell_short_marker'] = signal_dict[
            'sell_short_marker'][signal_dict['sell_short_signals']]
        
        # Set the dates for the sell short signals        
        signal_dict['sell_short_dates'] = df.index[
            signal_dict['sell_short_signals']]
        
        return signal_dict


    @staticmethod
    def _plot_signals(ax=None, signal_dict=None):
        """
        Plot trade signals on price-time chart

        Parameters
        ----------
        ax : Matplotlib Axes object
            The main price chart.
        signal_dict : Dict
            Dictionary containing the trade signal details.

        Returns
        -------
        ax : Matplotlib Axes object
            The main price chart.

        """
        # Plot the signals to buy and go long
        ax.scatter(signal_dict['buy_long_dates'], 
                    signal_dict['buy_long_marker'], 
                    marker='^', 
                    color='green', 
                    s=100, 
                    label='Go Long')
        
        # Plot the signals to buy and flatten the position
        ax.scatter(signal_dict['buy_flat_dates'], 
                    signal_dict['buy_flat_marker'], 
                    marker='^', 
                    color='deepskyblue', 
                    s=100, 
                    label='Flatten Short')

        # Plot the signals to sell and flatten the position        
        ax.scatter(signal_dict['sell_flat_dates'], 
                    signal_dict['sell_flat_marker'], 
                    marker='v', 
                    color='teal', 
                    s=100, 
                    label='Flatten Long')

        # Plot the signals to sell and go short
        ax.scatter(signal_dict['sell_short_dates'], 
                    signal_dict['sell_short_marker'], 
                    marker='v', 
                    color='red', 
                    s=100, 
                    label='Go Short')
        
        return ax


    @staticmethod
    def _indicator_format(
            ax=None, dates=None, indicator=None, entry_type=None, 
            entry_overbought=None, entry_oversold=None):
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
        if entry_type not in ['momentum', 'volatility']:
            
            # Plot horizontal overbought and oversold lines
            ax.axhline(y=entry_oversold, color='black', linewidth=1)
            ax.axhline(y=entry_overbought, color='black', linewidth=1)
    
            # For the RSI and CCI indicators
            if entry_type in ['rsi', 'cci']:
                
                # Fill the area over the overbought line to the indicator 
                ax.fill_between(
                    dates, indicator, entry_overbought, 
                    where=indicator>=entry_overbought, 
                    interpolate=True, color='red')
                
                
                # Fill the area under the oversold line to the indicator 
                ax.fill_between(
                    dates, indicator, entry_oversold, 
                    where=indicator<=entry_oversold, 
                    interpolate=True, color='red')    
        
        return ax


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
            ax1, ax2, ax3, perf_dict, entry_period, entry_signal_labels, 
            entry_type):
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
        # Title of main price chart is the contract longname plus the entry 
        # strategy
        ax1.set_title(perf_dict['longname']+' - '+perf_dict['entry_label'])
        ax1.set_ylabel('Price')
        
        # Title of the Indicator chart is the time period plus indicator name 
        ax2.set_title(str(entry_period)
                      +'-day '
                      +entry_signal_labels[entry_type])
        ax3.set_title("Equity Curve")
        ax3.set_ylabel('Equity')
        ax1.legend()
        
        return ax1, ax2, ax3

