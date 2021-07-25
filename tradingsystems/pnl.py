import numpy as np
import pandas as pd

class Profit():
    
    @classmethod         
    def _profit_data(cls, df, position_size, slippage, commission, equity):
        """
        Adds profit fields to the OHLC data

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        position_size : Int, optional
            The number of units to trade. The default is 100.
        slippage : Float, optional
            The amount of slippage to apply to traded prices in basis points. 
            The default is 5 bps per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.
        equity : Float
            The initial account equity level.    

        Returns
        -------
        df : DataFrame
            Returns the input DataFrame with additional columns.

        """
        
        # Create pnl data
        df = cls._pnl_mtm(df=df, slippage=slippage, commission=commission)
        
        # Create cumulative trade pnl, equity and drawdown data
        df = cls._cumulative_trade_pnl_and_equity(df=df, equity=equity)
        
        # Create perfect profit data
        df = cls._perfect_profit(df=df, position_size=position_size)
           
        return df    


    @classmethod
    def _pnl_mtm(cls, df, slippage, commission):
        """
        Calculate pnl and mark to market columns

        Parameters
        ----------
        df : DataFrame
            The OHLC data and trades signals.
        slippage : Float, optional
            The amount of slippage to apply to traded prices in basis points. 
            The default is 5 bps per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.

        Returns
        -------
        df : DataFrame
            The input data with additional columns.

        """
        
        # daily pnl
        df = cls._daily_pnl(df=df, slippage=slippage, commission=commission)
    
        # total pnl
        df['total_pnl'] = np.array([0]*len(df['Close']), dtype=float)
        df['total_pnl'] = df['daily_pnl'].cumsum()
        
        # position mtm
        df['position_mtm'] = df['position'] * df['Close']
        
        return df
    
    
    @staticmethod
    def _daily_pnl(df, slippage, commission):
        """
        Calculate daily PNL

        Parameters
        ----------
        df : DataFrame
            The OHLC data and trades signals.
        slippage : Float, optional
            The amount of slippage to apply to traded prices in basis points. 
            The default is 5 bps per unit.
        commission : Float, optional
            The amount of commission charge to apply to each trade. The 
            default is $0.00.

        Returns
        -------
        df : DataFrame
            The input data with additional columns.

        """
        
        # Create series of open, close and position
        open = df['Open']
        close = df['Close']
        pos = df['position']        
                
        # Create array of zeros 
        day_pnl = np.array([0]*len(close), dtype=float)
        last_day_trade_pnl = np.array([0]*len(close), dtype=float)
        
        # For each row of closing prices
        for row in range(1, len(close)):
            
            # If the current position is flat
            if pos[row] == 0:
                
                # If the previous days position was flat
                if pos[row - 1] == 0:
                    
                    # Set the pnl to zero
                    day_pnl[row] = 0
                    
                # Otherwise:
                else:
                    # Set the pnl to the previous day's position multiplied by 
                    # the difference between todays open and yesterdays close
                    # less the cost of slippage and commission
                    day_pnl[row] = (
                        (pos[row - 1] * (open[row] - close[row - 1]) 
                         - abs(pos[row - 1] * slippage * 0.0001 * open[row])) 
                        - commission)
            
            # If the current position is not flat
            else:
                
                # If the position is the same as the previous day
                if pos[row] == pos[row - 1]:
                    
                    # Set the pnl to the current position * the difference 
                    # between todays close and yesterdays close
                    day_pnl[row] = pos[row] * (close[row] - close[row - 1])
                
                # If the position is reversed from the previous day
                elif pos[row] == (-1) * pos[row - 1]:
                    day_pnl[row] = (
                        (pos[row] * (close[row] - open[row]) 
                         - abs(pos[row] * slippage * 0.0001 * open[row])) 
                        - commission)
                    last_day_trade_pnl[row] = (
                        (pos[row - 1] * (open[row] - close[row - 1]) 
                         - abs(pos[row - 1] * slippage * 0.0001 * open[row])) 
                        - commission)
                
                # If the position was opened from flat
                else:
                    
                    # Set the pnl to the current position * the difference 
                    # between todays open and todays close less the cost of 
                    # slippage and commission
                    day_pnl[row] = (
                        (pos[row] * (close[row] - open[row]) 
                         - abs(pos[row] * slippage * 0.0001 * open[row])) 
                        - commission)
        
        # Create daily pnl column in DataFrame, rounding to 2dp            
        df['current_trade_pnl'] = np.round(day_pnl, 2)    
        df['last_day_trade_pnl'] = last_day_trade_pnl
        df['daily_pnl'] = df['current_trade_pnl'] + df['last_day_trade_pnl']    
    
        return df
    
       
    @staticmethod    
    def _perfect_profit(df, position_size):
        """
        Theoretical optimal of buying every low and selling every high

        Parameters
        ----------
        df : DataFrame
            The OHLC data, trades signals and pnl.
        position_size : Int
            Number of units traded.

        Returns
        -------
        df : DataFrame
            The input data with additional columns.

        """
        
        # Absolute difference between high and low price multiplied by 
        # position size
        
        dpp = np.array([0.0]*len(df))
        
        for row in range(len(dpp)):
            
            # Calculate Daily Perfect Profit            
            dpp[row] = (abs(df['High'][row] - df['Low'][row]) * position_size)
            
            # If the High and Low are the same
            if dpp[row] == 0:
            
                # Use the previous close
                dpp[row] = (
                    abs(df['High'][row] - df['Close'][row-1]) * position_size)        
        
        # Set this to the daily perfect profit
        df['daily_perfect_profit'] = dpp
        
        # Create array of zeros
        df['total_perfect_profit'] = np.array(
            [0.0]*len(df['Close']), dtype=float)
        
        # Cumulative sum of daily perfect profit column
        df['total_perfect_profit'] = df['daily_perfect_profit'].cumsum()
    
        return df

    
    @staticmethod
    def _cumulative_trade_pnl_and_equity(df, equity):
        """
        Calculate cumulative per trade pnl and various account equity series

        Parameters
        ----------
        df : DataFrame
            The OHLC data and trades signals.
        equity : Float
            The initial account equity level.

        Returns
        -------
        df : DataFrame
            The input data with additional columns.

        """
        # Take the trade number and daily pnl series from df
        trade_number = df['trade_number']
        current_trade_pnl = df['current_trade_pnl']    
        last_day_trade_pnl = df['last_day_trade_pnl']        
        daily_pnl = df['daily_pnl']
    
        # Create arrays of zeros
        cumulative_trade_pnl = np.array([0.0]*len(daily_pnl))
        mtm_equity = np.array([0.0]*len(daily_pnl))
        max_mtm_equity = np.array([0.0]*len(daily_pnl))
        min_mtm_equity = np.array([0.0]*len(daily_pnl))
        closed_equity = np.array([0.0]*len(daily_pnl))
        max_closed_equity = np.array([0.0]*len(daily_pnl))
        open_equity = np.array([0.0]*len(daily_pnl))
        max_retracement = np.array([0.0]*len(daily_pnl))
        ulcer_index_d_sq = np.array([0.0]*len(daily_pnl))
        max_trade_pnl = np.array([0.0]*len(daily_pnl))
        trade_pnl_drawback = np.array([0.0]*len(daily_pnl))
        trade_pnl_drawback_perc = np.array([0.0]*len(daily_pnl))
        
        # Create max drawdown and max gain numpy arrays of zeros
        max_drawdown = np.array([0.0]*len(daily_pnl))
        max_drawdown_perc = np.array([0.0]*len(daily_pnl))
        max_gain = np.array([0.0]*len(daily_pnl))
        max_gain_perc = np.array([0.0]*len(daily_pnl))
    
        # Set the initial equity values
        mtm_equity[0] = equity
        max_mtm_equity[0] = equity
        min_mtm_equity[0] = equity
        closed_equity[0] = equity
        max_closed_equity[0] = equity
        
        # For each row of data
        for row in range(1, len(daily_pnl)):
            
            # The index location of the trade entry date
            trade_first_row = df.index.get_loc(
                df[trade_number==trade_number[row]].index[0])
            
            # The number of days since trade entry
            trade_row_num = row - trade_first_row
            
            # The index location of the trade entry date
            trade_last_row = df.index.get_loc(
                df[trade_number==trade_number[row]].index[-1])
    
            # If there is a current trade
            if trade_number[row] != 0:   
                
                # If it is the trade entry date and there was no prior trade 
                if (trade_row_num == 0 
                    and trade_number[row-1] == 0):
                       
                    # Set cumulative trade pnl to the days pnl
                    cumulative_trade_pnl[row] = daily_pnl[row]

                    # The maximum of the initial days pnl and zero
                    max_trade_pnl[row] = max(daily_pnl[row], 0)
                    
                    # Set the mtm equity to the previous days mtm equity plus
                    # the days pnl
                    mtm_equity[row] = mtm_equity[row-1] + daily_pnl[row]
                    
                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = closed_equity[row-1]


                # If it is the trade entry date and this reverses the position 
                # of a prior trade 
                elif (trade_row_num == 0 
                    and trade_number[row-1] != 0):
                       
                    # Set cumulative trade pnl to the previous days cumulative 
                    # pnl plus the last days pnl
                    cumulative_trade_pnl[row] = (cumulative_trade_pnl[row-1] 
                                                 + last_day_trade_pnl[row])
                   
                    #  Set cumulative trade pnl to the previous days cumulative 
                    # pnl plus the last days pnl
                    max_trade_pnl[row] = max(
                        cumulative_trade_pnl[row], max_trade_pnl[row-1])
                    
                    # Set the mtm equity to the previous days mtm equity plus
                    # the days pnl
                    mtm_equity[row] = mtm_equity[row-1] + daily_pnl[row]
                    
                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = (mtm_equity[row-1] 
                                          + last_day_trade_pnl[row])

    
                # If it is the trade exit date and not a reversal
                elif (trade_last_row - row == 0 
                      and trade_number[row] == trade_number[row-1]):
                    
                    # Set cumulative trade pnl to the previous days cumulative 
                    # pnl plus the days pnl
                    cumulative_trade_pnl[row] = (cumulative_trade_pnl[row-1] 
                                                 + daily_pnl[row]) 
                    
                    # The maximum of the current trade equity and the maximum 
                    # trade equity of the previous day
                    max_trade_pnl[row] = max(
                        cumulative_trade_pnl[row], max_trade_pnl[row-1])
                    
                    # Set the mtm equity to the previous days mtm equity plus
                    # the days pnl
                    mtm_equity[row] = mtm_equity[row-1] + daily_pnl[row]
                    
                    # Set the closed equity to the mtm equity
                    closed_equity[row] = mtm_equity[row] 

                
                # If it is the second day of a reversal trade
                elif (trade_row_num == 1 
                      and trade_number[row-1] != trade_number[row-2]):
                    
                    # Set cumulative trade pnl to the previous days current 
                    # pnl plus the days pnl
                    cumulative_trade_pnl[row] = (current_trade_pnl[row-1] 
                                                 + daily_pnl[row])
                    
                    # The maximum of the first and second days pnl and zero
                    max_trade_pnl[row] = max(cumulative_trade_pnl[row], 
                                             current_trade_pnl[row-1])
                    
                    # Set the mtm equity to the previous days mtm equity plus
                    # the days pnl
                    mtm_equity[row] = mtm_equity[row-1] + daily_pnl[row]
                    
                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = closed_equity[row-1]

                
                # For every other day in the trade 
                else:
                    # Set cumulative trade pnl to the previous days cumulative 
                    # pnl plus the days pnl
                    cumulative_trade_pnl[row] = (cumulative_trade_pnl[row-1] 
                                                 + daily_pnl[row])
                    
                    # The maximum of the current trade equity and the maximum 
                    # trade equity of the previous day
                    max_trade_pnl[row] = max(
                        cumulative_trade_pnl[row], max_trade_pnl[row-1])
                    
                    # Set the mtm equity to the previous days mtm equity plus
                    # the days pnl
                    mtm_equity[row] = mtm_equity[row-1] + daily_pnl[row]
                    
                    # Set the closed equity to the previous days closed equity
                    closed_equity[row] = closed_equity[row-1]


            # If there is no current trade 
            else:
                # Set cumulative trade pnl to zero
                cumulative_trade_pnl[row] = 0
                
                # Set the mtm equity to the previous days mtm equity 
                mtm_equity[row] = mtm_equity[row-1]
                
                # Set the closed equity to the previous days closed equity                    
                closed_equity[row] = closed_equity[row-1]

            # Maximum mtm equity to this point            
            max_mtm_equity[row] = np.max(mtm_equity[0:row+1])
            
            # Minimum mtm equity to this point            
            min_mtm_equity[row] = np.min(mtm_equity[0:row+1])
            
            # Maximum closed equity to this point
            max_closed_equity[row] = np.max(closed_equity[:row+1])    
            
            # Current open equity
            open_equity[row] = mtm_equity[row] - closed_equity[row]
            
            # Maximum of max closed equity and current mtm equity, used in 
            # calculating Average Max Retracement
            max_retracement[row] = max(
                (max_closed_equity[row] - mtm_equity[row]), 0)
            
            # Maximum drawdown is the smallest value of the current cumulative 
            # pnl less the max of all previous rows cumulative pnl and zero
            max_drawdown[row] = mtm_equity[row] - max_mtm_equity[row]
            
            # Percentage Maximum drawdown
            max_drawdown_perc[row] = ((mtm_equity[row] - max_mtm_equity[row]) 
                                      / max_mtm_equity[row])            
          
            # Maximum gain is the largest value of the current cumulative 
            # pnl less the min of all previous rows and zero
            max_gain[row] = mtm_equity[row] - min_mtm_equity[row]
        
            # Percentage Maximum gain
            max_gain_perc[row] = ((mtm_equity[row] - min_mtm_equity[row]) 
                                  / min_mtm_equity[row])
            
            # Squared difference between max mtm equity and current mtm equity,
            # used in calculating Ulcer Index
            ulcer_index_d_sq[row] = (
                (((max_mtm_equity[row] - mtm_equity[row]) 
                 / max_mtm_equity[row]) * 100) ** 2)
            
            # The difference between the highest equity peak of the trade and 
            # the current trade open equity
            trade_pnl_drawback[row] = (max_trade_pnl[row] 
                                       - cumulative_trade_pnl[row])
            
            # The percentage difference between the highest equity peak of the 
            # trade and the current trade open equity
            if max_trade_pnl[row] !=0:
                trade_pnl_drawback_perc[row] = (
                    (max_trade_pnl[row] - cumulative_trade_pnl[row]) 
                    / max_trade_pnl[row])
                       
        df['cumulative_trade_pnl'] = cumulative_trade_pnl
        df['max_trade_pnl'] = max_trade_pnl
        df['trade_pnl_drawback'] = trade_pnl_drawback
        df['trade_pnl_drawback_perc'] = trade_pnl_drawback_perc
        df['mtm_equity'] = mtm_equity
        df['closed_equity'] = closed_equity
        df['open_equity'] = open_equity
        df['max_closed_equity'] = max_closed_equity
        df['max_retracement'] = max_retracement
        df['max_mtm_equity'] = max_mtm_equity
        df['min_mtm_equity'] = min_mtm_equity
        df['max_dd'] = max_drawdown
        df['max_dd_perc'] = max_drawdown_perc
        df['max_gain'] = max_gain
        df['max_gain_perc'] = max_gain_perc
        df['ulcer_index_d_sq'] = ulcer_index_d_sq
            
        return df
    
    
    @staticmethod
    def _trade_data(df):
        """
        Create dictionary of trades, count of the number of trades and lists / 
        dictionaries of winning and losing trades 

        Parameters
        ----------
        df : DataFrame
            The OHLC data with trade signals.

        Returns
        -------
        trades : Dict
            Dictionary containing all the trades.
        num_trades : Int
            Number of trades.
        trades_win_dict : Dict
            Dictionary of winning trades.
        trades_win_list : TYPE
            List of winning trades.
        trades_loss_dict : TYPE
            Dictionary of losing trades.
        trades_loss_list : TYPE
            List of losing trades.

        """
        # profit per trade
        # Create empty trades dictionary
        trades = {}
        
        # Count the number of unique trade numbers (less 1 for the 0 start)
        num_trades = len(pd.unique(df['trade_number'])) - 1
        
        # For each trade number
        for trade_number in range(1, num_trades+1):
            
            # Calculate profit as the sum of daily pnl for that trade number
            profit = df[df['trade_number']==trade_number]['daily_pnl'].sum()
            
            # Assign this number (rounded to 2dp) to the trades dictionary
            trades[trade_number] = np.round(profit, 2)
    
        # Split winning and losing trades
        # Create empty win/loss dictionaries and lists
        trades_win_dict = {}
        trades_win_list = list()
        trades_loss_dict = {}
        trades_loss_list = list()
        
        # for each trade and profit in the trades dictionary
        for key, value in trades.items():
            
            # If the profit is negative
            if value < 0:
                
                # Add the trade to the loss dictionary and list (as a tuple)
                trades_loss_dict[key] = value
                trades_loss_list.append((key, value))
            
            # If the profit is positive
            else:
                
                # Otherwise add to the win dictionary and list (as a tuple)
                trades_win_dict[key] = value
                trades_win_list.append((key, value))
    
        return (trades, num_trades, trades_win_dict, trades_win_list, 
                trades_loss_dict, trades_loss_list)
    
    
    @staticmethod
    def _trade_runs(input_trades_list, run_type='win'):
        """
        Produce data for winning or losing runs of trades

        Parameters
        ----------
        input_trades_list : List
            List of winning or losing trades.
        run_type : Str, optional
            Whether winning or losing run data is being calculated. The 
            default is 'win'.

        Returns
        -------
        max_run_pnl : Float
            Largest profit or loss.
        max_run_count : Int
            Number of trades in largest profit or loss.
        min_run_pnl : Float
            Smallest profit or loss.
        min_run_count : Int
            Number of trades in smallest profit or loss.
        num_runs : Int
            Count of number of winning or losing runs.
        av_run_count : Int
            Count of average number of trades in winning or losing runs.
        av_run_pnl : Float
            Average profit or loss.
        pnl : Tuple
            PNL for each run and number of trades.

        """
        
        # Set initial values
        max_run_count = 1
        run_count = 1
        run_trades_list = []
        total_run_trades_list = []
        last_trade_count = 0
        
        # For each trade in the winning or losing trades list, sorting by the 
        # trade number
        for num, trade in enumerate(sorted(input_trades_list)):
            # For the first trade
            if num == 0:
                
                # Add the trade pnl to the winning / losing trades run list 
                run_trades_list.append(trade[1])
                
                # If this is the last trade
                if num == len(input_trades_list) - 1:
                    total_run_trades_list.append(run_trades_list)
            
            # Otherwise, if the trade number is next in sequence after the 
            # last stored trade number
            elif trade[0] == last_trade_count + 1:
                
                # Increase the run count by one
                run_count +=1
                
                # Update the longest run count
                max_run_count = max(max_run_count, run_count)
                
                # Add the trade pnl to the winning / losing trades run list
                run_trades_list.append(trade[1])
                
                # If this is the last trade
                if num == len(input_trades_list) - 1:
                    total_run_trades_list.append(run_trades_list)    
        
            # If the trade is not the next in sequence:
            else:
                
                # Add the current run to the list of all runs
                total_run_trades_list.append(run_trades_list)   
                
                # If this is not the last trade
                if num != len(input_trades_list) - 1:
                    
                    # Reset the winning / losing trades run list
                    run_trades_list = []
                    
                    # Add the trade pnl to the winning / losing trades run list
                    run_trades_list.append(trade[1])        
                    
                    # Increase the run count by one
                    run_count = 1
 
                # If it is the last trade                 
                else:
                    
                    # Reset the winning / losing trades run list
                    run_trades_list = []
                    
                    # Add the trade pnl to the winning / losing trades run list
                    run_trades_list.append(trade[1])
                    
                    # Add the current run to the list of all runs
                    total_run_trades_list.append(run_trades_list)

            # Set the last trade count number to the current trade number        
            last_trade_count = trade[0]
        
        # Tuple for each run of PNL and number of trades.
        pnl = sorted([(sum(x), len(x)) for x in total_run_trades_list])    
        
        # Values to select for winning runs
        if run_type == 'win':
            
            # If there are any winning trades
            if pnl:
                max_run_pnl = pnl[-1][0]
                max_run_count = pnl[-1][-1]
                min_run_pnl = pnl[0][0]
                min_run_count = pnl[0][-1]
            
            # Otherwise set the values to zero
            else:    
                max_run_pnl = 0
                max_run_count = 0
                min_run_pnl = 0
                min_run_count = 0
                
        # Values to select for losing runs
        else:
            
            # If there are any losing trades
            if pnl:
                max_run_pnl = pnl[0][0]
                max_run_count = pnl[0][-1]
                min_run_pnl = pnl[-1][0]
                min_run_count = pnl[-1][-1]
                
            # Otherwise set the values to zero    
            else:    
                max_run_pnl = 0
                max_run_count = 0
                min_run_pnl = 0
                min_run_count = 0    
                                        
        # Count number of runs as the length of the pnl list     
        num_runs = len(pnl)
        
        if pnl:
 
            # Take the average number of runs as the sum of run lengths in pnl 
            # tuple divided by the number of runs
            av_run_count = int(np.round(sum(j for i, j in pnl) / len(pnl), 0))
        
            # Take the average run pnl as the sum of run pnls in pnl tuple
            # divided by the number of runs
            av_run_pnl = np.round(sum(i for i, j in pnl) / len(pnl), 2)
            
        else:
            av_run_count = 0
            av_run_pnl = 0
            
        return max_run_pnl, max_run_count, min_run_pnl, min_run_count, \
            num_runs, av_run_count, av_run_pnl, pnl
    
    
    @staticmethod    
    def _time_to_recover(df):
        """
        Calculate the time taken for largest loss from peak to trough

        Parameters
        ----------
        df : DataFrame
            The core OHLC DataFrame.

        Returns
        -------
        dd_length : Int
            The number of days from peak to trough.

        """
        # Extract the largest drawdown        
        max_dd_val = min(df['max_dd'])
        
        # Find the start location of the max drawdown
        dd_start = df.index.get_loc(df[df['max_dd']==max_dd_val].index[0])
        
        # Calculate the point where equity has returned to the pre drawdown 
        # level
        if max(df['max_dd'][dd_start:]) == 0:
            dd_length = (df['max_dd'][dd_start:].values == 0).argmax()
            return dd_length
        
        # Otherwise return N/A
        else:
            return 'N/A'


    @staticmethod
    def _time_max_gain(df):
        """
        Calculate the time taken for largest gain from trough to peak

        Parameters
        ----------
        df : DataFrame
            The core OHLC DataFrame.

        Returns
        -------
        gain_length : Int
            The number of days from trough to peak.

        """
        # Extract the largest pnl gain
        max_gain_val = max(df['max_gain'])
        
        # Reverse the max gain column in time
        gain_rev = df['max_gain'][::-1]
        
        # Find the location of the max gain
        max_gain_loc = gain_rev.index.get_loc(
            gain_rev[gain_rev==max_gain_val].index[0])
        
        # Calculate the time taken from zero to this max gain
        gain_length = (gain_rev[max_gain_loc:].values == 0).argmax()
        
        return gain_length


    @staticmethod
    def _create_monthly_data(df, equity):
        """
        Create monthly summary data 

        Parameters
        ----------
        df : DataFrame
            The OHLC data.
        equity : Int
            The initial equity level.

        Returns
        -------
        monthly_data : DataFrame
            The monthly summary data.

        """
        # Create empty DataFrame
        monthly_data = pd.DataFrame()
        
        # Summarize daily pnl data by resampling to monthly
        monthly_data['total_net_profit'] = df['daily_pnl'].resample('1M').sum()
        monthly_data['average_net_profit'] = df[
            'daily_pnl'].resample('1M').mean()
        monthly_data['max_net_profit'] = df['daily_pnl'].resample('1M').max()
        monthly_data['min_net_profit'] = df['daily_pnl'].resample('1M').min()
        
        # Create arrays of zeros to hold data
        monthly_data['beginning_equity'] = np.array([0.0]*len(monthly_data))
        monthly_data['additions'] = np.array([0.0]*len(monthly_data))
        monthly_data['withdrawals'] = np.array([0.0]*len(monthly_data))    
        monthly_data['end_equity'] = np.array([0.0]*len(monthly_data))
        monthly_data['return'] = np.array([0.0]*len(monthly_data))
        monthly_data['beginning_equity_raw'] = np.array([0.0]*len(monthly_data))
        monthly_data['end_equity_raw'] = np.array([0.0]*len(monthly_data))
        monthly_data['return_raw'] = np.array([0.0]*len(monthly_data))        
        monthly_data['abs_loss'] = np.array([0.0]*len(monthly_data))
        monthly_data['abs_loss_raw'] = np.array([0.0]*len(monthly_data))

        # Set initial values        
        monthly_data['additions'].iloc[0] = equity
        monthly_data['end_equity'].iloc[0] = (
            monthly_data['beginning_equity'].iloc[0]
            + monthly_data['additions'].iloc[0]
            + monthly_data['withdrawals'].iloc[0]
            + monthly_data['total_net_profit'].iloc[0])
        monthly_data['return'].iloc[0] = (
            (monthly_data['total_net_profit'].iloc[0]) 
            / (monthly_data['beginning_equity'].iloc[0]
               + monthly_data['additions'].iloc[0]
               + monthly_data['withdrawals'].iloc[0]))
        monthly_data['beginning_equity_raw'].iloc[0] = equity
        monthly_data['end_equity_raw'].iloc[0] = (
            monthly_data['beginning_equity_raw'].iloc[0]
            + monthly_data['total_net_profit'].iloc[0])
        monthly_data['return_raw'].iloc[0] = (
            (monthly_data['total_net_profit'].iloc[0]) 
            / (monthly_data['beginning_equity_raw'].iloc[0]))
    
        # For each month
        for row in range(1, len(monthly_data)):
            
            # Beginning equity is the closing equity from the prior period
            # Raw data keeps all the profits invested whereas the other resets 
            # the equity balance each year
            monthly_data['beginning_equity'][row] = monthly_data[
                'end_equity'][row-1]
            monthly_data['beginning_equity_raw'][row] = monthly_data[
                'end_equity_raw'][row-1]
            
            # For each change in year
            if monthly_data.index.year[row] != monthly_data.index.year[row-1]:
                
                # If the end of year equity level is less than the initial 
                # level
                if monthly_data['end_equity'][row-1] < equity:
                    
                    # Add back the difference to additions
                    monthly_data['additions'][row] = (
                        equity - monthly_data['end_equity'][row-1])
                else:
                    # Otherwise subtract from withdrawals
                    monthly_data['withdrawals'][row] = (
                        equity - monthly_data['end_equity'][row-1])
            
            # Ending equity is the beginning equity plus the sum of additions, 
            # withdrawals and the net pnl over the period        
            monthly_data['end_equity'][row] = (
                monthly_data['beginning_equity'][row] 
                + monthly_data['total_net_profit'][row]
                + monthly_data['additions'][row]
                + monthly_data['withdrawals'][row])
            
            # Ending equity raw is the beginning equity plus the net pnl over 
            # the period
            monthly_data['end_equity_raw'][row] = (
                monthly_data['beginning_equity_raw'][row] 
                + monthly_data['total_net_profit'][row])
            
            # Monthly return is the net pnl over the period divided by the 
            # beginning equity plus the sum of additions and withdrawals 
            monthly_data['return'][row] = (
                (monthly_data['total_net_profit'][row]) 
                / (monthly_data['beginning_equity'][row] 
                   + monthly_data['additions'][row]
                   + monthly_data['withdrawals'][row]))
            
            # Monthly return raw is the net pnl over the period divided by the 
            # beginning equity raw
            monthly_data['return_raw'][row] = (
                (monthly_data['total_net_profit'][row]) 
                / (monthly_data['beginning_equity_raw'][row]))
    
            # For use in Gain to Pain Ratio, absolute loss is the positive 
            # value of the negative monthly returns
            if monthly_data['return'][row] < 0:
                monthly_data['abs_loss'][row] = -monthly_data['return'][row]
            
            # For use in Gain to Pain Ratio, absolute loss is the positive 
            # value of the negative monthly returns
            if monthly_data['return_raw'][row] < 0:
                monthly_data['abs_loss_raw'][row] = -monthly_data[
                    'return_raw'][row]
                   
        return monthly_data