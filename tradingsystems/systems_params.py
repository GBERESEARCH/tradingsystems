# Dictionary containing all the default parameters
system_params_dict = {
    'df_profit_factor':2,
    'df_profit_bars':2,
    'df_stop_bars':2,
    'df_exit_factor':2,
    'df_initial_equity':100000,
    
    'df_ticker':'$SPX',
    'df_lookback':500,
    'df_short_ma':4,
    'df_medium_ma':9,
    'df_long_ma':18,
    'df_ma_1':5,
    'df_ma_2':12,
    'df_ma_3':20,
    'df_ma_4':40,
    'df_position_size':100,
    'df_source':'norgate',
    'df_slippage':0.05,
    'df_commission':0.00,
    'df_strategy':'3MA',
    
    # lists of parameters for each of the trend flags calculated 
    # in fields function
    'df_ma_list':[10, 14, 20, 30, 50, 100, 200],
    'df_macd_params':[12, 26, 9],
    'df_adx_list':[10, 14, 20, 30, 50, 100, 200],
    'df_ma_cross_list':[(10, 30), (20, 50), (50, 200)],
    'df_price_cross_list':[10, 14, 20, 30, 50, 100, 200],
    'df_rsi_list':[10, 20, 30, 50, 100, 200],
    'df_atr_list':[5, 7, 10, 12, 14, 20, 30, 50, 100, 200],
    'df_range_percent_list':[0.5, 0.6, 0.7],
    'df_range_bar_list':[1, 2, 3, 4, 5],
    'df_narrow_range_list':[4, 5, 10],
    'df_high_day_list':[5, 10, 20, 50, 100, 200, 250]}

