# Dictionary containing all the default parameters
system_params_dict = {
    'df_profit_factor':2,
    'df_profit_bars':2,
    'df_stop_bars':2,
    'df_exit_factor':2,
    'df_equity':100000.00,    
    'df_ticker':'$SPX',
    'df_bench_ticker':'$SPX',
    'df_ccy_1':'GBP',
    'df_ccy_2':'USD',
    'df_asset_type':'fx',
    'df_lookback':750,
    'df_ma1':5,
    'df_ma2':12,
    'df_ma3':20,
    'df_ma4':40,
    'df_simple_ma':True,
    'df_position_size':100,
    'df_pos_size_fixed':True,
    'df_ticker_source':'norgate',
    'df_bench_source':'norgate',
    'df_slippage':5,
    'df_commission':0.00,
    'df_riskfree':0.0025,
    'df_entry_type':'2ma',
    'df_exit_type':'trailing_stop',
    'df_stop_type':'initial_dollar',
    'df_entry_period':14,
    'df_exit_period':5,
    'df_stop_period':5, 
    'df_entry_threshold':0,
    'df_exit_threshold':0,
    'df_exit_amount':3000.00,
    'df_stop_amount':1500.00,
    'df_entry_oversold':25,
    'df_entry_overbought':75,
    'df_exit_oversold':25,
    'df_exit_overbought':75,
    'df_entry_acceleration_factor':0.02,    
    'df_exit_acceleration_factor':0.02,
    'df_sip_price':False,
    'df_signals':True,
    
    # Columns names used for strategy indicators 
    'df_entry_signal_indicators':{'2ma':('ma_1','ma_2'),
                                  '3ma':('ma_1','ma_2', 'ma_3'),
                                  '4ma':('ma_1','ma_2', 'ma_3', 'ma_4'),
                                  'sar':'sar_entry',
                                  'channel_breakout':(
                                      'rolling_low_close_entry', 
                                      'rolling_high_close_entry'),
                                  'stoch_cross':(
                                      'slow_k_entry', 'slow_d_entry'),
                                  'stoch_over_under':(
                                      'slow_k_entry', 'slow_d_entry'),
                                  'stoch_pop':(
                                      'slow_k_entry', 'slow_d_entry'),
                                  'rsi':'RSI_entry',
                                  'cci':'CCI_entry',
                                  'momentum':'momentum',
                                  'volatility':'ATR_entry'},
    
    # Signals requiring changes to default parameters
    'df_entry_signal_labels':{'2ma':('Double ','MA Crossover'),
                              '3ma':('Triple ','MA Crossover'),
                              '4ma':('Quad ','MA Crossover'),
                              'sar':'Parabolic SAR',
                              'channel_breakout':'Channel Breakout',
                              'stoch_cross':'Stochastic Crossover',
                              'stoch_over_under':'Stochastic Over Under',
                              'stoch_pop':'Stochastic Pop',
                              'rsi':'Relative Strength Index',
                              'cci':'Commodity Channel Index',
                              'momentum':'Momentum',
                              'volatility':'Volatility'},
    
    'df_exit_signal_labels':{'sar':'Parabolic SAR',
                             'sup_res':'Support / Resistance',
                             'rsi_trail':'Trailing Relative Strength Index',
                             'key_reversal':'Key Reversal Day',
                             'trailing_stop':'Trailing Stop',
                             'volatility':'Volatility',
                             'stoch_cross':'Stochastic Crossover',
                             'profit_target':'Profit Target',
                             'nday_range':'High/Low Range',
                             'random':'Random'},
                                 
    'df_stop_signal_labels':{'initial_dollar':'Initial Dollar',
                             'sup_res':'Support / Resistance',
                             'immediate_profit':'Immediate Profit',
                             'breakeven':'Breakeven',
                             'trail_close':'Trailing Close Stop',
                             'trail_high_low':'Trailing High / Low Stop'},
 
    # Signal parameter values differing from standard defaults
    'df_entry_signal_dict':{'2ma':{'ma1':9,
                                   'ma2':18},
                            '3ma':{'ma1':4,
                                   'ma2':9,
                                   'ma3':18},
                            '4ma':{'ma1':5,
                                   'ma2':12,
                                   'ma3':20,
                                   'ma4':40},
                            'sar':{'entry_acceleration_factor':0.02},
                            'channel_breakout':{'entry_period':10},
                            'stoch_cross':{'entry_period':14,
                                           'entry_oversold':25,
                                           'entry_overbought':75},
                            'stoch_over_under':{'entry_period':14,
                                                'entry_oversold':25,
                                                'entry_overbought':75},
                            'stoch_pop':{'entry_period':14,
                                         'entry_oversold':25,
                                         'entry_overbought':75},
                            'rsi':{'entry_period':14,
                                   'entry_oversold':30,
                                   'entry_overbought':70},
                            'cci':{'entry_period':20,
                                   'entry_threshold':0,
                                   'entry_oversold':-100,
                                   'entry_overbought':100},
                            'momentum':{'entry_period':10,
                                        'entry_threshold':0},
                            'volatility':{'entry_period':10,
                                          'entry_threshold':1.5}},
     
    'df_exit_signal_dict':{'sar':{'exit_period':5,
                                  'exit_acceleration_factor':0.02,
                                  'sip_price':True},
                           'sup_res':{'exit_period':5},
                           'rsi_trail':{'exit_period':9,
                                        'exit_oversold':25,
                                        'entry_overbought':75},
                           'key_reversal':{'exit_period':10},
                           'trailing_stop':{'exit_amount':3000.00},
                           'volatility':{'exit_period':5,
                                         'exit_threshold':1},
                           'stoch_cross':{'exit_period':14,
                                          'exit_oversold':25,
                                          'entry_overbought':75},
                           'profit_target':{'exit_amount':3000.00},
                           'nday_range':{'exit_period':10}},

    'df_stop_signal_dict':{'initial_dollar':{'stop_amount':1500.00},
                           'sup_res':{'stop_period':20},
                           'immediate_profit':{'stop_period':5},
                           'breakeven':{'stop_amount':1500.00},
                           'trail_close':{'stop_amount':1500.00}, 
                           'trail_high_low':{'stop_amount':1500.00}}
    }

