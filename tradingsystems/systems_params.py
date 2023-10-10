"""
Dictionary containing all the default parameters
"""

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
    'df_start_date':None,
    'df_end_date':None,
    'df_lookback':750,
    'df_ma1':10,
    'df_ma2':20,
    'df_ma3':50,
    'df_ma4':100,
    'df_simple_ma':True,
    'df_fixed_pos_size':5,
    'df_atr_pos_size':14,
    'df_position_type':'atr',
    'df_margin_%':0.2,
    'df_ticker_source':'norgate',
    'df_bench_source':'norgate',
    'df_slippage':5.0,
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
    'df_api_key':'',
    'df_position_risk_bps':500,
    'df_equity_inv_perc':0.75,

    # Columns names used for strategy indicators
    'df_entry_signal_indicators':{
        '2ma':('ma_1','ma_2'),
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
        'adx':(
            'ADX_entry', 'DI_plus_entry', 'DI_minus_entry'),
        'macd':(
            'MACD_entry', 'MACD_Signal_entry', 'MACD_Hist_entry'),
        'cci':'CCI_entry',
        'momentum':'momentum',
        'volatility':'ATR_entry'
        },

    # Signals requiring changes to default parameters
    'df_entry_signal_labels':{
        '2ma':('Double ','MA Crossover'),
        '3ma':('Triple ','MA Crossover'),
        '4ma':('Quad ','MA Crossover'),
        'sar':'Parabolic SAR',
        'channel_breakout':'Channel Breakout',
        'stoch_cross':'Stochastic Crossover',
        'stoch_over_under':'Stochastic Over Under',
        'stoch_pop':'Stochastic Pop',
        'rsi':'Relative Strength Index',
        'adx':'ADX',
        'macd':'MACD',
        'cci':'Commodity Channel Index',
        'momentum':'Momentum',
        'volatility':'Volatility'
        },

    'df_exit_signal_labels':{
        'sar':'Parabolic SAR',
        'sup_res':'Support / Resistance',
        'rsi_trail':'Trailing Relative Strength Index',
        'key_reversal':'Key Reversal Day',
        'trailing_stop':'Trailing Stop',
        'volatility':'Volatility',
        'stoch_cross':'Stochastic Crossover',
        'profit_target':'Profit Target',
        'nday_range':'High/Low Range',
        'random':'Random'
        },

    'df_stop_signal_labels':{
        'initial_dollar':'Initial Dollar',
        'sup_res':'Support / Resistance',
        'immediate_profit':'Immediate Profit',
        'breakeven':'Breakeven',
        'trail_close':'Trailing Close Stop',
        'trail_high_low':'Trailing High / Low Stop'
        },

    # Signal parameter values differing from standard defaults
    'df_entry_signal_dict':{
        '2ma':{
            'ma1':9,
            'ma2':18
            },
        '3ma':{
            'ma1':4,
            'ma2':9,
            'ma3':18
            },
        '4ma':{
            'ma1':10,
            'ma2':20,
            'ma3':50,
            'ma4':100
            },
        'sar':{
            'entry_acceleration_factor':0.02
            },
        'channel_breakout':{
            'entry_period':10
            },
        'stoch_cross':{
            'entry_period':14,
            'entry_oversold':25,
            'entry_overbought':75
            },
        'stoch_over_under':{
            'entry_period':14,
            'entry_oversold':25,
            'entry_overbought':75
            },
        'stoch_pop':{
            'entry_period':14,
            'entry_oversold':25,
            'entry_overbought':75
            },
        'rsi':{
            'entry_period':14,
            'entry_oversold':30,
            'entry_overbought':70
            },
        'adx':{
            'adx_threshold':25
            },
        'macd':{
            'macd_params':(12, 26, 9)
            },
        'cci':{
            'entry_period':20,
            'entry_threshold':0,
            'entry_oversold':-100,
            'entry_overbought':100
            },
        'momentum':{
            'entry_period':10,
            'entry_threshold':0
            },
        'volatility':{
            'entry_period':10,
            'entry_threshold':1.5
            }
        },

    'df_exit_signal_dict':{
        'sar':{
            'exit_period':5,
            'exit_acceleration_factor':0.02,
            'sip_price':True
            },
        'sup_res':{
            'exit_period':5
            },
        'rsi_trail':{
            'exit_period':9,
            'exit_oversold':25,
            'entry_overbought':75
            },
        'key_reversal':{
            'exit_period':10
            },        
        'volatility':{
            'exit_period':5,
            'exit_threshold':1
            },
        'stoch_cross':{
            'exit_period':14,
            'exit_oversold':25,
            'entry_overbought':75
            },
        'nday_range':{
            'exit_period':10
            },
        'random':{},    
        'trailing_stop':{
            'exit_amount':3000.00
            },
        'profit_target':{
            'exit_amount':3000.00
            }
        },

    'df_stop_signal_dict':{
        'sup_res':{
            'stop_period':20
            },
        'immediate_profit':{
            'stop_period':5
            },
        'initial_dollar':{
            'stop_amount':1500.00
            },
        'breakeven':{
            'stop_amount':1500.00
            },
        'trail_close':{
            'stop_amount':1500.00
            },
        'trail_high_low':{
            'stop_amount':1500.00
            }
        },

    'df_params':{
        'equity':100000.00,
        'ticker':'$SPX',
        'bench_ticker':'$SPX',
        'ccy_1':'GBP',
        'ccy_2':'USD',
        'asset_type':'fx',
        'start_date':None,
        'end_date':None,
        'lookback':750,
        'ma1':10,
        'ma2':20,
        'ma3':50,
        'ma4':100,
        'simple_ma':True,
        'return_data':False,
        'refresh_data':True,
        'input_data':'refresh',
        'fixed_pos_size':5,
        'atr_pos_size':14,
        'position_type':'atr',
        'ticker_source':'norgate',
        'bench_source':'norgate',
        'equity_source':'yahoo',
        'slippage':5.0,
        'commission':0.00,
        'riskfree':0.0025,
        'entry_type':'2ma',
        'exit_type':'trailing_stop',
        'stop_type':'initial_dollar',
        'entry_period':14,
        'exit_period':5,
        'stop_period':5,
        'entry_threshold':0,
        'exit_threshold':0,
        'exit_amount':3000.00,
        'stop_amount':1500.00,
        'entry_oversold':25,
        'entry_overbought':75,
        'exit_oversold':25,
        'exit_overbought':75,
        'entry_acceleration_factor':0.02,
        'exit_acceleration_factor':0.02,
        'adx_threshold':25,
        'macd_params':(12, 26, 9),
        'sip_price':False,
        'signals':True,
        'api_key':'',
        'position_risk_bps':500,
        'equity_inv_perc':0.75,
        'margin_%':0.2,
        'contract_months':{
            '01':'F',
            '02':'G',
            '03':'H',
            '04':'J',
            '05':'K',
            '06':'M',
            '07':'N',
            '08':'Q',
            '09':'U',
            '10':'V',
            '11':'X',
            '12':'Z'
        }},

    'df_contract_months':{
        '01':'F',
        '02':'G',
        '03':'H',
        '04':'J',
        '05':'K',
        '06':'M',
        '07':'N',
        '08':'Q',
        '09':'U',
        '10':'V',
        '11':'X',
        '12':'Z'
        }
    }
