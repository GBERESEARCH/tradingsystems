# tradingsystems
## End of day backtesting of Technical trading rules

&nbsp;

### Entry strategies: 
  - Double MA Crossover
  - Triple MA Crossover
  - Quad MA Crossover
  - Parabolic SAR
  - Channel Breakout
  - Stochastic Crossover
  - Stochastic Over Under
  - Stochastic Pop
  - Relative Strength Index (RSI)
  - Average Directional Index (ADX)
  - Moving Average Convergence Divergence (MACD)
  - Commodity Channel Index (CCI)
  - Momentum
  - Volatility

&nbsp;

### Exit strategies:
  - Parabolic SAR
  - Support / Resistance
  - Trailing Relative Strength Index
  - Key Reversal Day
  - Trailing Stop
  - Volatility
  - Stochastic Crossover
  - Profit Target
  - High/Low Range
  - Random

&nbsp;

### Stop strategies:
  - Initial Dollar
  - Support / Resistance
  - Immediate Profit
  - Breakeven
  - Trailing Close Stop
  - Trailing High / Low Stop

&nbsp;

### Data Sources:
  - Norgate Data: Commodities (requires a [Futures package] subscription) 
  - Alpha Vantage: Cash Equities, FX, Crypto (requires an [Alpha Vantage API key])
  - Yahoo Finance: US Cash Equities

&nbsp;

### Installation
Install from PyPI:
```
$ pip install tradingsystems
```

&nbsp;

Install in a new environment using Python venv:

Create base environment of Python 3.13
```
$ py -3.13 -m venv .venv
```
Activate new environment
```
$ .venv\scripts\activate
```
Ensure pip is up to date
``` 
$ (.venv) python -m pip install --upgrade pip
```
Install Spyder
```
$ (.venv) python -m pip install spyder
```
Install package
```
$ (.venv) python -m pip install tradingsystems
```

&nbsp;

### Setup

Import tradingsystems
```
from tradingsystems.systems import TestStrategy
```
Run the Quad MA Cross entry strategy for 6 years against Brent Crude futures with a Parabolic SAR exit and a Trailing Close Stop. Typing ```help(TestStrategy)``` gives a list of all the optional arguments.
```
strat = TestStrategy(ticker='&BRN', lookback=1500, entry_type='4ma', exit_type='sar', stop_type='trail_close')
```

&nbsp;

### Output

A table of results is printed to the console

![Brent_6y_4MA_report](images/Brent_6y_4MA_report.png)

&nbsp;

And a graph of the performance of the strategy with trade signals and the equity curve.

![Brent_6y_4MA_graph](images/Brent_6y_4MA_graph.png)

&nbsp;

Nickel with a 14 Day Parabolic SAR entry:

```
strat = TestStrategy(ticker='@NI', entry_type='sar', exit_amount=5000, stop_amount=3000)
```

![NI_3y_SAR_report](images/NI_3y_SAR_report.png)

&nbsp;

Display a graph of the performance of the strategy with trade signals, the momentum indicator and the equity curve.

![NI_3y_SAR_graph](images/NI_3y_SAR_graph.png)

&nbsp;

Platinum with a 20 Day Momentum entry:

```
strat = TestStrategy(ticker='&PL', lookback=750, entry_type='momentum', entry_period=20)
```

![Platinum_3y_Momentum_report](images/Platinum_3y_Momentum_report.png)

&nbsp;

Display a graph of the performance of the strategy with trade signals, the momentum indicator and the equity curve.

![Platinum_3y_Momentum_graph](images/Platinum_3y_Momentum_graph.png)

&nbsp;

Apple with a 20 Day Channel Breakout entry:

```
strat = TestStrategy(ticker='AAPL', start_date='2016-01-01', entry_type='channel_breakout', entry_period=20, ticker_source='yahoo')
```

![AAPL_5y_Channel_report](images/AAPL_5y_Channel_report.png)

&nbsp;

Display a graph of the performance of the strategy with trade signals and the equity curve.

![AAPL_5y_Channel_graph](images/AAPL_5y_Channel_graph.png)


&nbsp;

Wheat with a 20 Day Stochastic Pop entry:

```
strat = TestStrategy(ticker='&KE', lookback=750, entry_type='stoch_pop', entry_period=20)
```

![Wheat_3y_stoch_pop_report](images/Wheat_3y_stoch_pop_report.png)

&nbsp;

Display a graph of the performance of the strategy with trade signals, the slow-k and slow-d stochastics and the equity curve.

![Wheat_3y_stoch_pop_graph](images/Wheat_3y_stoch_pop_graph.png)

&nbsp;

GBPUSD with a 10 Day 150% Volatility entry:

```
strat = TestStrategy(ccy_1='GBP', ccy_2='USD', lookback=750, ticker_source='alpha', asset_type='fx', entry_type='volatility', exit_amount=5000, stop_amount=2000)
```

![GBPUSD_3y_volatility_report](images/GBPUSD_3y_volatility_report.png)

&nbsp;

Display a graph of the performance of the strategy with trade signals, the 10 Day ATR and the equity curve.


![GBPUSD_3y_volatility_graph](images/GBPUSD_3y_volatility_graph.png)

&nbsp;

Changes to the backtest parameters can be made by running the following with the
relevant key word arguments:

```
strat.run_backtest(**kwargs)
```
&nbsp;

The performance report can be printed with:

```
strat.performance_report()
```
&nbsp;

And the graph of the strategy performance:
```
strat.performance_graph()
```
&nbsp;

The following volumes served as a reference for some of the methods and report design:
* [Design, Testing, and Optimization of Trading Systems, Robert Pardo]
* [Technical Traders Guide to Computer Analysis of the Futures Markets, Charles LeBeau & David W. Lucas]

[Design, Testing, and Optimization of Trading Systems, Robert Pardo]:
<https://www.amazon.co.uk/Testing-Optimization-Trading-Trader′s-Exchange/dp/0471554464/>  

[Technical Traders Guide to Computer Analysis of the Futures Markets, Charles LeBeau & David W. Lucas]:
<https://www.amazon.co.uk/Technical-Traders-Computer-Analysis-Futures/dp/1556234686/>

[Futures package]:
<https://norgatedata.com/futurespackage.php>

[Alpha Vantage API key]:
<https://www.alphavantage.co/support/#api-key>