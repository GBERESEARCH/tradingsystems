# tradingsystems
## End of day backtesting of technical trading rules

&nbsp;  

### *** Still under development ***
#### Performance data requires additional fields populated
#### Currently only 3MA and 4MA strategies implemented

&nbsp;  

### Installation
Install from PyPI:
```
$ pip install tradingsystems
```

&nbsp;

To install in new environment using anaconda:
```
$ conda create --name systems
```
Activate new environment
```
$ activate systems
```
Install Python
```
(systems) $ conda install python==3.8.8
```
Install Spyder
```
(systems) $ conda install spyder==4.2.5
```
Install Pandas
```
(systems) $ conda install pandas==1.1.4
```


Install tradingsystems
```
(systems) $ python -m pip install tradingsystems
```

&nbsp;

### Setup

Import tradingsystems and initialize a Data object 
```
import tradingsystems.systems
data = systems.Data()
```
Run the Triple MA Cross strategy against Eurostoxx Index data.
```
data.test_strategy_3MA(ticker='$STOXX50')
```

&nbsp;

### Output

A table of results is printed to the console

![performance_analysis](images/performance_analysis.png)

&nbsp;

The following volumes served as a reference for some of the methods and report design:
* [Design, Testing, and Optimization of Trading Systems, Robert Pardo]
* [Technical Traders Guide to Computer Analysis of the Futures Markets, Charles LeBeau & David W. Lucas]

[Design, Testing, and Optimization of Trading Systems, Robert Pardo]:
<https://www.amazon.co.uk/Testing-Optimization-Trading-Trader′s-Exchange/dp/0471554464/>  

[Technical Traders Guide to Computer Analysis of the Futures Markets, Charles LeBeau & David W. Lucas]:
<https://www.amazon.co.uk/Technical-Traders-Computer-Analysis-Futures/dp/1556234686/>

