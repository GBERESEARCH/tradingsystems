"""
Unit tests for tradingsystems

"""

import unittest
from tradingsystems.systems import TestStrategy

class TradingSystemNorgateTickerTestCase(unittest.TestCase):
    """
    Unit tests for Norgate tickers

    """

    def test_strat(self):
        """
        Unit test for no parameters provided.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for no parameters provided" )

        strat = TestStrategy()
        strat_dict = strat.__dict__
        ticker = strat_dict['params']['ticker']
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, 0)
        print("Length of price history for ", ticker, ": ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, 0)
        print("Length of benchmark history for ", ticker, ": ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ticker, ": ",
              strat_monthly_data_length, " monthly")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ticker, ": ",
              strat_params_length)



    def test_brent(self):
        """
        Unit test for brent crude.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for brent crude - Norgate" )

        ticker='&BRN'
        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type
            )

        strat_dict = strat.__dict__
        ticker = strat_dict['params']['ticker']
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, lookback)
        print("Length of price history for ", ticker, ": ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, lookback)
        print("Length of benchmark history for ", ticker, ": ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ticker, ": ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ticker, ": ",
              strat_params_length)



    def test_pall(self):
        """
        Unit test for palladium.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for palladium - Norgate" )

        ticker='&PA'
        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type
            )

        strat_dict = strat.__dict__
        ticker = strat_dict['params']['ticker']
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, lookback)
        print("Length of price history for ", ticker, ": ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, lookback)
        print("Length of benchmark history for ", ticker, ": ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ticker, ": ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ticker, ": ",
              strat_params_length)



    def test_ftse(self):
        """
        Unit test for FTSE100.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for FTSE100 - Norgate" )

        ticker='$FT100'
        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type
            )

        strat_dict = strat.__dict__
        ticker = strat_dict['params']['ticker']
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, lookback)
        print("Length of price history for ", ticker, ": ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, lookback)
        print("Length of benchmark history for ", ticker, ": ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ticker, ": ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ticker, ": ",
              strat_params_length)



    def test_ten_year(self):
        """
        Unit test for 10 Year Treasury.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for 10 Year Treasury - Norgate" )

        ticker='%TNX'
        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type
            )

        strat_dict = strat.__dict__
        ticker = strat_dict['params']['ticker']
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, lookback)
        print("Length of price history for ", ticker, ": ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, lookback)
        print("Length of benchmark history for ", ticker, ": ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ticker, ": ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ticker, ": ",
              strat_params_length)



    def test_iron_ore(self):
        """
        Unit test for Iron Ore.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for Iron Ore - Norgate" )

        ticker='%TNX'
        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type
            )

        strat_dict = strat.__dict__
        ticker = strat_dict['params']['ticker']
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, lookback)
        print("Length of price history for ", ticker, ": ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, lookback)
        print("Length of benchmark history for ", ticker, ": ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ticker, ": ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ticker, ": ",
              strat_params_length)



class TradingSystemYahooTickerTestCase(unittest.TestCase):
    """
    Unit tests for Yahoo tickers

    """

    def test_aapl(self):
        """
        Unit test for aapl.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for aapl - Yahoo" )

        ticker='AAPL'
        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            ticker_source='yahoo',
            bench_source='yahoo'
            )

        strat_dict = strat.__dict__
        ticker = strat_dict['params']['ticker']
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, lookback)
        print("Length of price history for ", ticker, ": ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, lookback)
        print("Length of benchmark history for ", ticker, ": ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ticker, ": ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ticker, ": ",
              strat_params_length)



    def test_tsla(self):
        """
        Unit test for tsla.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for tsla - Yahoo" )

        ticker='TSLA'
        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            ticker_source='yahoo'
            )

        strat_dict = strat.__dict__
        ticker = strat_dict['params']['ticker']
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, lookback)
        print("Length of price history for ", ticker, ": ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, lookback)
        print("Length of benchmark history for ", ticker, ": ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ticker, ": ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ticker, ": ",
              strat_params_length)



    def test_msft(self):
        """
        Unit test for msft.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for msft - Yahoo" )

        ticker='MSFT'
        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            ticker_source='yahoo'
            )

        strat_dict = strat.__dict__
        ticker = strat_dict['params']['ticker']
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, lookback)
        print("Length of price history for ", ticker, ": ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, lookback)
        print("Length of benchmark history for ", ticker, ": ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ticker, ": ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ticker, ": ",
              strat_params_length)



class TradingSystemAlphaTickerTestCase(unittest.TestCase):
    """
    Unit tests for Alphavantage tickers

    """

    def test_gbpusd(self):
        """
        Unit test for gbpusd.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for gbpusd - Alphavantage" )

        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'
        ccy_1='GBP'
        ccy_2='USD'

        strat = TestStrategy(
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            ticker_source='alpha',
            asset_type='fx',
            ccy_1=ccy_1,
            ccy_2=ccy_2
            )

        strat_dict = strat.__dict__
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, lookback)
        print("Length of price history for ", ccy_1, ccy_2, " alpha: ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, lookback)
        print("Length of benchmark history for ", ccy_1, ccy_2, " alpha: ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ccy_1, ccy_2, " alpha: ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ccy_1, ccy_2, " alpha: ",
              strat_params_length)


    def test_btcusd(self):
        """
        Unit test for btcusd.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for btcusd - Alphavantage" )

        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'
        ccy_1='BTC'
        ccy_2='USD'

        strat = TestStrategy(
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            ticker_source='alpha',
            asset_type='crypto',
            ccy_1=ccy_1,
            ccy_2=ccy_2
            )

        strat_dict = strat.__dict__
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, 0)
        print("Length of price history for ", ccy_1, ccy_2, " alpha: ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, 0)
        print("Length of benchmark history for ", ccy_1, ccy_2, " alpha: ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ccy_1, ccy_2, " alpha: ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ccy_1, ccy_2, " alpha: ",
              strat_params_length)


    def test_msft(self):
        """
        Unit test for msft.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for msft - Alphavantage" )

        ticker='MSFT'
        lookback = 1500
        entry_type='4ma'
        exit_type='sar'
        stop_type='trail_close'

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            ticker_source='alpha'
            )

        strat_dict = strat.__dict__
        ticker = strat_dict['params']['ticker']
        strat_price_length = len(strat_dict['tables']['prices'])
        strat_benchmark_length = len(strat_dict['tables']['benchmark'])
        strat_monthly_data_length = len(strat_dict['tables']['monthly_data'])
        strat_params_length = len(strat_dict['params'])

        self.assertGreater(strat_price_length, lookback)
        print("Length of price history for ", ticker, " alpha: ",
              strat_price_length, " days")

        self.assertGreater(strat_benchmark_length, lookback)
        print("Length of benchmark history for ", ticker, " alpha: ",
              strat_benchmark_length, " days")

        self.assertGreater(strat_monthly_data_length, 0)
        print("Length of monthly history for ", ticker, " alpha: ",
              strat_monthly_data_length, " months")

        self.assertGreater(strat_params_length, 0)
        print("Length of parameter dictionary for ", ticker, " alpha: ",
              strat_params_length)


if __name__ == '__main__':
    unittest.main()
