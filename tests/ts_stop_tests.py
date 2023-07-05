"""
Unit tests for tradingsystems

"""

import unittest
from tradingsystems.systems import TestStrategy

class TradingSystemExitTestCase(unittest.TestCase):
    """
    Unit tests for Entry Signals

    """


    def test_sup_res(self):
        """
        Unit test for Support / Resistance stop.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for Support / Resistance stop" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type=None
        stop_type='sup_res'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            stop_amount=stop_amount
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



    def test_immediate_profit(self):
        """
        Unit test for immediate profit stop.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for immediate profit stop" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type=None
        stop_type='immediate_profit'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            stop_amount=stop_amount
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



    def test_initial_dollar(self):
        """
        Unit test for initial dollar stop.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for initial dollar stop" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type=None
        stop_type='initial_dollar'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            stop_amount=stop_amount
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



    def test_breakeven(self):
        """
        Unit test for breakeven stop.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for breakeven stop" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type=None
        stop_type='breakeven'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            stop_amount=stop_amount
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



    def test_trail_close(self):
        """
        Unit test for trailing stop based on close.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for trailing stop based on close" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type=None
        stop_type='trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            stop_amount=stop_amount
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



    def test_trail_high_low(self):
        """
        Unit test for trailing stop based on high / low price.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for trailing stop based on high / low price" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type=None
        stop_type='trail_high_low'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            stop_type=stop_type,
            stop_amount=stop_amount
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


if __name__ == '__main__':
    unittest.main()
