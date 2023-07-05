"""
Unit tests for tradingsystems

"""

import unittest
from tradingsystems.systems import TestStrategy

class TradingSystemExitTestCase(unittest.TestCase):
    """
    Unit tests for Entry Signals

    """


    def test_sar(self):
        """
        Unit test for parabolic sar exit.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for parabolic sar exit" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type='sar'
        exit_amount=5000
        stop_type=None#'trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            exit_amount=exit_amount,
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



    def test_sup_res(self):
        """
        Unit test for Support / Resistance exit.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for Support / Resistance exit" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type='sup_res'
        exit_amount=5000
        stop_type=None#'trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            exit_amount=exit_amount,
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



    def test_rsi_trail(self):
        """
        Unit test for trailing rsi exit.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for trailing rsi exit" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type='rsi_trail'
        exit_amount=5000
        stop_type=None#'trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            exit_amount=exit_amount,
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



    def test_key_reversal(self):
        """
        Unit test for key reversal day exit.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for key reversal day exit" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type='key_reversal'
        exit_amount=5000
        stop_type=None#'trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            exit_amount=exit_amount,
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



    def test_volatility(self):
        """
        Unit test for volatility exit.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for volatility exit" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type='volatility'
        exit_amount=5000
        stop_type=None#'trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            exit_amount=exit_amount,
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



    def test_stoch_cross(self):
        """
        Unit test for stochastic crossover exit.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for stochastic crossover exit" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type='stoch_cross'
        exit_amount=5000
        stop_type=None#'trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            exit_amount=exit_amount,
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



    def test_nday_range(self):
        """
        Unit test for n-day range exit.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for n-day range exit" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type='nday_range'
        exit_amount=5000
        stop_type=None#'trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            exit_amount=exit_amount,
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



    def test_random(self):
        """
        Unit test for random exit.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for random exit" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type='random'
        exit_amount=5000
        stop_type=None#'trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            exit_amount=exit_amount,
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



    def test_trailing_stop(self):
        """
        Unit test for trailing stop exit.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for trailing stop exit" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type='trailing_stop'
        exit_amount=5000
        stop_type=None#'trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            exit_amount=exit_amount,
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



    def test_profit_target(self):
        """
        Unit test for profit_target exit.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for profit target exit" )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type='profit_target'
        exit_amount=5000
        stop_type=None#'trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            entry_type=entry_type,
            exit_type=exit_type,
            exit_amount=exit_amount,
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
