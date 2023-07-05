"""
Unit tests for tradingsystems

"""

import unittest
from tradingsystems.systems import TestStrategy

class TradingSystemEntryTestCase(unittest.TestCase):
    """
    Unit tests for Entry Signals

    """


    def test_double_ma(self):
        """
        Unit test for double moving average entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for double moving average entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='2ma'
        exit_type=None#'nday_range'
        stop_type='trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            ma1=20,
            ma2=50,
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



    def test_triple_ma(self):
        """
        Unit test for triple moving average entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for triple moving average entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='3ma'
        exit_type=None#'nday_range'
        stop_type='trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            ma1=20,
            ma2=50,
            ma3=100,
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



    def test_quad_ma(self):
        """
        Unit test for quad moving average entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for quad moving average entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='4ma'
        exit_type=None#'nday_range'
        stop_type='trail_close'
        stop_amount=5000

        strat = TestStrategy(
            ticker=ticker,
            lookback=lookback,
            ma1=20,
            ma2=50,
            ma3=100,
            ma4=200,
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



    def test_sar(self):
        """
        Unit test for parabolic sar entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for parabolic sar entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='sar'
        exit_type=None#'nday_range'
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



    def test_channel_breakout(self):
        """
        Unit test for channel breakout entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for channel breakout entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='channel_breakout'
        exit_type=None#'nday_range'
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



    def test_stoch_cross(self):
        """
        Unit test for stochastic crossover entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for stochastic crossover entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='stoch_cross'
        exit_type=None#'nday_range'
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



    def test_stoch_over_under(self):
        """
        Unit test for stoch_over_under entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for stoch_over_under entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='stoch_over_under'
        exit_type=None#'nday_range'
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



    def test_stoch_pop(self):
        """
        Unit test for stoch_pop entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for stoch_pop entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='stoch_pop'
        exit_type=None#'nday_range'
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



    def test_rsi(self):
        """
        Unit test for rsi entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for rsi entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='rsi'
        exit_type=None#'nday_range'
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



    def test_adx(self):
        """
        Unit test for adx entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for adx entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='adx'
        exit_type=None#'nday_range'
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



    def test_macd(self):
        """
        Unit test for macd entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for macd entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='macd'
        exit_type=None#'nday_range'
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



    def test_cci(self):
        """
        Unit test for cci entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for cci entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='cci'
        exit_type=None#'nday_range'
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



    def test_momentum(self):
        """
        Unit test for momentum entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for momentum entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='momentum'
        exit_type=None#'nday_range'
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



    def test_volatility(self):
        """
        Unit test for volatility entry.

        Returns
        -------
        Pass / Fail.

        """
        print("Unit test for volatility entry." )

        ticker='&BRN'
        lookback = 1500
        entry_type='volatility'
        exit_type=None#'nday_range'
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


if __name__ == '__main__':
    unittest.main()
