from time_series_analyzer import TimeSeriesAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, MSTL
from numba import jit
import unittest
import os
import json

from concurrent.futures import ProcePoolExecutor, as_completed
import logging


class AdvancedTimeSeriesAnalyzer(TimeSeriesAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = {}

    def plot_seasonal_decompose(self, model='additive'):
        """
        Plot the seasonal decomposition of the time series.

        Args:
            model (str): The type of seasonal component. Either 'additive' or 'multiplicative'.
        """
        result = seasonal_decompose(self.processed_data[self.value_column], model=model)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
        
        result.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        result.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        result.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonal')
        result.resid.plot(ax=ax4)
        ax4.set_title('Residual')
        
        plt.tight_layout()
        plt.show()

    def perform_mstl_analysis(self, seasonal_periods=[144, 1008, 52560], seasonal_windows=[51, 401, 11001], iterate=2, n_jobs=None):
        '''
        Perform MSTL (Multiple Seasonal-Trend decomposition using LOESS) analysis
        with paraleel processing using concurrent.futures for large datasets.

        Args:
            seasonal_periods (list): List of integers representing the seasonal periods
                Default is [144, 1008, 52560] for daily, weekly, and yearly seasonality
                with 10-minute intervals.
            n_jobs (int): Number of jobs to run in parallel. If None, uses all available CPU cores.

        Returns:
            pd.DataFrame: The combined MSTL decomposition results.
        '''
        # Ensure the data is a pandas Series with a datetime index
        ts = pd.Series(self.processed_data[self.value_column], index=self.processed_data.index)

        # Function toperform MSTL on a chunk of data
        def mstl_chunk(chunk):
            mstl = MSTL(chunk, periods=seasonal_periods, windows=seasonal_windows, iterate=iterate)
            return mstl.fit()

        # Split the data into chunks
        chunk_size = len(ts) // (n_jobs or 1)
        chunks = [ts[i:i+chunk_size] for i in range(0, len(ts), chunk_size)]

        # Perform parallel MSTL decomposition
        results = []
        with ProcessPoolExcutor(max_workers=n_jobs) as excutor:
            future_to_chunk = {excutor.submit(mstl_chunk, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_comlete(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result()
                    result.append((chunk_index, result))
                except Exception as exc:
                    logging.error(f'MSTL processing generated an exception for chunk {chunk_index}: {exc}')

        results.sort(key=lambda x:x[0])
        combined_result = pd.concat([result[1].trend for _, result in results])
        for i, period in enumerate(seasonal_periods):
            seasonal = pd.concat([result[1].seasonal.iloc[:, i] for _, result in results])
            combined_result = pd.concat([combined_result, seasonal], axis=1)
        combined_result = pd.concat([combined_result, pd.concat([result[1].resid for _, result in results])], axis=1)

        fig, axes = plt.subplots(len(seasonal_periods) + 3, 1, figsize=(15, 3*(len(seasonal_periods) + 3)))

        ts.plot(ax=axes[0])
        axes[0].set_title('Original Time Series')

        combined_result.iloc[:, 0].plot(ax=axes[1])
        axes[1].set_title('Trend')

        for i, period in enumerate(seasonal_periods):
            combined_result.iloc[:, i+1].plot(ax=axes[i+2])
            axes[i+2].set_title(f'Seasonal (period={period}')

        combined_result.iloc[:, -1].plot(ax=axes[-1])
        axes[-1].set_title('Residuals')

        plt.tight_layout()
        plt.savefig(7mstl_decomposition.png')
        plt.close()

        logging.info('MSTL decomposition plot saved as mstl_decomposition.png')

        return combined_result
        

    @jit(nopython=True)
    def _fast_acf(self, x, nlags):
        """
        A faster implementation of ACF using Numba.
        """
        mean = np.mean(x)
        c0 = np.sum((x - mean) ** 2) / len(x)
        
        acf = np.zeros(nlags + 1)
        for t in range(nlags + 1):
            acf[t] = np.sum((x[:-t] - mean) * (x[t:] - mean)) / len(x) / c0
        
        return acf

    def calculate_statistics(self):
        """
        Override the parent method with a faster ACF calculation.
        """
        series = self.processed_data[self.value_column].values
        return {
            'mean': np.mean(series),
            'std': np.std(series),
            'acf': self._fast_acf(series, 20)[1:]  # Calculate ACF for 20 lags
        }

    def save_results(self, filename='analysis_results.json'):
        """
        Save the analysis results to a JSON file.
        """
        self.results['statistics'] = self.calculate_statistics()
        self.results['normality_test'] = self.check_normality()
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"Results saved to {filename}")

    def run_advanced_analysis(self, *args, **kwargs):
        """
        Run the complete analysis including new features.
        """
        super().run_analysis(*args, **kwargs)
        self.plot_seasonal_decompose()
        self.save_results()

class TestAdvancedTimeSeriesAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
        self.df = pd.DataFrame({'date': dates, 'value': values})
        self.analyzer = AdvancedTimeSeriesAnalyzer(self.df, 'date', 'value')

    def test_seasonal_decompose(self):
        # This test just checks if the method runs without errors
        try:
            self.analyzer.plot_seasonal_decompose()
        except Exception as e:
            self.fail(f"plot_seasonal_decompose raised {type(e).__name__} unexpectedly!")

    def test_fast_acf(self):
        acf = self.analyzer._fast_acf(self.df['value'].values, 20)
        self.assertEqual(len(acf), 21)
        self.assertAlmostEqual(acf[0], 1, places=7)

    def test_save_results(self):
        self.analyzer.run_advanced_analysis()
        self.assertTrue(os.path.exists('analysis_results.json'))

if __name__ == '__main__':
    unittest.main()
