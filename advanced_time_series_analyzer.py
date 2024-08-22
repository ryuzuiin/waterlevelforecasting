import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, MSTL
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

class AdvancedTimeSeriesAnalyzer(TimeSeriesAnalyzer):
    def __init__(self, data, date_column, value_column, freq='10T'):
        super().__init__(data, date_column, value_column, freq)

    def perform_stl_analysis(self, period=144, seasonal=7, trend=None, low_pass=None, robust=False):
        if self.processed_data is None:
            self.is_full_range_time_series()

        stl = STL(self.processed_data[self.value_column], 
                  period=period, 
                  seasonal=seasonal, 
                  trend=trend, 
                  low_pass=low_pass, 
                  robust=robust)
        result = stl.fit()
        self.plot_stl_results(result)
        return result

    def plot_stl_results(self, result):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
        ax1.plot(self.processed_data.index, result.observed)
        ax1.set_title('Observed')
        ax2.plot(self.processed_data.index, result.trend)
        ax2.set_title('Trend')
        ax3.plot(self.processed_data.index, result.seasonal)
        ax3.set_title('Seasonal')
        ax4.plot(self.processed_data.index, result.resid)
        ax4.set_title('Residual')
        plt.tight_layout()
        plt.savefig('stl_decomposition.png')
        plt.show()  # 添加显示图像的功能
        plt.close()
        logging.info("STL decomposition plot saved as stl_decomposition.png")

    @staticmethod
    def mstl_chunk(chunk, periods):
        mstl = MSTL(chunk, periods=periods)
        return mstl.fit()

    def perform_mstl_analysis(self, seasonal_periods=[144, 1008, 52560], n_jobs=None):
        if self.processed_data is None:
            self.is_full_range_time_series()

        ts = self.processed_data[self.value_column]

        chunk_size = len(ts) // (n_jobs or 1)
        chunks = [ts[i:i+chunk_size] for i in range(0, len(ts), chunk_size)]

        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_chunk = {executor.submit(self.mstl_chunk, chunk, periods=seasonal_periods): i for i, chunk in enumerate(chunks)}
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append((chunk_index, result))
                except Exception as exc:
                    logging.error(f'MSTL processing generated an exception for chunk {chunk_index}: {exc}')

        
        results.sort(key=lambda x: x[0])

        combined_trend = pd.concat([result.trend for _, result in results])
        combined_seasonal = pd.concat([pd.DataFrame(result.seasonal, columns=[f"seasonal_{i+1}" for i in range(len(seasonal_periods))]) for _, result in results])
        combined_resid = pd.concat([result.resid for _, result in results])

        combined_result = pd.concat([combined_trend, combined_seasonal, combined_resid], axis=1)

        self.plot_mstl_results(ts, combined_result, seasonal_periods)
        return combined_result

    def plot_mstl_results(self, original_series, combined_result, seasonal_periods):
        fig, axes = plt.subplots(len(seasonal_periods) + 3, 1, figsize=(15, 3*(len(seasonal_periods) + 3)))
        original_series.plot(ax=axes[0])
        axes[0].set_title('Original Time Series')

        combined_result.iloc[:, 0].plot(ax=axes[1])
        axes[1].set_title('Trend')

        for i, period in enumerate(seasonal_periods):
            combined_result.iloc[:, i+1].plot(ax=axes[i+2])
            axes[i+2].set_title(f'Seasonal (period={period})')

        combined_result.iloc[:, -1].plot(ax=axes[-1])
        axes[-1].set_title('Residuals')

        plt.tight_layout()
        plt.savefig('mstl_decomposition.png')
        plt.show()  
        plt.close()
        logging.info("MSTL decomposition plot saved as mstl_decomposition.png")

    def run_advanced_analysis(self, stl_period=144, mstl_periods=[144, 1008, 3000], n_jobs=None):
        super().run_analysis()  # Run the basic analysis from TimeSeriesAnalyzer
        
        stl_result = self.perform_stl_analysis(period=stl_period)
        mstl_result = self.perform_mstl_analysis(seasonal_periods=mstl_periods, n_jobs=n_jobs)
        
        return stl_result, mstl_result
