import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

class TImeSeriesAnalyzer:
    def __init__(self, data, date_column='time', value_column='water_level'):
        self.data = data
        self.date_column = date_column
        self.value_column = value_column
        self.processed_data = None

    def preprocess_data(self):
        if not pd.api.types.is_datetime64_any_dtype(self.data[self.date_column]):
            print(f"transfer {self.date_columna} into datetime type")
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])

        self.data.set_index(self.date_column, inplace=True)

        # handle missing value
        self.data[self.value_column].interpolate(method='time', inplace=True)

        #construct a full range time series
        full_range = pd.date_range(start=self.data.index.min(), end=self.data.index.max(), freq='10T')
        self.processed_data = self.data.reindex(full_range)

        # handle the missing data again
        self.processed_data[self.value_column].interpolate(method='time', inplace=True)

        # drop out 15 minute and 45 minute data
        self.processed_data = self.processed_data[~((self.processed_data.index.minute == 15) | (self.processed_data.index.minute == 45))]

    def check_stationarity(self):
        result = adfuller(self.processed_data[self.value_column].dropna())
        return result[1] <= 0.05
    
    def make_stationary(self):
        if not self.check_stationarity():
            self.processed_data['diff'] = self.processed_data[self.value_column].diff()
            self.value_column = 'diff'
            self.processed_data.dropna(inplace=True)

    def plot_acf_pacf(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        acf_values = acf(self.processed_data[self.value_column].dropna())
        pacf_values = pacf(self.processed_data[self.value_column].dropna())

        ax1.plot(acf_values)
        ax1.set_title('Autocorrelation Function (ACF)')
        ax2.plot(pacf_values)
        ax2.set_title('Partial Autocorrelation Function (PACF)')

        plt.tight_layout()
        plt.show()

    def check_white_noise(self):
        result = acorr_ljungbox(self.processed_data[self.value_column].dropna(), lags=[10])
        return result.iloc[0, 1] > 0.05
    
    def calculate_statistics(self):
        series = self.processed_data[self.value_column].dropna()
        return {
            'mean': series.mean(),
            'std': series.std(),
            'acf': acf(series)[1:],
        }
    
    def run_analysis(self):
        self.preprocess_data()
        is_stationary = self.check_stationarity()
        if not is_stationary:
            print('time series is non-stationary, processing differentiating process...')
            self.make_stationary()

        self.plot_acf_pacf()

        is_white_noise = self.check_white_noise()
        print(f"if time series is stationary: {'yes' if is_white_noise else 'no'}")

        stats = self.calculate_statistics()
        print(f"mean: {stats['mean']}")
        print(f"std: {stats['std']}")
        print(f"ACF value: {stats['acf'][:5]}")




