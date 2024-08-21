import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Union, Tuple, Any
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from arch.unitroot import PhillipsPerron
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro, kstest, norm

class TimeSeriesAnalyzer:
    """
    A class for analyzing time series data, including stationarity tests,
    missing value handling, and various statistical analyses.
    """

    def __init__(self, data: pd.DataFrame, date_column: str, value_column: str, 
                 freq: str = '10T', season_start: int = 4, season_end: int = 9):
        """
        Initialize the TimeSeriesAnalyzer.

        Args:
            data (pd.DataFrame): The input time series data.
            date_column (str): The name of the column containing date/time information.
            value_column (str): The name of the column containing the values to analyze.
            freq (str, optional): The frequency of the time series. Defaults to '10T' (10 minutes).
            season_start (int, optional): The starting month of the season. Defaults to 4 (April).
            season_end (int, optional): The ending month of the season. Defaults to 9 (September).
        """
        self.data = data.copy()
        self.date_column = date_column
        self.value_column = value_column
        self.freq = freq
        self.processed_data: Optional[pd.DataFrame] = None
        self.sample_size = len(data)
        self.season_start = season_start
        self.season_end = season_end

    def is_datetime_column(self) -> bool:
        """
        Detect if the date column is of datetime type.

        Returns:
            bool: True if the column is datetime, False otherwise.

        Raises:
            ValueError: If the specified date column is not found in the data.
        """
        if self.date_column in self.data.columns:
            return pd.api.types.is_datetime64_any_dtype(self.data[self.date_column])
        else:
            raise ValueError(f'Column {self.date_column} is not found in data')

    def make_datetime(self) -> pd.Series:
        """
        Convert the date column to datetime type if it's not already.

        Returns:
            pd.Series: The date column as datetime type.
        """
        if not self.is_datetime_column():
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
        return self.data[self.date_column]

    def is_time_ordered(self) -> bool:
        """
        Check if the time series is ordered chronologically.

        Returns:
            bool: True if the series is time-ordered, False otherwise.
        """
        if self.is_datetime_column():
            return self.data[self.date_column].is_monotonic_increasing
        else:
            return self.make_datetime().is_monotonic_increasing

    def set_date_index(self) -> pd.DataFrame:
        """
        Set the date column as the index and ensure it's sorted.

        Returns:
            pd.DataFrame: The data with date index set and sorted.
        """
        if not self.is_time_ordered():
            self.data.sort_values(by=self.date_column, inplace=True)
        self.data.set_index(self.date_column, inplace=True)
        return self.data

    def is_full_range_time_series(self) -> pd.DataFrame:
        """
        Check if the time series covers the full range of expected dates and
        construct a full range time series if necessary.

        Returns:
            pd.DataFrame: The processed data covering the full date range.
        """
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.set_date_index()

        years = self.data.index.year.unique()
        full_range = pd.DatetimeIndex([])
        for year in years:
            season_start = pd.Timestamp(year=year, month=self.season_start, day=1)
            season_end = pd.Timestamp(year=year, month=self.season_end, day=30) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            year_range = pd.date_range(start=season_start, end=season_end, freq=self.freq)
            full_range = full_range.union(year_range)

        is_full_range = self.data.index.isin(full_range).all() and (len(self.data) == len(full_range))

        if is_full_range:
            print('The time series is a full range time series')
            self.processed_data = self.data.copy()
        else:
            print('The time series is not a full range time series')
            self.processed_data = self.data.reindex(full_range)
            self.check_missing_values(fill_method='time')
            self.processed_data.sort_index(inplace=True)

        return self.processed_data

    def check_missing_values(self, fill_method: Optional[str] = None) -> pd.DataFrame:
        """
        Detect and handle missing values in the time series.

        Args:
            fill_method (str, optional): Method to fill missing values. 
                Options: 'ffill', 'bfill', 'mean', 'drop', 'linear', 'time'.

        Returns:
            pd.DataFrame: The processed data with missing values handled.

        Raises:
            ValueError: If processed_data is None or if an unsupported fill method is specified.
        """
        if self.processed_data is None:
            raise ValueError("processed_data is None. Run is_full_range_time_series() first.")
        
        missing_values = self.processed_data[self.value_column].isnull().sum()

        if missing_values > 0:
            print(f'{missing_values} missing data found')

            if fill_method == 'ffill':
                self.processed_data[self.value_column].fillna(method='ffill', inplace=True)
            elif fill_method == 'bfill':
                self.processed_data[self.value_column].fillna(method='bfill', inplace=True)
            elif fill_method == 'mean':
                self.fill_with_mean_of_neighbor()
            elif fill_method == 'drop':
                self.processed_data.dropna(subset=[self.value_column], inplace=True)
            elif fill_method in ['linear', 'time']:
                self.processed_data[self.value_column].interpolate(method=fill_method, inplace=True)
            elif fill_method is not None:
                raise ValueError(f"Unsupported fill method: {fill_method}")
            else:
                print('No fill method specified. Missing values will be retained.')
        else:
            print('No missing data found')

        return self.processed_data

    def fill_with_mean_of_neighbor(self) -> None:
        """
        Fill missing values with the mean of neighboring values.
        """
        for i in range(1, len(self.processed_data) - 1):
            if pd.isnull(self.processed_data.iloc[i][self.value_column]):
                prev_value = self.processed_data.iloc[i-1][self.value_column]
                next_value = self.processed_data.iloc[i+1][self.value_column]
                if pd.notnull(prev_value) and pd.notnull(next_value):
                    self.processed_data.iloc[i][self.value_column] = (prev_value + next_value) / 2

    def drop_minute_value(self, minutes_to_drop: List[int] = [15, 45]) -> pd.DataFrame:
        """
        Drop rows with specified minute values from the time series.

        Args:
            minutes_to_drop (List[int], optional): List of minute values to drop. Defaults to [15, 45].

        Returns:
            pd.DataFrame: The processed data with specified minute values dropped.
        """
        if not isinstance(self.processed_data.index, pd.DatetimeIndex):
            self.set_date_index()

        self.processed_data = self.processed_data[~self.processed_data.index.minute.isin(minutes_to_drop)]
        return self.processed_data

    def is_stationary_adf(self, significance_level: float = 0.05) -> bool:
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Args:
            significance_level (float, optional): The significance level for the test. Defaults to 0.05.

        Returns:
            bool: True if the series is stationary, False otherwise.
        """
        result = adfuller(self.processed_data[self.value_column])
        print('====================================================')
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')
        print('====================================================')
        return result[1] <= significance_level
    
    def is_stationary_pp(self, significance_level: float = 0.05) -> bool:
        """
        Perform Phillips-Perron test for stationarity.

        Args:
            significance_level (float, optional): The significance level for the test. Defaults to 0.05.

        Returns:
            bool: True if the series is stationary, False otherwise.
        """
        pp_test = PhillipsPerron(self.processed_data[self.value_column])
        result = pp_test.statistic, pp_test.pvalue, pp_test.critical_values
        print('====================================================')
        print('Phillips-Perron Test Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical values:')
        for key, value in result[2].items():
            print(f'\t{key}: {value}')
        print('====================================================')
        return result[1] <= significance_level

    def is_trend_stationary(self, significance_level: float = 0.05) -> bool:
        """
        Perform KPSS test for trend stationarity.

        Args:
            significance_level (float, optional): The significance level for the test. Defaults to 0.05.

        Returns:
            bool: True if the series is trend stationary, False otherwise.
        """
        result = kpss(self.processed_data[self.value_column], regression='ct')
        print('====================================================')
        print('KPSS Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('====================================================')
        return result[1] > significance_level

    def is_necessary_differencing(self, adf_threshold: int = 1000) -> bool:
        """
        Check if differencing is necessary based on stationarity tests.

        Args:
            adf_threshold (int, optional): Threshold for using ADF vs PP test. Defaults to 1000.

        Returns:
            bool: True if differencing is necessary, False otherwise.
        """
        kpss_stationary = self.is_trend_stationary()

        if self.sample_size <= adf_threshold:
            print('Using ADF test for small sample size')
            unit_root_test_stationary = self.is_stationary_adf()
            test_name = 'ADF'
        else:
            print('Using PP test for large sample size')
            unit_root_test_stationary = self.is_stationary_pp()
            test_name = 'PP'

        print('=============================================')
        print('Stationarity Test Results:')
        print(f'{test_name} Test: {"Stationary" if unit_root_test_stationary else "Non-stationary"}')
        print(f'KPSS Test: {"Trend Stationary" if kpss_stationary else "Not Trend Stationary"}')
        print('=============================================')

        return not (unit_root_test_stationary and kpss_stationary)

    def make_stationarity(self, differencing_order: int = 1) -> pd.DataFrame:
        """
        Make the time series stationary through differencing if necessary.

        Args:
            differencing_order (int, optional): The order of differencing. Defaults to 1.

        Returns:
            pd.DataFrame: The processed data after making it stationary.
        """
        if self.is_necessary_differencing():
            for _ in range(differencing_order):
                self.processed_data[self.value_column] = self.processed_data[self.value_column].diff()
                self.processed_data.dropna(inplace=True)

        return self.processed_data

    def plot_acf_pacf(self, lags: int = 10) -> None:
        """
        Plot the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

        Args:
            lags (int, optional): Number of lags to plot. Defaults to 10.
        """
        plt.figure(figsize=(12, 10))

        plt.subplot(221)
        self.data[self.value_column].plot(kind='line')
        plt.title('Original Time Series')

        plt.subplot(222)
        self.data[self.value_column].plot(kind='hist')
        plt.title('Original Time Series Histogram')

        self.make_stationarity()

        plt.subplot(223)
        self.processed_data[self.value_column].plot(kind='line')
        plt.title('Stationary Time Series')

        plt.subplot(224)
        self.processed_data[self.value_column].plot(kind='hist')
        plt.title('Stationary Time Series Histogram')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_acf(self.processed_data[self.value_column], lags=lags, ax=plt.gca(), alpha=0.05)
        plt.title('Autocorrelation Function (ACF)')

        plt.subplot(122)
        plot_pacf(self.processed_data[self.value_column], lags=lags, ax=plt.gca(), alpha=0.05)
        plt.title('Partial Autocorrelation Function (PACF)')

        plt.tight_layout()
        plt.show()

    def check_normality(self, sample_size_threshold: int = 50) -> Tuple[float, float]:
        """
        Check the normality of the time series data.

        Args:
            sample_size_threshold (int, optional): Threshold for choosing between Shapiro-Wilk and Kolmogorov-Smirnov tests. Defaults to 50.

        Returns:
            Tuple[float, float]: The test statistic and p-value.
        """
        series = self.processed_data[self.value_column]
        n = len(series)

        if n <= sample_size_threshold:
            print('===========================================')
            print('Using Shapiro-Wilk test')
            print('===========================================')
            statistic, p_value = shapiro(series)
        else:
            print('===========================================')
            print('Using Kolmogorov-Smirnov test')
            print('===========================================')
            statistic, p_value = kstest(series, 'norm', args=(series.mean(), series.std()))

        print('====================================================')
        print('Statistic: %f' % statistic)
        print('p-value: %f' % p_value)
        print('====================================================')

        if p_value > 0.05:
            print('The data is normally distributed')
        else:
            print('The data is not normally distributed')

        return statistic, p_value

    def check_white_noise(self, lags: int = 10) -> None:
        """
        Check if the time series is white noise using the Ljung-Box test.

        Args:
            lags (int, optional): Number of lags to use in the test. Defaults to 10.
        """
        lb_test = acorr_ljungbox(self.processed_data[self.value_column], lags=[lags], return_df=True)
        print('====================================================')
        print(lb_test)
        print('====================================================')
        if lb_test['lb_pvalue'].iloc[0] > 0.05:
            print('The time series is white noise')
        else:
            print('The time series is not white noise')

    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate basic statistics of the time series.

        Returns:
            Dict[str, Any]: A dictionary containing mean, standard deviation, and ACF of the series.
        """
        series = self.processed_data[self.value_column]

        return {
            'mean': series.mean(),
            'std': series.std(),
            'acf': acf(series)[1:]
        }

    def run_analysis(self, fill_method: Optional[str] = None, differencing_order: int = 1, lags: int = 20) -> None:
        """
        Run a complete analysis of the time series.

        Args:
            fill_method (Optional[str], optional): Method to fill missing values. Defaults to None.
            differencing_order (int, optional): Order of differencing for making the series stationary. Defaults to 1.
            lags (int, optional): Number of lags to use in ACF and PACF plots. Defaults to 20.
        """
        self.is_full_range_time_series()
        self.drop_minute_value()
        self.check_missing_values(fill_method)
        self.make_stationarity(differencing_order)
        self.plot_acf_pacf(lags)
        self.check_white_noise(lags)
        self.check_normality()
        stats = self.calculate_statistics()

        print('====================================================')
        print(f"Mean: {stats['mean']}")
        print(f"Standard Deviation: {stats['std']}")
        print('ACF:')
        print(stats['acf'])
        print('====================================================')

# Example usage:
# data = pd.read_csv('your_data.csv')
# analyzer = TimeSeriesAnalyzer(data, 'date_column', 'value_column')
# analyzer.run_analysis(fill_method='linear', differencing_order=1, lags=30)
