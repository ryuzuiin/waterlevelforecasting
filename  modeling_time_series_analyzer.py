from advanced_time_series_analyzer import AdvancedTimeSeriesAnalyzer
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import itertools

class ModelingTimeSeriesAnalyzer(AdvancedTimeSeriesAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_models = {}

    def optimize_ar_model(self, max_lag=20):
        """
        Optimize AR model using grid search.
        """
        best_order = 1
        best_aic = np.inf
        
        for order in range(1, max_lag + 1):
            model = AutoReg(self.processed_data[self.value_column], lags=order)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
        
        best_model = AutoReg(self.processed_data[self.value_column], lags=best_order).fit()
        self.best_models['AR'] = best_model
        print(f"Best AR model order: {best_order}")
        return best_model

    def optimize_ma_model(self, max_order=5):
        """
        Optimize MA model using grid search.
        """
        best_order = 0
        best_aic = np.inf
        
        for order in range(1, max_order + 1):
            model = ARIMA(self.processed_data[self.value_column], order=(0, 0, order))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
        
        best_model = ARIMA(self.processed_data[self.value_column], order=(0, 0, best_order)).fit()
        self.best_models['MA'] = best_model
        print(f"Best MA model order: {best_order}")
        return best_model

    def optimize_arima_model(self, p_range=range(0, 6), d_range=range(0, 3), q_range=range(0, 6)):
        """
        Optimize ARIMA model using grid search.
        """
        best_aic = np.inf
        best_order = None
        
        for p, d, q in itertools.product(p_range, d_range, q_range):
            try:
                model = ARIMA(self.processed_data[self.value_column], order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except:
                continue
        
        best_model = ARIMA(self.processed_data[self.value_column], order=best_order).fit()
        self.best_models['ARIMA'] = best_model
        print(f"Best ARIMA model order: {best_order}")
        return best_model

    def compare_models(self, test_size=0.2):
        """
        Compare the performance of optimized models.
        """
        train_size = int(len(self.processed_data) * (1 - test_size))
        train, test = self.processed_data[:train_size], self.processed_data[train_size:]
        
        results = {}
        for name, model in self.best_models.items():
            predictions = model.forecast(steps=len(test))
            mse = mean_squared_error(test[self.value_column], predictions)
            results[name] = mse
        
        best_model = min(results, key=results.get)
        print("Model Comparison Results (MSE):")
        for name, mse in results.items():
            print(f"{name}: {mse}")
        print(f"\nBest model: {best_model}")
        
        return results

    def run_modeling_analysis(self, ar_max_lag=20, ma_max_order=5, 
                              arima_p_range=range(0, 6), arima_d_range=range(0, 3), arima_q_range=range(0, 6)):
        """
        Run the complete modeling analysis.
        """
        super().run_advanced_analysis()
        self.optimize_ar_model(max_lag=ar_max_lag)
        self.optimize_ma_model(max_order=ma_max_order)
        self.optimize_arima_model(p_range=arima_p_range, d_range=arima_d_range, q_range=arima_q_range)
        self.compare_models()

# Example usage:
# if __name__ == '__main__':
#     data = pd.read_csv('your_data.csv')
#     modeling_analyzer = ModelingTimeSeriesAnalyzer(data, 'date_column', 'value_column')
#     modeling_analyzer.run_modeling_analysis()