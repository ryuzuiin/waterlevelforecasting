import pandas as pd
import matplotlib.pyplot as plt
from modeling_time_series_analyzer import ModelingTimeSeriesAnalyzer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """
    Load data from a file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        pd.DataFrame or None: Loaded data if successful, None otherwise.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None

def plot_results(analyzer):
    """
    Plot analysis results.

    Args:
        analyzer (ModelingTimeSeriesAnalyzer): The analyzer instance with results.
    """
    plt.figure(figsize=(15, 10))
    
    # Original data and best model prediction
    plt.subplot(2, 1, 1)
    analyzer.processed_data[analyzer.value_column].plot(label='Original Data')
    best_model_name = min(analyzer.best_models, key=lambda x: analyzer.compare_models()[x])
    best_model = analyzer.best_models[best_model_name]
    best_model.fittedvalues.plot(label=f'Best Model ({best_model_name}) Fit')
    plt.legend()
    plt.title('Original Data vs Best Model Fit')
    
    # Residuals plot
    plt.subplot(2, 1, 2)
    best_model.resid.plot()
    plt.title('Residuals of Best Model')
    
    plt.tight_layout()
    plt.savefig('time_series_analysis_results.png')
    logging.info("Results plot saved as time_series_analysis_results.png")

def main():
    """
    Main function to run the time series analysis.
    """
    # Load data
    data = load_data('your_data.csv')  # Replace with your data file path
    if data is None:
        return

    # Create analyzer instance
    try:
        analyzer = ModelingTimeSeriesAnalyzer(data, 'date_column', 'value_column')  # Replace with your column names
        logging.info("ModelingTimeSeriesAnalyzer instance created successfully")
    except Exception as e:
        logging.error(f"Error creating analyzer: {str(e)}")
        return

    # Run analysis
    try:
        analyzer.run_modeling_analysis()
        logging.info("Modeling analysis completed successfully")
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        return

    # Plot and save results
    plot_results(analyzer)

    # Save best model
    best_model_name = min(analyzer.best_models, key=lambda x: analyzer.compare_models()[x])
    best_model = analyzer.best_models[best_model_name]
    best_model.save(f'best_{best_model_name}_model.pkl')
    logging.info(f"Best model ({best_model_name}) saved as best_{best_model_name}_model.pkl")

if __name__ == "__main__":
    main()