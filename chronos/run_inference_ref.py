import pandas as pd
import torch
import matplotlib.pyplot as plt
from chronos import BaseChronosPipeline, ChronosPipeline
from typing import Tuple, List, Optional
import numpy as np


class ChronosForecaster:
    """A wrapper class for Chronos time series forecasting models."""
    
    def __init__(self, model_name: str = "amazon/chronos-bolt-small", 
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """
        Initialize the Chronos forecaster.
        
        Args:
            model_name: Name of the Chronos model to use
            device: Device to run inference on ('cuda' or 'cpu')
            dtype: Data type for model weights
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the Chronos model pipeline."""
        try:
            if "bolt" in self.model_name:
                self.pipeline = BaseChronosPipeline.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    torch_dtype=self.dtype,
                )
            else:
                self.pipeline = ChronosPipeline.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    torch_dtype=self.dtype,
                )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_data(self, data_path: str, value_column: str) -> pd.DataFrame:
        """
        Load time series data from CSV file.
        
        Args:
            data_path: Path to CSV file (local or URL)
            value_column: Name of the column containing time series values
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(data_path)
            if value_column not in df.columns:
                raise ValueError(f"Column '{value_column}' not found in data")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def prepare_train_test_split(self, data: pd.Series, prediction_length: int) -> Tuple[pd.Series, pd.Series]:
        """
        Split data into training and test sets, holding out the last prediction_length samples.
        
        Args:
            data: Complete time series data
            prediction_length: Number of samples to hold out for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if len(data) <= prediction_length:
            raise ValueError(f"Data length ({len(data)}) must be greater than prediction_length ({prediction_length})")
        
        train_data = data[:-prediction_length]
        test_data = data[-prediction_length:]
        
        return train_data, test_data
    
    def predict_quantiles(self, context: torch.Tensor, prediction_length: int,
                         quantile_levels: List[float] = [0.1, 0.5, 0.9]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate quantile forecasts.
        
        Args:
            context: Historical time series data as 1D tensor
            prediction_length: Number of future periods to forecast
            quantile_levels: List of quantile levels to predict
            
        Returns:
            Tuple of (quantiles, mean) tensors
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded")
        
        return self.pipeline.predict_quantiles(
            context=context,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )
    
    def get_embeddings(self, context: torch.Tensor) -> Tuple[torch.Tensor, any]:
        """
        Get embeddings for the input time series.
        
        Args:
            context: Historical time series data as 1D tensor
            
        Returns:
            Tuple of (embeddings, tokenizer_state)
        """
        if hasattr(self.pipeline, 'embed'):
            return self.pipeline.embed(context)
        else:
            raise NotImplementedError("Embedding not available for this model type")


class TimeSeriesVisualizer:
    """Utility class for visualizing time series forecasts."""
    
    @staticmethod
    def plot_forecast_with_ground_truth(train_data: pd.Series, test_data: pd.Series, 
                                      quantiles: torch.Tensor, prediction_length: int,
                                      title: str = "Time Series Forecast vs Ground Truth",
                                      save_path: Optional[str] = None, 
                                      figsize: Tuple[int, int] = (14, 8)):
        """
        Plot training data, ground truth, and forecast with prediction intervals.
        
        Args:
            train_data: Training time series data
            test_data: Ground truth test data
            quantiles: Quantile predictions tensor [batch_size, prediction_length, num_quantiles]
            prediction_length: Number of forecasted periods
            title: Plot title
            save_path: Path to save the plot (optional)
            figsize: Figure size tuple
        """
        # Extract quantiles (assuming first batch and 3 quantiles: low, median, high)
        low = quantiles[0, :, 0].cpu().numpy()
        median = quantiles[0, :, 1].cpu().numpy()
        high = quantiles[0, :, 2].cpu().numpy()
        
        # Create indices
        train_index = range(len(train_data))
        test_index = range(len(train_data), len(train_data) + len(test_data))
        forecast_index = range(len(train_data), len(train_data) + prediction_length)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Plot training data
        plt.plot(train_index, train_data.values, color="royalblue", 
                label="Training Data", linewidth=2)
        
        # Plot ground truth test data
        plt.plot(test_index, test_data.values, color="darkgreen", 
                label="Ground Truth", linewidth=2, marker='o', markersize=4)
        
        # Plot forecast
        plt.plot(forecast_index, median, color="tomato", 
                label="Median Forecast", linewidth=2, linestyle='--')
        
        # Plot prediction intervals
        plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, 
                        label="80% Prediction Interval")
        
        # Add vertical line to separate train/test
        plt.axvline(x=len(train_data)-0.5, color='gray', linestyle=':', alpha=0.7, 
                   label='Train/Test Split')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("Time Period", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def calculate_metrics(ground_truth: np.ndarray, predictions: np.ndarray) -> dict:
        """
        Calculate forecast accuracy metrics.
        
        Args:
            ground_truth: Actual values
            predictions: Predicted values
            
        Returns:
            Dictionary containing various accuracy metrics
        """
        mae = np.mean(np.abs(ground_truth - predictions))
        mse = np.mean((ground_truth - predictions) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((ground_truth - predictions) / ground_truth)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }


def main():
    """Main function demonstrating the usage of Chronos models with ground truth evaluation."""
    
    # Configuration
    DATA_URL = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    VALUE_COLUMN = "#Passengers"
    PREDICTION_LENGTH = 12
    QUANTILE_LEVELS = [0.1, 0.5, 0.9]
    
    # Initialize forecaster
    print("Loading Chronos-Bolt model...")
    forecaster = ChronosForecaster(
        model_name="amazon/chronos-bolt-small",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16
    )
    
    # Load data
    print("Loading time series data...")
    df = forecaster.load_data(DATA_URL, VALUE_COLUMN)
    print(f"Total data shape: {df.shape}")
    
    # Split data into train and test
    print(f"Splitting data: holding out last {PREDICTION_LENGTH} samples for testing...")
    train_data, test_data = forecaster.prepare_train_test_split(
        df[VALUE_COLUMN], PREDICTION_LENGTH
    )
    
    print(f"Training data length: {len(train_data)}")
    print(f"Test data length: {len(test_data)}")
    print(f"Training data range: {train_data.min()} to {train_data.max()}")
    print(f"Test data range: {test_data.min()} to {test_data.max()}")
    
    # Prepare context tensor (only training data)
    context = torch.tensor(train_data.values, dtype=torch.float32)
    
    # Generate forecasts
    print("Generating forecasts...")
    quantiles, mean = forecaster.predict_quantiles(
        context=context,
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=QUANTILE_LEVELS
    )
    
    print(f"Forecast quantiles shape: {quantiles.shape}")
    print(f"Mean forecast shape: {mean.shape}")
    
    # Extract median predictions for evaluation
    median_predictions = quantiles[0, :, 1].cpu().numpy()  # Middle quantile (0.5)
    
    # Calculate accuracy metrics
    print("\nForecast Accuracy Metrics:")
    visualizer = TimeSeriesVisualizer()
    metrics = visualizer.calculate_metrics(test_data.values, median_predictions)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Visualize results with ground truth
    print("\nCreating visualization with ground truth...")
    visualizer.plot_forecast_with_ground_truth(
        train_data=train_data,
        test_data=test_data,
        quantiles=quantiles,
        prediction_length=PREDICTION_LENGTH,
        title="Air Passengers Forecast vs Ground Truth - Chronos Model",
        save_path="air_passengers_forecast_with_ground_truth.png"
    )
    
    # Print some sample predictions vs actual
    print("\nSample Predictions vs Actual:")
    print("Period | Actual | Predicted | Error")
    print("-" * 35)
    for i in range(min(6, len(test_data))):
        actual = test_data.iloc[i]
        predicted = median_predictions[i]
        error = abs(actual - predicted)
        print(f"{i+1:6d} | {actual:6.1f} | {predicted:9.1f} | {error:5.1f}")
    
    # Optional: Get embeddings (for T5-based models)
    try:
        print("\nGenerating embeddings...")
        t5_forecaster = ChronosForecaster(
            model_name="amazon/chronos-t5-small",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        embeddings, tokenizer_state = t5_forecaster.get_embeddings(context)
        print(f"Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"Embeddings not available: {e}")


if __name__ == "__main__":
    main()
