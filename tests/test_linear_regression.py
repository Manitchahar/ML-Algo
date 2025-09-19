"""
Unit tests for Linear Regression implementation.

This module contains comprehensive tests for the LinearRegression class
to ensure correctness and robustness of the implementation.

Test Coverage:
- Parameter validation
- Fitting with different methods
- Prediction accuracy
- Edge cases and error handling
- Mathematical correctness

Author: ML-Algo Contributors
Date: 2025
License: MIT
"""

import numpy as np
import pytest
import sys
import os

# Add the parent directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.supervised.regression import LinearRegression
from utils.metrics import mean_squared_error, r2_score


class TestLinearRegression:
    """Test suite for Linear Regression implementation."""
    
    def setup_method(self):
        """
        Set up test fixtures before each test method.
        
        Creates sample data that will be used across multiple tests.
        This ensures consistent test conditions and reduces code duplication.
        """
        # Set random seed for reproducible tests
        np.random.seed(42)
        
        # Generate simple synthetic data
        self.n_samples = 100
        self.n_features = 2
        
        # Create feature matrix
        self.X = np.random.randn(self.n_samples, self.n_features)
        
        # Create target with known linear relationship
        self.true_coef = np.array([2.0, -1.5])
        self.true_intercept = 0.5
        self.y = self.X @ self.true_coef + self.true_intercept + 0.1 * np.random.randn(self.n_samples)
        
        # Create perfect linear data (no noise) for exact tests
        self.X_perfect = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_perfect = np.array([5, 8, 11, 14])  # y = 1*x1 + 2*x2
    
    def test_initialization_default_parameters(self):
        """Test that default parameters are set correctly."""
        model = LinearRegression()
        
        assert model.method == 'normal_equation'
        assert model.learning_rate == 0.01
        assert model.max_iterations == 1000
        assert model.tolerance == 1e-6
        assert model.fit_intercept == True
        assert model.is_fitted_ == False
    
    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = LinearRegression(
            method='gradient_descent',
            learning_rate=0.1,
            max_iterations=500,
            tolerance=1e-4,
            fit_intercept=False
        )
        
        assert model.method == 'gradient_descent'
        assert model.learning_rate == 0.1
        assert model.max_iterations == 500
        assert model.tolerance == 1e-4
        assert model.fit_intercept == False
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Invalid method
        with pytest.raises(ValueError, match="method must be one of"):
            LinearRegression(method='invalid_method')
        
        # Invalid learning rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            LinearRegression(learning_rate=-0.1)
        
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            LinearRegression(learning_rate=0)
        
        # Invalid max_iterations
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            LinearRegression(max_iterations=-10)
        
        # Invalid tolerance
        with pytest.raises(ValueError, match="tolerance must be positive"):
            LinearRegression(tolerance=-1e-6)
    
    def test_fit_normal_equation(self):
        """Test fitting with normal equation method."""
        model = LinearRegression(method='normal_equation')
        
        # Fit the model
        result = model.fit(self.X, self.y)
        
        # Check that fit returns self (for method chaining)
        assert result is model
        
        # Check that model is marked as fitted
        assert model.is_fitted_ == True
        
        # Check that coefficients are reasonable (close to true values)
        np.testing.assert_allclose(model.coef_, self.true_coef, atol=0.2)
        np.testing.assert_allclose(model.intercept_, self.true_intercept, atol=0.2)
    
    def test_fit_gradient_descent(self):
        """Test fitting with gradient descent method."""
        model = LinearRegression(
            method='gradient_descent',
            learning_rate=0.01,
            max_iterations=1000,
            tolerance=1e-6
        )
        
        # Fit the model
        model.fit(self.X, self.y)
        
        # Check that model is fitted
        assert model.is_fitted_ == True
        
        # Check that cost history is recorded
        assert len(model.cost_history_) > 0
        assert model.n_iterations_ > 0
        
        # Check that coefficients are reasonable
        np.testing.assert_allclose(model.coef_, self.true_coef, atol=0.2)
        np.testing.assert_allclose(model.intercept_, self.true_intercept, atol=0.2)
    
    def test_perfect_linear_data(self):
        """Test on perfect linear data (should recover exact parameters)."""
        model = LinearRegression(method='normal_equation')
        model.fit(self.X_perfect, self.y_perfect)
        
        # For perfect linear data, should recover exact parameters
        # y = 1*x1 + 2*x2 + 0 (no intercept in this case)
        expected_coef = np.array([1.0, 2.0])
        expected_intercept = 0.0
        
        np.testing.assert_allclose(model.coef_, expected_coef, atol=1e-10)
        np.testing.assert_allclose(model.intercept_, expected_intercept, atol=1e-10)
    
    def test_prediction_before_fitting(self):
        """Test that prediction before fitting raises an error."""
        model = LinearRegression()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(self.X)
    
    def test_prediction_after_fitting(self):
        """Test predictions after fitting."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        
        # Make predictions
        predictions = model.predict(self.X)
        
        # Check shape
        assert predictions.shape == (self.n_samples,)
        
        # Check that predictions are reasonable (high R²)
        r2 = r2_score(self.y, predictions)
        assert r2 > 0.8  # Should have good fit
    
    def test_prediction_dimension_mismatch(self):
        """Test prediction with wrong number of features."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        
        # Try to predict with wrong number of features
        X_wrong = np.random.randn(10, 3)  # 3 features instead of 2
        
        with pytest.raises(ValueError, match="has 3 features, but model was trained on 2 features"):
            model.predict(X_wrong)
    
    def test_single_sample_prediction(self):
        """Test prediction on a single sample."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        
        # Predict single sample
        single_sample = self.X[0:1]  # Keep 2D shape
        prediction = model.predict(single_sample)
        
        assert prediction.shape == (1,)
        
        # Also test with 1D input (should be reshaped automatically)
        single_sample_1d = self.X[0]
        prediction_1d = model.predict(single_sample_1d)
        
        assert prediction_1d.shape == (1,)
        np.testing.assert_allclose(prediction, prediction_1d)
    
    def test_score_method(self):
        """Test the R² score calculation."""
        model = LinearRegression()
        model.fit(self.X, self.y)
        
        # Calculate R² score
        score = model.score(self.X, self.y)
        
        # Should be between 0 and 1 for reasonable data
        assert 0 <= score <= 1
        
        # Should be high for this synthetic data
        assert score > 0.8
    
    def test_fit_without_intercept(self):
        """Test fitting without intercept term."""
        # Create data with no intercept: y = 2*x1 - x2
        X = np.random.randn(100, 2)
        y = 2*X[:, 0] - X[:, 1]  # No intercept term
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        
        # Intercept should be zero
        assert model.intercept_ == 0.0
        
        # Coefficients should be close to [2, -1]
        np.testing.assert_allclose(model.coef_, [2.0, -1.0], atol=0.1)
    
    def test_input_validation_fit(self):
        """Test input validation during fitting."""
        model = LinearRegression()
        
        # Test with invalid shapes
        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            model.fit(np.array([1, 2, 3]), np.array([1, 2, 3]))
        
        with pytest.raises(ValueError, match="y must be 1-dimensional"):
            model.fit(self.X, self.y.reshape(-1, 1))
        
        # Test with mismatched sample sizes
        with pytest.raises(ValueError, match="X and y must have same number of samples"):
            model.fit(self.X, self.y[:-1])
        
        # Test with empty data
        with pytest.raises(ValueError, match="Cannot fit model with 0 samples"):
            model.fit(np.empty((0, 2)), np.empty(0))
        
        # Test with NaN values
        X_nan = self.X.copy()
        X_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="X contains NaN or infinite values"):
            model.fit(X_nan, self.y)
        
        y_nan = self.y.copy()
        y_nan[0] = np.nan
        with pytest.raises(ValueError, match="y contains NaN or infinite values"):
            model.fit(self.X, y_nan)
    
    def test_convergence_gradient_descent(self):
        """Test that gradient descent converges properly."""
        model = LinearRegression(
            method='gradient_descent',
            learning_rate=0.01,
            max_iterations=1000,
            tolerance=1e-8
        )
        
        model.fit(self.X, self.y)
        
        # Cost should be decreasing
        costs = model.cost_history_
        assert len(costs) > 1
        
        # Check that cost generally decreases (allowing for some fluctuation)
        final_cost = np.mean(costs[-10:])  # Average of last 10 costs
        initial_cost = np.mean(costs[:10])  # Average of first 10 costs
        assert final_cost < initial_cost
    
    def test_get_params(self):
        """Test the get_params method."""
        model = LinearRegression(
            method='gradient_descent',
            learning_rate=0.05,
            max_iterations=500
        )
        
        params = model.get_params()
        
        expected_params = {
            'method': 'gradient_descent',
            'learning_rate': 0.05,
            'max_iterations': 500,
            'tolerance': 1e-6,  # default value
            'fit_intercept': True  # default value
        }
        
        assert params == expected_params
    
    def test_repr(self):
        """Test string representation."""
        model = LinearRegression()
        repr_str = repr(model)
        
        assert 'LinearRegression' in repr_str
        assert 'method=normal_equation' in repr_str
        assert 'learning_rate=0.01' in repr_str
    
    def test_method_comparison(self):
        """Test that both methods give similar results."""
        # Normal equation
        model_ne = LinearRegression(method='normal_equation')
        model_ne.fit(self.X, self.y)
        
        # Gradient descent
        model_gd = LinearRegression(
            method='gradient_descent',
            learning_rate=0.01,
            max_iterations=1000
        )
        model_gd.fit(self.X, self.y)
        
        # Both should give similar coefficients
        np.testing.assert_allclose(model_ne.coef_, model_gd.coef_, atol=1e-3)
        np.testing.assert_allclose(model_ne.intercept_, model_gd.intercept_, atol=1e-3)
        
        # Both should give similar predictions
        pred_ne = model_ne.predict(self.X)
        pred_gd = model_gd.predict(self.X)
        np.testing.assert_allclose(pred_ne, pred_gd, atol=1e-3)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])