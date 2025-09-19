"""
Linear Regression Implementation

This module contains a from-scratch implementation of Linear Regression
using the Normal Equation and Gradient Descent methods.

Author: ML-Algo Contributors
Date: 2025
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union


class LinearRegression:
    """
    Linear Regression implementation using Normal Equation and Gradient Descent.
    
    Linear regression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.
    
    Mathematical Foundation:
    -----------------------
    The linear model assumes that the target variable y is a linear combination
    of the input features X:
    
    y = X * w + b
    
    Where:
    - y: target variable (n_samples,)
    - X: input features (n_samples, n_features)  
    - w: coefficients/weights (n_features,)
    - b: intercept/bias term (scalar)
    
    Cost Function (Mean Squared Error):
    J(w,b) = (1/2m) * Σ(h(x^(i)) - y^(i))^2
    
    Where:
    - m: number of training examples
    - h(x^(i)): predicted value for i-th example
    - y^(i)): actual value for i-th example
    
    Parameters:
    -----------
    method : str, default='normal_equation'
        Method to use for fitting:
        - 'normal_equation': Direct analytical solution
        - 'gradient_descent': Iterative optimization
    learning_rate : float, default=0.01
        Step size for gradient descent (only used if method='gradient_descent')
    max_iterations : int, default=1000
        Maximum number of iterations for gradient descent
    tolerance : float, default=1e-6
        Convergence tolerance for gradient descent
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model
    
    Attributes:
    -----------
    coef_ : ndarray, shape (n_features,)
        Estimated coefficients for the linear regression problem
    intercept_ : float
        Independent term in the linear model
    cost_history_ : list
        Cost function values during gradient descent training
    n_iterations_ : int
        Number of iterations used in gradient descent
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from algorithms.supervised.regression import LinearRegression
    >>> 
    >>> # Generate sample data
    >>> X = np.random.randn(100, 2)
    >>> y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + np.random.randn(100) * 0.1
    >>> 
    >>> # Fit using normal equation (default)
    >>> model = LinearRegression()
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    >>> 
    >>> # Fit using gradient descent
    >>> model_gd = LinearRegression(method='gradient_descent', learning_rate=0.01)
    >>> model_gd.fit(X, y)
    >>> 
    >>> print(f"Coefficients: {model.coef_}")
    >>> print(f"Intercept: {model.intercept_}")
    
    References:
    -----------
    [1] Hastie, T., Tibshirani, R., & Friedman, J. (2009). 
        The elements of statistical learning: data mining, inference, and prediction.
    [2] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). 
        An introduction to statistical learning.
    """
    
    def __init__(
        self,
        method: str = 'normal_equation',
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        fit_intercept: bool = True
    ):
        """
        Initialize Linear Regression with specified parameters.
        
        Parameters:
        -----------
        method : str, default='normal_equation'
            Optimization method to use. Options:
            - 'normal_equation': Fast, direct solution but requires matrix inversion
            - 'gradient_descent': Slower but works with large datasets and singular matrices
        learning_rate : float, default=0.01
            Learning rate for gradient descent. Higher values converge faster but may overshoot
            Typical range: [0.001, 0.1]
        max_iterations : int, default=1000
            Maximum iterations for gradient descent to prevent infinite loops
        tolerance : float, default=1e-6
            Convergence criteria - stop when cost change is smaller than this value
        fit_intercept : bool, default=True
            Whether to fit an intercept term. Set to False if data is already centered
        """
        # Validate method parameter
        valid_methods = ['normal_equation', 'gradient_descent']
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")
        
        # Validate numerical parameters
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        
        # Store hyperparameters
        self.method = method
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        
        # Initialize model parameters (set during fitting)
        self.coef_ = None
        self.intercept_ = 0.0
        self.cost_history_ = []
        self.n_iterations_ = 0
        self.is_fitted_ = False
        
        # Store data information
        self.n_features_ = None
        self.n_samples_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model to training data.
        
        This method finds the optimal coefficients that minimize the mean squared error
        between predicted and actual target values using either the normal equation
        or gradient descent optimization.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training input samples. Each row is a sample, each column is a feature.
        y : array-like, shape (n_samples,)
            Target values (real numbers). Must have same number of samples as X.
        
        Returns:
        --------
        self : LinearRegression
            Returns the instance itself for method chaining.
        
        Algorithm Steps:
        ----------------
        1. Validate and preprocess input data
        2. Add intercept column if fit_intercept=True
        3. Apply selected optimization method:
           - Normal Equation: θ = (X^T * X)^(-1) * X^T * y
           - Gradient Descent: Iteratively update θ using gradient information
        4. Extract coefficients and intercept from solution
        
        Mathematical Details:
        --------------------
        Normal Equation:
        The normal equation provides the analytical solution by setting the gradient
        of the cost function to zero and solving for the parameters:
        
        ∇J(θ) = X^T(Xθ - y) = 0
        X^T*X*θ = X^T*y
        θ = (X^T*X)^(-1)*X^T*y
        
        Gradient Descent:
        Iteratively updates parameters in the direction of steepest descent:
        
        θ := θ - α * ∇J(θ)
        where ∇J(θ) = (1/m) * X^T * (X*θ - y)
        
        Time Complexity:
        - Normal Equation: O(n^3) due to matrix inversion
        - Gradient Descent: O(k*m*n) where k is iterations, m is samples, n is features
        
        Space Complexity: O(n^2) for normal equation, O(n) for gradient descent
        """
        # Step 1: Validate input data
        X, y = self._validate_data(X, y)
        
        # Store dataset information
        self.n_samples_, self.n_features_ = X.shape
        
        # Step 2: Add intercept column if needed
        if self.fit_intercept:
            # Add column of ones for intercept term
            # Shape becomes (n_samples, n_features + 1)
            X_with_intercept = np.column_stack([np.ones(self.n_samples_), X])
        else:
            X_with_intercept = X.copy()
        
        # Step 3: Apply selected optimization method
        if self.method == 'normal_equation':
            theta = self._fit_normal_equation(X_with_intercept, y)
        else:  # gradient_descent
            theta = self._fit_gradient_descent(X_with_intercept, y)
        
        # Step 4: Extract coefficients and intercept
        if self.fit_intercept:
            self.intercept_ = theta[0]  # First element is intercept
            self.coef_ = theta[1:]      # Remaining elements are feature coefficients
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
        
        # Mark as fitted
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for input samples.
        
        Uses the fitted linear model to make predictions on new data:
        y_pred = X * coef_ + intercept_
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input samples to predict. Must have same number of features as training data.
        
        Returns:
        --------
        y_pred : ndarray, shape (n_samples,)
            Predicted target values for input samples.
        
        Raises:
        -------
        ValueError
            If model has not been fitted yet or if X has wrong number of features.
        """
        # Check if model has been fitted
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        # Validate input data
        X = self._validate_input(X)
        
        # Check feature dimension consistency
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained on {self.n_features_} features"
            )
        
        # Make predictions using linear combination
        # y = X * w + b
        predictions = X @ self.coef_ + self.intercept_
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score (coefficient of determination).
        
        R² represents the proportion of variance in the dependent variable
        that is predictable from the independent variables.
        
        R² = 1 - SS_res / SS_tot
        where:
        - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
        - SS_tot = Σ(y_true - y_mean)² (total sum of squares)
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test input samples
        y : array-like, shape (n_samples,)
            True target values
        
        Returns:
        --------
        score : float
            R² score. Range: (-∞, 1]. 
            - 1.0: Perfect prediction
            - 0.0: Model performs as well as simply predicting the mean
            - Negative: Model performs worse than predicting the mean
        """
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate R² score
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        
        # Handle edge case where all y values are the same
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score
    
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit using the normal equation method.
        
        Solves the linear regression analytically using:
        θ = (X^T * X)^(-1) * X^T * y
        
        This method is fast for small to medium datasets but becomes
        computationally expensive for large feature sets due to matrix inversion.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features + intercept)
            Design matrix with intercept column if fit_intercept=True
        y : ndarray, shape (n_samples,)
            Target values
        
        Returns:
        --------
        theta : ndarray, shape (n_features + intercept,)
            Optimal parameters including intercept if applicable
        
        Notes:
        ------
        - Requires X^T * X to be invertible (non-singular)
        - Time complexity: O(n³) where n is number of features
        - May be numerically unstable for ill-conditioned matrices
        """
        try:
            # Calculate X transpose
            X_transpose = X.T
            
            # Calculate (X^T * X)
            XTX = X_transpose @ X
            
            # Calculate (X^T * y)
            XTy = X_transpose @ y
            
            # Solve the normal equation: θ = (X^T * X)^(-1) * X^T * y
            # Using np.linalg.solve is more stable than computing inverse directly
            theta = np.linalg.solve(XTX, XTy)
            
            return theta
            
        except np.linalg.LinAlgError:
            # Matrix is singular (not invertible)
            raise ValueError(
                "Normal equation failed due to singular matrix. "
                "Try using method='gradient_descent' or add regularization."
            )
    
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit using gradient descent optimization.
        
        Iteratively updates parameters to minimize the cost function:
        
        θ := θ - α * ∇J(θ)
        
        where:
        - α is the learning rate
        - ∇J(θ) = (1/m) * X^T * (X*θ - y) is the gradient
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features + intercept)
            Design matrix with intercept column if fit_intercept=True
        y : ndarray, shape (n_samples,)
            Target values
        
        Returns:
        --------
        theta : ndarray, shape (n_features + intercept,)
            Optimized parameters after convergence
        
        Algorithm Details:
        -----------------
        1. Initialize parameters randomly or to zero
        2. For each iteration:
           a. Calculate predictions: h = X * θ
           b. Calculate cost: J = (1/2m) * Σ(h - y)²
           c. Calculate gradient: ∇J = (1/m) * X^T * (h - y)
           d. Update parameters: θ := θ - α * ∇J
           e. Check convergence criteria
        3. Return optimized parameters
        
        Convergence Criteria:
        --------------------
        - Maximum iterations reached, OR
        - Cost change < tolerance between consecutive iterations
        """
        m, n = X.shape  # m = samples, n = features (including intercept if applicable)
        
        # Initialize parameters
        # Small random initialization can help with convergence
        np.random.seed(42)  # For reproducible results
        theta = np.random.normal(0, 0.01, n)
        
        # Initialize cost history for tracking convergence
        self.cost_history_ = []
        
        # Gradient descent main loop
        for iteration in range(self.max_iterations):
            # Forward pass: calculate predictions
            predictions = X @ theta
            
            # Calculate cost (Mean Squared Error)
            cost = np.mean((predictions - y) ** 2) / 2
            self.cost_history_.append(cost)
            
            # Calculate gradient
            # ∇J(θ) = (1/m) * X^T * (X*θ - y)
            gradient = (X.T @ (predictions - y)) / m
            
            # Update parameters
            theta = theta - self.learning_rate * gradient
            
            # Check convergence (cost change tolerance)
            if iteration > 0:
                cost_change = abs(self.cost_history_[-2] - self.cost_history_[-1])
                if cost_change < self.tolerance:
                    break
        
        # Store number of iterations used
        self.n_iterations_ = iteration + 1
        
        return theta
    
    def _validate_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and preprocess input data for training.
        
        Performs comprehensive validation and preprocessing:
        1. Convert to numpy arrays
        2. Check shapes and dimensions
        3. Handle missing values
        4. Ensure numerical data types
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Target values
        
        Returns:
        --------
        X_validated : ndarray, shape (n_samples, n_features)
            Validated and preprocessed feature matrix
        y_validated : ndarray, shape (n_samples,)
            Validated and preprocessed target vector
        
        Raises:
        -------
        ValueError
            If data has incorrect shape, contains invalid values, or has mismatched dimensions
        """
        # Convert to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Validate dimensions
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        
        # Check sample size consistency
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {X.shape[0]} samples, y: {y.shape[0]} samples"
            )
        
        # Check for minimum sample size
        if X.shape[0] == 0:
            raise ValueError("Cannot fit model with 0 samples")
        
        # Check for minimum feature count
        if X.shape[1] == 0:
            raise ValueError("X must have at least 1 feature")
        
        # Check for infinite or NaN values
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or infinite values")
        
        return X, y
    
    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate input data for prediction.
        
        Lighter validation for prediction phase since we assume
        the data format is consistent with training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features for prediction
        
        Returns:
        --------
        X_validated : ndarray, shape (n_samples, n_features)
            Validated feature matrix
        """
        # Convert to numpy array
        X = np.asarray(X, dtype=np.float64)
        
        # Basic validation
        if X.ndim == 1:
            # Single sample case - reshape to 2D
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        # Check for invalid values
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        
        return X
    
    def plot_cost_history(self) -> None:
        """
        Plot the cost function history during gradient descent training.
        
        This method is only available if the model was trained using
        gradient descent method. Useful for:
        - Debugging convergence issues
        - Tuning learning rate
        - Understanding training dynamics
        
        Raises:
        -------
        ValueError
            If model was not fitted using gradient descent
        """
        if self.method != 'gradient_descent':
            raise ValueError("Cost history is only available for gradient descent method")
        
        if not self.cost_history_:
            raise ValueError("No cost history available. Model may not be fitted.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history_, 'b-', linewidth=2)
        plt.title('Cost Function During Training', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (MSE)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"Final cost: {self.cost_history_[-1]:.6f}")
        print(f"Converged after {self.n_iterations_} iterations")
    
    def get_params(self) -> dict:
        """
        Get parameters for this estimator.
        
        Returns:
        --------
        params : dict
            Parameter names mapped to their values
        """
        return {
            'method': self.method,
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations,
            'tolerance': self.tolerance,
            'fit_intercept': self.fit_intercept
        }
    
    def __repr__(self) -> str:
        """String representation of the LinearRegression object."""
        params = self.get_params()
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        return f"LinearRegression({param_str})"