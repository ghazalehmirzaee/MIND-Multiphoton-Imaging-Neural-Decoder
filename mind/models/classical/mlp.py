"""
Multilayer Perceptron model implementation for calcium imaging data.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class MLPModel:
    """
    Multilayer Perceptron model for decoding behavior from calcium imaging signals.

    This class implements an MLP classifier with hyperparameter optimization for
    decoding mouse forelimb movements from calcium imaging data.
    """

    def __init__(self,
                 hidden_layer_sizes: Tuple[int, ...] = (64, 128, 32),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 batch_size: str = 'auto',
                 learning_rate: str = 'adaptive',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 300,  
                 early_stopping: bool = True,
                 validation_fraction: float = 0.1,
                 n_iter_no_change: int = 15, 
                 random_state: int = 42,
                 optimize_hyperparams: bool = False):
        """
        Initialize the MLP model.
        """
        # Store hyperparameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.optimize_hyperparams = optimize_hyperparams

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            random_state=random_state,
            verbose=False  
        )

        # Initialize scaler for data normalization
        self.scaler = StandardScaler()

        logger.info(f"Initialized MLP model with hidden layers {hidden_layer_sizes}")

    def _prepare_data(self, X, y=None):
        """
        Prepare the data for the model.
        """
        # Convert torch tensors to numpy arrays if needed
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Reshape X to 2D if needed (n_samples, window_size * n_neurons)
        if X.ndim == 3:
            n_samples, window_size, n_neurons = X.shape
            X = X.reshape(n_samples, window_size * n_neurons)

        return X, y

    def optimize_hyperparameters(self, X_train, y_train, cv: int = 3, n_iter: int = 15):
        """
        Optimize model hyperparameters using RandomizedSearchCV.
        """
        logger.info("Optimizing MLP hyperparameters")

        # Prepare and scale data
        X_train, y_train = self._prepare_data(X_train, y_train)
        X_train = self.scaler.fit_transform(X_train)

        # Define parameter grid - focused for calcium imaging data
        param_grid = {
            'hidden_layer_sizes': [
                (64,), (128,),
                (64, 32), (128, 64),
                (64, 128, 32), (128, 256, 64) 
            ],
            'activation': ['relu', 'tanh'], 
            'alpha': [0.0001, 0.001, 0.01],  
            'learning_rate_init': [0.001, 0.005, 0.01],
            'batch_size': ['auto', 32, 64],  
            'solver': ['adam'] 
        }

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='balanced_accuracy',  
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Fit RandomizedSearchCV
        random_search.fit(X_train, y_train)

        # Get best parameters
        best_params = random_search.best_params_
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")

        # Update model with best parameters
        self.hidden_layer_sizes = best_params.get('hidden_layer_sizes', self.hidden_layer_sizes)
        self.activation = best_params.get('activation', self.activation)
        self.alpha = best_params.get('alpha', self.alpha)
        self.learning_rate_init = best_params.get('learning_rate_init', self.learning_rate_init)
        self.batch_size = best_params.get('batch_size', self.batch_size)
        self.solver = best_params.get('solver', self.solver)

        # Reinitialize model with best parameters
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
            verbose=False
        )

        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the MLP model.
        """
        logger.info("Training MLP model")

        # Prepare data
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Fit the scaler on training data
        X_train = self.scaler.fit_transform(X_train)

        # Optimize hyperparameters if requested
        if self.optimize_hyperparams:
            self.optimize_hyperparameters(X_train, y_train)

        # If validation data is provided and early stopping is enabled
        if X_val is not None and y_val is not None and self.early_stopping:
            X_val, y_val = self._prepare_data(X_val, y_val)
            X_val = self.scaler.transform(X_val)  

            self.model.fit(X_train, y_train)

            # Report validation score
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")
        else:
            # Use built-in early stopping
            self.model.fit(X_train, y_train)

        logger.info(f"MLP model training complete. Final loss: {self.model.loss_:.4f}")
        logger.info(f"Number of iterations: {self.model.n_iter_}")

        return self

    def predict(self, X):
        """
        Make predictions with the trained model.
        """
        # Prepare and scale data
        X, _ = self._prepare_data(X)
        X = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities.
        """
        # Prepare and scale data
        X, _ = self._prepare_data(X)
        X = self.scaler.transform(X)

        # Predict probabilities
        probabilities = self.model.predict_proba(X)

        return probabilities

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Estimate feature importance using weight magnitudes.

        This is a rough approximation based on the magnitude of weights in the first layer.
        """
        # Make sure the model is trained
        if not hasattr(self.model, 'coefs_'):
            raise ValueError("Model must be trained before getting feature importance")

        # Get weights from the first layer
        first_layer_weights = self.model.coefs_[0]  # Shape: (n_features, n_hidden_1)

        # Calculate feature importance as the sum of absolute weights
        feature_importance = np.abs(first_layer_weights).sum(axis=1)

        # Normalize feature importance
        feature_importance = feature_importance / feature_importance.sum()

        # Reshape to (window_size, n_neurons)
        feature_importance = feature_importance.reshape(window_size, n_neurons)

        return feature_importance

    
