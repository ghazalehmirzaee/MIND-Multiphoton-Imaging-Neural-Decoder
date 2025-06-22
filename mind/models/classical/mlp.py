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
                 max_iter: int = 300,  # Increased from 200 for better convergence
                 early_stopping: bool = True,
                 validation_fraction: float = 0.1,
                 n_iter_no_change: int = 15,  # Increased from 10
                 random_state: int = 42,
                 optimize_hyperparams: bool = False):
        """
        Initialize an MLP model.

        Parameters
        ----------
        hidden_layer_sizes : Tuple[int, ...], optional
            Hidden layer sizes, by default (64, 128, 32)
        activation : str, optional
            Activation function, by default 'relu'
        solver : str, optional
            Solver for weight optimization, by default 'adam'
        alpha : float, optional
            L2 penalty (regularization term) parameter, by default 0.0001
        batch_size : str, optional
            Batch size for gradient-based solvers, by default 'auto'
        learning_rate : str, optional
            Learning rate schedule, by default 'adaptive'
        learning_rate_init : float, optional
            Initial learning rate, by default 0.001
        max_iter : int, optional
            Maximum number of iterations, by default 300
        early_stopping : bool, optional
            Whether to use early stopping, by default True
        validation_fraction : float, optional
            Fraction of training data for validation, by default 0.1
        n_iter_no_change : int, optional
            Maximum number of epochs with no improvement, by default 15
        random_state : int, optional
            Random seed for reproducibility, by default 42
        optimize_hyperparams : bool, optional
            Whether to optimize hyperparameters, by default False
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

        # Initialize the model
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
            verbose=False  # Set to True for debugging
        )

        # Initialize scaler for data normalization
        self.scaler = StandardScaler()

        logger.info(f"Initialized MLP model with hidden layers {hidden_layer_sizes}")

    def _prepare_data(self, X, y=None):
        """
        Prepare the data for the model.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features, shape (n_samples, window_size, n_neurons)
        y : torch.Tensor or np.ndarray, optional
            Target labels, shape (n_samples,)

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Prepared X and y (if provided)
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

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        cv : int, optional
            Number of cross-validation folds, by default 3
        n_iter : int, optional
            Number of parameter settings sampled, by default 15

        Returns
        -------
        self
            The model with optimized hyperparameters
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
                (64, 128, 32), (128, 256, 64)  # Removed very deep architectures
            ],
            'activation': ['relu', 'tanh'],  # Removed 'logistic' - rarely optimal
            'alpha': [0.0001, 0.001, 0.01],  # Focused range
            'learning_rate_init': [0.001, 0.005, 0.01],
            'batch_size': ['auto', 32, 64],  # Removed 128 - often too large
            'solver': ['adam']  # Removed 'sgd' - adam is generally better
        }

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='balanced_accuracy',  # Better for imbalanced data
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

        Parameters
        ----------
        X_train : torch.Tensor or np.ndarray
            Training features
        y_train : torch.Tensor or np.ndarray
            Training labels
        X_val : torch.Tensor or np.ndarray, optional
            Validation features, by default None
        y_val : torch.Tensor or np.ndarray, optional
            Validation labels, by default None

        Returns
        -------
        self
            The trained model
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
            X_val = self.scaler.transform(X_val)  # Use transform, not fit_transform

            # SKLearn's MLPClassifier handles validation internally
            # We'll just use the built-in early stopping
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

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predicted labels
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

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predicted class probabilities
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

        Parameters
        ----------
        window_size : int
            Size of the sliding window
        n_neurons : int
            Number of neurons

        Returns
        -------
        np.ndarray
            Feature importance scores, shape (window_size, n_neurons)
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

#
# """
# Enhanced MLP with robust scale handling for extreme input ranges.
#
# SOLUTION: This implementation uses minimal scale conditioning (not full standardization)
# combined with ultra-conservative learning parameters for raw calcium signals.
# """
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import RandomizedSearchCV
# import logging
# from typing import Dict, Any, Optional, Tuple, List, Union
# import hashlib
#
# logger = logging.getLogger(__name__)
#
#
# class MLPModel:
#     """
#     Enhanced MLP with robust handling of extreme signal scales.
#
#     SOLUTION: Uses minimal scale conditioning (simple division) for numerical stability
#     while preserving the relative patterns and characteristics of each signal type.
#     """
#
#     def __init__(self,
#                  hidden_layer_sizes: Tuple[int, ...] = (64, 128, 32),
#                  activation: str = 'relu',
#                  solver: str = 'adam',
#                  alpha: float = 0.0001,
#                  batch_size: str = 'auto',
#                  learning_rate: str = 'adaptive',
#                  learning_rate_init: float = 0.001,
#                  max_iter: int = 1000,  # Increased for better convergence
#                  early_stopping: bool = True,
#                  validation_fraction: float = 0.1,
#                  n_iter_no_change: int = 30,  # More patience
#                  random_state: int = 42,
#                  optimize_hyperparams: bool = False,
#                  **kwargs):
#         """
#         Initialize enhanced MLP with robust scale handling.
#         """
#         # Store hyperparameters
#         self.hidden_layer_sizes = hidden_layer_sizes
#         self.activation = activation
#         self.solver = solver
#         self.alpha = alpha
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.learning_rate_init = learning_rate_init
#         self.max_iter = max_iter
#         self.early_stopping = early_stopping
#         self.validation_fraction = validation_fraction
#         self.n_iter_no_change = n_iter_no_change
#         self.random_state = random_state
#         self.optimize_hyperparams = optimize_hyperparams
#
#         # Log ignored parameters
#         if kwargs:
#             logger.info(f"MLP ignoring extra parameters: {list(kwargs.keys())}")
#
#         # Initialize with default parameters (will be adapted)
#         self.model = None
#         self.scale_factor = 1.0  # Will be determined based on input scale
#         self.adapted_params = {}
#         self.data_fingerprints = {}
#
#         logger.info(f"Initialized Enhanced MLP with robust scale handling")
#
#     def _determine_scale_conditioning(self, X):
#         """
#         Determine minimal scale conditioning needed for numerical stability.
#
#         SOLUTION: Instead of full standardization, we use simple division by a factor
#         that brings large numbers into a reasonable range while preserving patterns.
#         """
#         abs_mean = np.abs(X).mean()
#         data_max = np.abs(X).max()
#
#         logger.info(f"Analyzing input for scale conditioning:")
#         logger.info(f"  Absolute mean: {abs_mean:.6f}")
#         logger.info(f"  Maximum absolute value: {data_max:.6f}")
#
#         if abs_mean > 1000:  # Raw calcium range
#             # Scale down by a factor that brings values to hundreds, not units
#             # This preserves the relative patterns while preventing numerical issues
#             self.scale_factor = 1000.0  # Divide by 1000: 6000 -> 6.0
#             scale_type = "LARGE_SCALE_CONDITIONING (Raw Calcium)"
#
#             # Ultra-conservative parameters for large scale
#             self.adapted_params = {
#                 'learning_rate_init': self.learning_rate_init * 0.001,  # 1000x smaller!
#                 'alpha': self.alpha * 100,  # Much more regularization
#                 'max_iter': self.max_iter * 2,  # More training time
#                 'early_stopping': True,
#                 'n_iter_no_change': 50,  # Much more patience
#                 'validation_fraction': 0.15  # Larger validation set
#             }
#
#         elif abs_mean < 0.01:  # Deconvolved range
#             # No scaling needed, but boost learning for sparse signals
#             self.scale_factor = 1.0
#             scale_type = "SPARSE_SIGNAL_BOOST (Deconvolved)"
#
#             self.adapted_params = {
#                 'learning_rate_init': self.learning_rate_init * 3.0,  # Boost learning
#                 'alpha': self.alpha * 0.1,  # Less regularization
#                 'max_iter': self.max_iter * 2,  # More time for sparse learning
#                 'early_stopping': True,
#                 'n_iter_no_change': 40
#             }
#
#         else:  # ΔF/F range - ideal scale
#             self.scale_factor = 1.0
#             scale_type = "OPTIMAL_SCALE (ΔF/F)"
#
#             self.adapted_params = {
#                 'learning_rate_init': self.learning_rate_init,
#                 'alpha': self.alpha,
#                 'max_iter': self.max_iter,
#                 'early_stopping': self.early_stopping,
#                 'n_iter_no_change': self.n_iter_no_change
#             }
#
#         logger.info(f"Scale conditioning strategy: {scale_type}")
#         logger.info(f"  Scale factor: {self.scale_factor}")
#         logger.info(f"  Conditioned range: [{data_max / self.scale_factor:.3f}] (max)")
#         logger.info(f"  Learning rate: {self.learning_rate_init:.6f} → {self.adapted_params['learning_rate_init']:.6f}")
#
#         return scale_type
#
#     def _condition_data(self, X):
#         """
#         Apply minimal scale conditioning while preserving signal characteristics.
#
#         SOLUTION: Simple division that maintains all relative patterns and relationships
#         while bringing extreme values into a numerically stable range.
#         """
#         if self.scale_factor != 1.0:
#             X_conditioned = X / self.scale_factor
#
#             logger.info(f"Applied scale conditioning:")
#             logger.info(f"  Original range: [{X.min():.3f}, {X.max():.3f}]")
#             logger.info(f"  Conditioned range: [{X_conditioned.min():.3f}, {X_conditioned.max():.3f}]")
#             logger.info(f"  Pattern preservation: All relative relationships maintained")
#
#             return X_conditioned
#         else:
#             return X
#
#     def _prepare_data(self, X, y=None):
#         """
#         Prepare data with minimal scale conditioning.
#         """
#         # Convert torch tensors to numpy arrays if needed
#         if hasattr(X, 'numpy'):
#             X = X.numpy()
#         if y is not None and hasattr(y, 'numpy'):
#             y = y.numpy()
#
#         # Reshape X to 2D if needed
#         original_shape = X.shape
#         if X.ndim == 3:
#             n_samples, window_size, n_neurons = X.shape
#             X = X.reshape(n_samples, window_size * n_neurons)
#
#         # Apply scale conditioning if needed
#         X_conditioned = self._condition_data(X)
#
#         logger.info(f"MLP data preparation with minimal conditioning:")
#         logger.info(f"  Shape: {original_shape} → {X_conditioned.shape}")
#         logger.info(f"  Final range: [{X_conditioned.min():.6f}, {X_conditioned.max():.6f}]")
#         logger.info(f"  Final mean: {X_conditioned.mean():.6f}")
#
#         return X_conditioned, y
#
#     def fit(self, X_train, y_train, X_val=None, y_val=None):
#         """
#         Train MLP with robust scale handling.
#         """
#         logger.info("Training Enhanced MLP with robust scale handling")
#
#         # Determine scale conditioning strategy
#         # We analyze the original data to decide on conditioning
#         X_analysis = X_train.numpy() if hasattr(X_train, 'numpy') else X_train
#         if X_analysis.ndim == 3:
#             X_analysis = X_analysis.reshape(X_analysis.shape[0], -1)
#
#         scale_type = self._determine_scale_conditioning(X_analysis)
#
#         # Create model with adapted parameters
#         self.model = MLPClassifier(
#             hidden_layer_sizes=self.hidden_layer_sizes,
#             activation=self.activation,
#             solver=self.solver,
#             alpha=self.adapted_params['alpha'],
#             batch_size=self.batch_size,
#             learning_rate=self.learning_rate,
#             learning_rate_init=self.adapted_params['learning_rate_init'],
#             max_iter=self.adapted_params['max_iter'],
#             early_stopping=self.adapted_params['early_stopping'],
#             validation_fraction=self.validation_fraction,
#             n_iter_no_change=self.adapted_params['n_iter_no_change'],
#             random_state=self.random_state,
#             verbose=False
#         )
#
#         # Prepare training data with conditioning
#         X_train, y_train = self._prepare_data(X_train, y_train)
#
#         # SOLUTION: Additional safeguards for extreme scales
#         if scale_type == "LARGE_SCALE_CONDITIONING (Raw Calcium)":
#             # Check class distribution and warn about potential issues
#             unique_labels, counts = np.unique(y_train, return_counts=True)
#             class_ratio = counts.min() / counts.max()
#
#             logger.info(f"Class distribution analysis for large scale:")
#             logger.info(f"  Classes: {unique_labels}")
#             logger.info(f"  Counts: {counts}")
#             logger.info(f"  Minority/Majority ratio: {class_ratio:.3f}")
#
#             if class_ratio < 0.3:
#                 logger.warning("⚠️ Severe class imbalance detected with large scale")
#                 logger.warning("   This combination is challenging - monitoring convergence carefully")
#
#         # Train with enhanced monitoring
#         try:
#             self.model.fit(X_train, y_train)
#             training_successful = True
#
#             logger.info(f"Enhanced MLP training complete for {scale_type}")
#             logger.info(f"  Final loss: {self.model.loss_:.6f}")
#             logger.info(f"  Iterations used: {self.model.n_iter_}")
#             logger.info(f"  Converged: {self.model.n_iter_ < self.model.max_iter}")
#
#             # SOLUTION: Check for classifier collapse
#             train_predictions = self.model.predict(X_train)
#             unique_predictions = np.unique(train_predictions)
#
#             if len(unique_predictions) == 1:
#                 logger.error(f"🚨 CLASSIFIER COLLAPSE DETECTED!")
#                 logger.error(f"   Model only predicts class: {unique_predictions[0]}")
#                 logger.error(f"   This indicates learning failure - consider more aggressive conditioning")
#             else:
#                 logger.info(f"✅ Model predicts both classes: {unique_predictions}")
#
#         except Exception as e:
#             logger.error(f"Enhanced MLP training failed: {e}")
#             training_successful = False
#             raise
#
#         # Validation if provided
#         if X_val is not None and y_val is not None:
#             X_val, y_val = self._prepare_data(X_val, y_val)
#             val_score = self.model.score(X_val, y_val)
#
#             # Check validation predictions too
#             val_predictions = self.model.predict(X_val)
#             val_unique = np.unique(val_predictions)
#
#             logger.info(f"  Validation accuracy: {val_score:.4f}")
#             logger.info(f"  Validation predictions: {val_unique}")
#
#             if len(val_unique) == 1:
#                 logger.warning(f"⚠️ Validation also shows single-class prediction")
#
#         return self
#
#     def predict(self, X):
#         """Make predictions with scale conditioning."""
#         if self.model is None:
#             raise ValueError("Model not trained. Call fit() first.")
#
#         X, _ = self._prepare_data(X)
#         return self.model.predict(X)
#
#     def predict_proba(self, X):
#         """Predict probabilities with scale conditioning."""
#         if self.model is None:
#             raise ValueError("Model not trained. Call fit() first.")
#
#         X, _ = self._prepare_data(X)
#         return self.model.predict_proba(X)
#
#     def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
#         """Get feature importance accounting for scale conditioning."""
#         if self.model is None or not hasattr(self.model, 'coefs_'):
#             raise ValueError("Model must be trained before getting feature importance")
#
#         first_layer_weights = self.model.coefs_[0]
#         feature_importance = np.abs(first_layer_weights).sum(axis=1)
#
#         if feature_importance.sum() > 0:
#             feature_importance = feature_importance / feature_importance.sum()
#
#         n_features = min(len(feature_importance), window_size * n_neurons)
#         importance_matrix = feature_importance[:n_features].reshape(window_size, n_neurons)
#
#         logger.info(f"Feature importance extracted with scale factor: {self.scale_factor}")
#         return importance_matrix
#
#     def get_conditioning_info(self):
#         """Return information about the conditioning applied."""
#         return {
#             'scale_factor': self.scale_factor,
#             'adapted_params': self.adapted_params,
#             'conditioning_type': 'minimal_division' if self.scale_factor != 1.0 else 'none'
#         }
#
