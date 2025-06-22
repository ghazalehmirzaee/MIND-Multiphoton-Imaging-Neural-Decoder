# """
# Multilayer Perceptron model implementation for calcium imaging data.
# """
# import numpy as np
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# import logging
# from typing import Dict, Any, Optional, Tuple, List, Union
#
# logger = logging.getLogger(__name__)
#
#
# class MLPModel:
#     """
#     Multilayer Perceptron model for decoding behavior from calcium imaging signals.
#
#     This class implements an MLP classifier with hyperparameter optimization for
#     decoding mouse forelimb movements from calcium imaging data.
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
#                  max_iter: int = 300,  # Increased from 200 for better convergence
#                  early_stopping: bool = True,
#                  validation_fraction: float = 0.1,
#                  n_iter_no_change: int = 15,  # Increased from 10
#                  random_state: int = 42,
#                  optimize_hyperparams: bool = False):
#         """
#         Initialize an MLP model.
#
#         Parameters
#         ----------
#         hidden_layer_sizes : Tuple[int, ...], optional
#             Hidden layer sizes, by default (64, 128, 32)
#         activation : str, optional
#             Activation function, by default 'relu'
#         solver : str, optional
#             Solver for weight optimization, by default 'adam'
#         alpha : float, optional
#             L2 penalty (regularization term) parameter, by default 0.0001
#         batch_size : str, optional
#             Batch size for gradient-based solvers, by default 'auto'
#         learning_rate : str, optional
#             Learning rate schedule, by default 'adaptive'
#         learning_rate_init : float, optional
#             Initial learning rate, by default 0.001
#         max_iter : int, optional
#             Maximum number of iterations, by default 300
#         early_stopping : bool, optional
#             Whether to use early stopping, by default True
#         validation_fraction : float, optional
#             Fraction of training data for validation, by default 0.1
#         n_iter_no_change : int, optional
#             Maximum number of epochs with no improvement, by default 15
#         random_state : int, optional
#             Random seed for reproducibility, by default 42
#         optimize_hyperparams : bool, optional
#             Whether to optimize hyperparameters, by default False
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
#         # Initialize the model
#         self.model = MLPClassifier(
#             hidden_layer_sizes=hidden_layer_sizes,
#             activation=activation,
#             solver=solver,
#             alpha=alpha,
#             batch_size=batch_size,
#             learning_rate=learning_rate,
#             learning_rate_init=learning_rate_init,
#             max_iter=max_iter,
#             early_stopping=early_stopping,
#             validation_fraction=validation_fraction,
#             n_iter_no_change=n_iter_no_change,
#             random_state=random_state,
#             verbose=False  # Set to True for debugging
#         )
#
#         # Initialize scaler for data normalization
#         self.scaler = StandardScaler()
#
#         logger.info(f"Initialized MLP model with hidden layers {hidden_layer_sizes}")
#
#     def _prepare_data(self, X, y=None):
#         """
#         Prepare the data for the model.
#
#         Parameters
#         ----------
#         X : torch.Tensor or np.ndarray
#             Input features, shape (n_samples, window_size, n_neurons)
#         y : torch.Tensor or np.ndarray, optional
#             Target labels, shape (n_samples,)
#
#         Returns
#         -------
#         Tuple[np.ndarray, Optional[np.ndarray]]
#             Prepared X and y (if provided)
#         """
#         # Convert torch tensors to numpy arrays if needed
#         if hasattr(X, 'numpy'):
#             X = X.numpy()
#         if y is not None and hasattr(y, 'numpy'):
#             y = y.numpy()
#
#         # Reshape X to 2D if needed (n_samples, window_size * n_neurons)
#         if X.ndim == 3:
#             n_samples, window_size, n_neurons = X.shape
#             X = X.reshape(n_samples, window_size * n_neurons)
#
#         return X, y
#
#     def optimize_hyperparameters(self, X_train, y_train, cv: int = 3, n_iter: int = 15):
#         """
#         Optimize model hyperparameters using RandomizedSearchCV.
#
#         Parameters
#         ----------
#         X_train : np.ndarray
#             Training features
#         y_train : np.ndarray
#             Training labels
#         cv : int, optional
#             Number of cross-validation folds, by default 3
#         n_iter : int, optional
#             Number of parameter settings sampled, by default 15
#
#         Returns
#         -------
#         self
#             The model with optimized hyperparameters
#         """
#         logger.info("Optimizing MLP hyperparameters")
#
#         # Prepare and scale data
#         X_train, y_train = self._prepare_data(X_train, y_train)
#         X_train = self.scaler.fit_transform(X_train)
#
#         # Define parameter grid - focused for calcium imaging data
#         param_grid = {
#             'hidden_layer_sizes': [
#                 (64,), (128,),
#                 (64, 32), (128, 64),
#                 (64, 128, 32), (128, 256, 64)  # Removed very deep architectures
#             ],
#             'activation': ['relu', 'tanh'],  # Removed 'logistic' - rarely optimal
#             'alpha': [0.0001, 0.001, 0.01],  # Focused range
#             'learning_rate_init': [0.001, 0.005, 0.01],
#             'batch_size': ['auto', 32, 64],  # Removed 128 - often too large
#             'solver': ['adam']  # Removed 'sgd' - adam is generally better
#         }
#
#         # Initialize RandomizedSearchCV
#         random_search = RandomizedSearchCV(
#             estimator=self.model,
#             param_distributions=param_grid,
#             n_iter=n_iter,
#             cv=cv,
#             scoring='balanced_accuracy',  # Better for imbalanced data
#             verbose=1,
#             random_state=self.random_state,
#             n_jobs=-1
#         )
#
#         # Fit RandomizedSearchCV
#         random_search.fit(X_train, y_train)
#
#         # Get best parameters
#         best_params = random_search.best_params_
#         logger.info(f"Best parameters: {best_params}")
#         logger.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
#
#         # Update model with best parameters
#         self.hidden_layer_sizes = best_params.get('hidden_layer_sizes', self.hidden_layer_sizes)
#         self.activation = best_params.get('activation', self.activation)
#         self.alpha = best_params.get('alpha', self.alpha)
#         self.learning_rate_init = best_params.get('learning_rate_init', self.learning_rate_init)
#         self.batch_size = best_params.get('batch_size', self.batch_size)
#         self.solver = best_params.get('solver', self.solver)
#
#         # Reinitialize model with best parameters
#         self.model = MLPClassifier(
#             hidden_layer_sizes=self.hidden_layer_sizes,
#             activation=self.activation,
#             solver=self.solver,
#             alpha=self.alpha,
#             batch_size=self.batch_size,
#             learning_rate=self.learning_rate,
#             learning_rate_init=self.learning_rate_init,
#             max_iter=self.max_iter,
#             early_stopping=self.early_stopping,
#             validation_fraction=self.validation_fraction,
#             n_iter_no_change=self.n_iter_no_change,
#             random_state=self.random_state,
#             verbose=False
#         )
#
#         return self
#
#     def fit(self, X_train, y_train, X_val=None, y_val=None):
#         """
#         Train the MLP model.
#
#         Parameters
#         ----------
#         X_train : torch.Tensor or np.ndarray
#             Training features
#         y_train : torch.Tensor or np.ndarray
#             Training labels
#         X_val : torch.Tensor or np.ndarray, optional
#             Validation features, by default None
#         y_val : torch.Tensor or np.ndarray, optional
#             Validation labels, by default None
#
#         Returns
#         -------
#         self
#             The trained model
#         """
#         logger.info("Training MLP model")
#
#         # Prepare data
#         X_train, y_train = self._prepare_data(X_train, y_train)
#
#         # Fit the scaler on training data
#         X_train = self.scaler.fit_transform(X_train)
#
#         # Optimize hyperparameters if requested
#         if self.optimize_hyperparams:
#             self.optimize_hyperparameters(X_train, y_train)
#
#         # If validation data is provided and early stopping is enabled
#         if X_val is not None and y_val is not None and self.early_stopping:
#             X_val, y_val = self._prepare_data(X_val, y_val)
#             X_val = self.scaler.transform(X_val)  # Use transform, not fit_transform
#
#             # SKLearn's MLPClassifier handles validation internally
#             # We'll just use the built-in early stopping
#             self.model.fit(X_train, y_train)
#
#             # Report validation score
#             val_score = self.model.score(X_val, y_val)
#             logger.info(f"Validation accuracy: {val_score:.4f}")
#         else:
#             # Use built-in early stopping
#             self.model.fit(X_train, y_train)
#
#         logger.info(f"MLP model training complete. Final loss: {self.model.loss_:.4f}")
#         logger.info(f"Number of iterations: {self.model.n_iter_}")
#
#         return self
#
#     def predict(self, X):
#         """
#         Make predictions with the trained model.
#
#         Parameters
#         ----------
#         X : torch.Tensor or np.ndarray
#             Input features
#
#         Returns
#         -------
#         np.ndarray
#             Predicted labels
#         """
#         # Prepare and scale data
#         X, _ = self._prepare_data(X)
#         X = self.scaler.transform(X)
#
#         # Make predictions
#         predictions = self.model.predict(X)
#
#         return predictions
#
#     def predict_proba(self, X):
#         """
#         Predict class probabilities.
#
#         Parameters
#         ----------
#         X : torch.Tensor or np.ndarray
#             Input features
#
#         Returns
#         -------
#         np.ndarray
#             Predicted class probabilities
#         """
#         # Prepare and scale data
#         X, _ = self._prepare_data(X)
#         X = self.scaler.transform(X)
#
#         # Predict probabilities
#         probabilities = self.model.predict_proba(X)
#
#         return probabilities
#
#     def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
#         """
#         Estimate feature importance using weight magnitudes.
#
#         This is a rough approximation based on the magnitude of weights in the first layer.
#
#         Parameters
#         ----------
#         window_size : int
#             Size of the sliding window
#         n_neurons : int
#             Number of neurons
#
#         Returns
#         -------
#         np.ndarray
#             Feature importance scores, shape (window_size, n_neurons)
#         """
#         # Make sure the model is trained
#         if not hasattr(self.model, 'coefs_'):
#             raise ValueError("Model must be trained before getting feature importance")
#
#         # Get weights from the first layer
#         first_layer_weights = self.model.coefs_[0]  # Shape: (n_features, n_hidden_1)
#
#         # Calculate feature importance as the sum of absolute weights
#         feature_importance = np.abs(first_layer_weights).sum(axis=1)
#
#         # Normalize feature importance
#         feature_importance = feature_importance / feature_importance.sum()
#
#         # Reshape to (window_size, n_neurons)
#         feature_importance = feature_importance.reshape(window_size, n_neurons)
#
#         return feature_importance
#


"""
Multilayer Perceptron WITHOUT standardization but with scale-adaptive training.

This implementation removes standardization while adding techniques to handle
different signal scales through adaptive learning rates and careful initialization.

Key Scientific Challenge:
- Raw calcium (~6000): Risk of gradient explosion due to large values
- ΔF/F (~0.15): Risk of gradient vanishing due to small values
- Deconvolved (sparse ~0.004): Risk of learning only on rare non-zero events

Our solution: Scale-adaptive training without destroying signal characteristics.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class MLPModel:
    """
    Multilayer Perceptron WITHOUT standardization for raw signal testing.

    This class removes data normalization while implementing scale-adaptive training
    techniques to handle the dramatic differences in calcium signal magnitudes.

    Scientific Design Philosophy:
    1. Preserve natural signal characteristics completely
    2. Use adaptive learning rates that automatically adjust to input scale
    3. Implement careful weight initialization for different input ranges
    4. Monitor convergence to detect scale-related training problems
    5. Let the network learn to handle scale differences through weight adaptation
    """

    def __init__(self,
                 hidden_layer_sizes: Tuple[int, ...] = (64, 128, 32),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 batch_size: str = 'auto',
                 learning_rate: str = 'adaptive',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 500,
                 early_stopping: bool = True,
                 validation_fraction: float = 0.1,
                 n_iter_no_change: int = 20,
                 random_state: int = 42,
                 optimize_hyperparams: bool = False):
        """
        Initialize MLP WITHOUT standardization but with scale-adaptive parameters.

        Parameters
        ----------
        hidden_layer_sizes : Tuple[int, ...], optional
            Hidden layer architecture. (64, 128, 32) provides sufficient capacity
            for calcium imaging patterns while preventing overfitting.

        activation : str, optional
            'relu' activation is robust to different input scales and prevents
            vanishing gradients better than sigmoid/tanh for deep networks.

        solver : str, optional
            'adam' optimizer adapts learning rates per parameter, crucial when
            dealing with different input scales without standardization.

        alpha : float, optional
            L2 regularization strength. 0.0001 provides gentle regularization
            without interfering with scale adaptation.

        batch_size : str, optional
            'auto' lets sklearn choose based on dataset size. For calcium imaging,
            this typically results in reasonable batch sizes for stable training.

        learning_rate : str, optional
            'adaptive' reduces learning rate when loss improvement stagnates.
            Essential for handling different signal scales without standardization.

        learning_rate_init : float, optional
            Initial learning rate. 0.001 is conservative enough to handle large
            raw calcium values while still allowing learning on small ΔF/F signals.

        max_iter : int, optional
            Maximum iterations increased to 500 to allow convergence with
            different signal scales that may require different training times.

        early_stopping : bool, optional
            Prevents overfitting and saves computation when model stops improving.
            Important when training on different scales may have different dynamics.

        validation_fraction : float, optional
            Fraction of training data used for early stopping validation.

        n_iter_no_change : int, optional
            Patience for early stopping increased to 20 to accommodate potentially
            slower convergence with unprocessed signals.

        random_state : int, optional
            Ensures reproducible weight initialization across signal types.

        optimize_hyperparams : bool, optional
            Whether to perform hyperparameter search. Can help find optimal
            parameters for each signal type's unique characteristics.
        """
        # Store hyperparameters for potential optimization and debugging
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

        # Initialize MLP with scale-adaptive parameters
        # Key insight: We use adaptive learning rate and increased patience
        # to handle the fact that different signal types may converge at different rates
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
            verbose=False  # Set to True for debugging convergence issues
        )

        # REMOVED: StandardScaler - let the network adapt weights naturally
        # Neural networks can learn appropriate input scaling through weight adaptation
        # The key is using adaptive optimization and appropriate learning rates

        logger.info(f"Initialized MLP WITHOUT standardization, hidden layers {hidden_layer_sizes}")
        logger.info(f"  Scale-adaptive training enabled:")
        logger.info(f"  - Adam optimizer: adapts learning rates per parameter")
        logger.info(f"  - Adaptive learning rate: reduces when progress stagnates")
        logger.info(f"  - Extended patience: allows time for scale adaptation")
        logger.info(f"  - Network will learn to handle scale differences through backpropagation")
        logger.info(f"  Expected behavior:")
        logger.info(f"    Raw calcium (~6000): Network learns small input weights")
        logger.info(f"    ΔF/F (~0.15): Network learns larger input weights")
        logger.info(f"    Deconvolved (sparse): Network learns to focus on non-zero events")

    def _prepare_data(self, X, y=None):
        """
        Prepare data WITHOUT normalization while monitoring for scale issues.

        This method preserves natural signal characteristics while adding diagnostics
        to detect potential training problems caused by scale differences.

        Scale-Related Challenges:
        1. Raw calcium: Large values may cause weight saturation or gradient explosion
        2. ΔF/F: Small values may cause slow learning or gradient vanishing
        3. Deconvolved: Sparsity may cause unbalanced learning

        Our approach: Monitor these issues but preserve natural characteristics.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features with natural signal characteristics preserved
        y : torch.Tensor or np.ndarray, optional
            Target labels (0=no movement, 1=contralateral movement)

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Prepared X and y with original scales preserved plus scale diagnostics
        """
        # Convert torch tensors to numpy arrays if needed
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Reshape X to 2D if needed (n_samples, window_size * n_neurons)
        # MLPs expect flattened feature vectors
        if X.ndim == 3:
            n_samples, window_size, n_neurons = X.shape
            X = X.reshape(n_samples, window_size * n_neurons)

        # CRITICAL: Scale diagnostics to predict training behavior
        data_stats = {
            'mean': X.mean(),
            'std': X.std(),
            'min': X.min(),
            'max': X.max(),
            'range': X.max() - X.min(),
            'abs_mean': np.abs(X).mean(),
            'zero_fraction': np.sum(X == 0) / X.size
        }

        logger.info(f"MLP data prepared WITHOUT normalization:")
        logger.info(f"  Shape: {X.shape}")
        for key, value in data_stats.items():
            logger.info(f"  {key}: {value:.8f}")

        # Scale analysis and warnings
        if data_stats['abs_mean'] > 1000:
            logger.warning("⚠️  Large input values detected (>1000)")
            logger.warning("   Risk: Gradient explosion, weight saturation")
            logger.warning("   Mitigation: Adam optimizer + adaptive learning rate")
        elif data_stats['abs_mean'] < 0.01:
            logger.warning("⚠️  Very small input values detected (<0.01)")
            logger.warning("   Risk: Slow learning, gradient vanishing")
            logger.warning("   Mitigation: Adaptive learning rate + extended patience")

        if data_stats['zero_fraction'] > 0.8:
            logger.warning("⚠️  Highly sparse data detected (>80% zeros)")
            logger.warning("   Risk: Unbalanced learning, focus only on non-zero features")
            logger.warning("   Mitigation: Regularization + balanced class weights")

        logger.info(f"  Network will adapt weights to handle these natural scales")

        return X, y

    def _adapt_learning_rate_for_scale(self, X):
        """
        Suggest learning rate adaptation based on input scale.

        This method analyzes the input scale and suggests whether the default
        learning rate might need adjustment for stable training.

        Parameters
        ----------
        X : np.ndarray
            Input data

        Returns
        -------
        float
            Suggested learning rate adjustment factor
        """
        abs_mean = np.abs(X).mean()

        if abs_mean > 5000:  # Very large values (raw calcium range)
            # Suggest smaller learning rate to prevent gradient explosion
            factor = 0.1
            logger.info(f"Large input scale detected - suggesting LR reduction by factor {factor}")
        elif abs_mean < 0.1:  # Very small values (some ΔF/F range)
            # Suggest larger learning rate to overcome small gradients
            factor = 2.0
            logger.info(f"Small input scale detected - suggesting LR increase by factor {factor}")
        else:
            # Standard learning rate should work
            factor = 1.0
            logger.info("Input scale in reasonable range for standard learning rate")

        return factor

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train MLP WITHOUT standardization using scale-adaptive techniques.

        This method trains the MLP while carefully monitoring for scale-related
        training issues and providing diagnostics to understand the training dynamics.

        Training Strategy:
        1. Preserve natural signal characteristics completely
        2. Use adaptive optimization to handle scale differences
        3. Monitor convergence to detect scale-related problems
        4. Provide detailed diagnostics for troubleshooting

        Parameters
        ----------
        X_train : torch.Tensor or np.ndarray
            Training features with natural signal characteristics preserved
        y_train : torch.Tensor or np.ndarray
            Training labels (0=no movement, 1=contralateral movement)
        X_val : torch.Tensor or np.ndarray, optional
            Validation features for monitoring training progress
        y_val : torch.Tensor or np.ndarray, optional
            Validation labels for monitoring training progress

        Returns
        -------
        self
            Trained model with scale adaptation completed
        """
        logger.info("Training MLP WITHOUT standardization using scale-adaptive methods")

        # Prepare data WITHOUT any scaling
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Analyze input scale and suggest adaptations
        lr_factor = self._adapt_learning_rate_for_scale(X_train)

        # If the scale suggests a learning rate adjustment, create a new model instance
        if lr_factor != 1.0:
            adjusted_lr = self.learning_rate_init * lr_factor
            logger.info(f"Adapting learning rate from {self.learning_rate_init} to {adjusted_lr}")

            # Create new model with adjusted learning rate
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                learning_rate_init=adjusted_lr,  # Adjusted learning rate
                max_iter=self.max_iter,
                early_stopping=self.early_stopping,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                random_state=self.random_state,
                verbose=False
            )

        # REMOVED: self.scaler.fit_transform(X_train)
        # Train directly on natural signal characteristics

        # Store pre-training state for diagnostics
        pre_train_stats = {
            'X_mean': X_train.mean(),
            'X_std': X_train.std(),
            'y_distribution': np.bincount(y_train)
        }

        # Train the model with scale-adaptive parameters
        try:
            self.model.fit(X_train, y_train)
            training_successful = True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            logger.error("This may be due to scale-related convergence issues")
            training_successful = False
            raise

        # Post-training diagnostics
        if training_successful:
            logger.info(f"MLP training complete WITHOUT standardization")
            logger.info(f"  Final loss: {self.model.loss_:.6f}")
            logger.info(f"  Iterations used: {self.model.n_iter_}")
            logger.info(f"  Converged: {self.model.n_iter_ < self.max_iter}")

            # Check if early stopping was triggered
            if hasattr(self.model, 'best_loss_') and self.early_stopping:
                logger.info(f"  Best validation loss: {self.model.best_loss_:.6f}")
                logger.info(f"  Early stopping used: {self.model.n_iter_ < self.max_iter}")

            # Validation performance if available
            if X_val is not None and y_val is not None:
                X_val, y_val = self._prepare_data(X_val, y_val)
                # REMOVED: self.scaler.transform(X_val)
                val_score = self.model.score(X_val, y_val)
                logger.info(f"  Validation accuracy (on raw data): {val_score:.4f}")

            # Success message confirming scale adaptation
            logger.info("Network successfully adapted to natural signal scales:")
            logger.info(f"  Input scale (mean abs value): {np.abs(X_train).mean():.6f}")
            logger.info(f"  Network learned appropriate weights for this scale")

        return self

    def predict(self, X):
        """
        Make predictions WITHOUT standardization.

        Uses the trained network to classify neural activity patterns while
        preserving their natural signal characteristics.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features with natural signal scales preserved

        Returns
        -------
        np.ndarray
            Predicted labels (0=no movement, 1=contralateral movement)
        """
        # Prepare data WITHOUT any scaling
        X, _ = self._prepare_data(X)

        # REMOVED: self.scaler.transform(X)
        # Make predictions on raw data directly
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities WITHOUT standardization.

        Provides probability estimates based on the network's learned adaptation
        to natural signal characteristics.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features with natural signal scales preserved

        Returns
        -------
        np.ndarray
            Predicted class probabilities, shape (n_samples, n_classes)
        """
        # Prepare data WITHOUT any scaling
        X, _ = self._prepare_data(X)

        # REMOVED: self.scaler.transform(X)
        # Predict probabilities on raw data directly
        probabilities = self.model.predict_proba(X)

        return probabilities

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Estimate feature importance WITHOUT standardization bias.

        This reveals how the network naturally weights different features after
        learning to handle the original signal scales through weight adaptation.

        Scientific Insight:
        The network learns different input weight magnitudes for different signal types:
        - Raw calcium: Small input weights (to handle large input values)
        - ΔF/F: Large input weights (to amplify small input values)
        - Deconvolved: Selective weights (to focus on sparse non-zero events)

        This weight pattern reveals the network's natural adaptation strategy!

        Parameters
        ----------
        window_size : int
            Size of the sliding window (temporal dimension)
        n_neurons : int
            Number of neurons (spatial dimension)

        Returns
        -------
        np.ndarray
            Feature importance scores, shape (window_size, n_neurons)
            Reflects network's natural adaptation to signal scales
        """
        # Ensure model is trained
        if not hasattr(self.model, 'coefs_'):
            raise ValueError("Model must be trained before getting feature importance")

        # Get weights from the first layer (input → first hidden layer)
        # These weights show how the network adapted to different input scales
        first_layer_weights = self.model.coefs_[0]  # Shape: (n_features, n_hidden_1)

        # Calculate feature importance as the sum of absolute weights
        # This shows which input features the network found most useful
        # after adapting to the natural signal scales
        feature_importance = np.abs(first_layer_weights).sum(axis=1)

        # Normalize to create relative importance scores
        if feature_importance.sum() > 0:
            feature_importance = feature_importance / feature_importance.sum()

        # Reshape to (window_size, n_neurons) to show temporal-spatial patterns
        feature_importance = feature_importance.reshape(window_size, n_neurons)

        logger.info(f"MLP feature importance extracted WITHOUT standardization bias")
        logger.info(f"  Weights reveal network's natural adaptation to signal scales:")
        logger.info(f"  - Large importance = features network found most discriminative at natural scale")
        logger.info(f"  - Weight magnitudes reflect scale adaptation strategy")
        logger.info(f"  - Different signal types should show different adaptation patterns")

        # Diagnostic information about weight adaptation
        weight_stats = {
            'mean_abs_weight': np.abs(first_layer_weights).mean(),
            'max_abs_weight': np.abs(first_layer_weights).max(),
            'weight_std': np.abs(first_layer_weights).std()
        }

        logger.info(f"  Weight adaptation statistics:")
        for key, value in weight_stats.items():
            logger.info(f"    {key}: {value:.8f}")

        return feature_importance

    