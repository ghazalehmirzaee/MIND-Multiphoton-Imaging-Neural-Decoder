# """
# Support Vector Machine model implementation for calcium imaging data.
# """
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.preprocessing import StandardScaler
# import logging
# from typing import Dict, Any, Optional, Tuple, List, Union
#
# logger = logging.getLogger(__name__)
#
#
# class SVMModel:
#     """
#     Support Vector Machine model for decoding behavior from calcium imaging signals.
#
#     This class implements an SVM classifier with PCA preprocessing for dimensionality
#     reduction, which is essential for handling high-dimensional calcium imaging data.
#     """
#
#     def __init__(self,
#                  C: float = 1.0,
#                  kernel: str = 'rbf',
#                  gamma: str = 'scale',
#                  class_weight: Optional[str] = 'balanced',
#                  probability: bool = True,
#                  random_state: int = 42,
#                  n_components: Optional[float] = 0.95,
#                  optimize_hyperparams: bool = False,
#                  use_pca: bool = True):
#         """
#         Initialize an SVM model.
#
#         Parameters
#         ----------
#         C : float, optional
#             Regularization parameter, by default 1.0
#         kernel : str, optional
#             Kernel type ('rbf' or 'linear'), by default 'rbf'
#         gamma : str, optional
#             Kernel coefficient, by default 'scale'
#         class_weight : Optional[str], optional
#             Weights for imbalanced classes, by default 'balanced'
#         probability : bool, optional
#             Whether to enable probability estimates, by default True
#         random_state : int, optional
#             Random seed for reproducibility, by default 42
#         n_components : Optional[float], optional
#             Number of PCA components or explained variance ratio, by default 0.95
#         optimize_hyperparams : bool, optional
#             Whether to optimize hyperparameters, by default False
#         use_pca : bool, optional
#             Whether to use PCA for dimensionality reduction, by default True
#         """
#         # Store hyperparameters
#         self.C = C
#         self.kernel = kernel
#         self.gamma = gamma
#         self.class_weight = class_weight
#         self.probability = probability
#         self.random_state = random_state
#         self.n_components = n_components
#         self.optimize_hyperparams = optimize_hyperparams
#         self.use_pca = use_pca
#
#         # Initialize SVM model
#         self.svm = SVC(
#             C=C,
#             kernel=kernel,
#             gamma=gamma,
#             class_weight=class_weight,
#             probability=probability,
#             random_state=random_state
#         )
#
#         # Initialize pipeline with optional PCA
#         if use_pca:
#             self.model = Pipeline([
#                 ('scaler', StandardScaler()),  # Added scaler for better PCA performance
#                 ('pca', PCA(n_components=n_components, random_state=random_state)),
#                 ('svm', self.svm)
#             ])
#             logger.info(f"Initialized SVM model with PCA (n_components={n_components})")
#         else:
#             self.model = Pipeline([
#                 ('scaler', StandardScaler()),  # Always scale for SVM
#                 ('svm', self.svm)
#             ])
#             logger.info("Initialized SVM model without PCA")
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
#         logger.info("Optimizing SVM hyperparameters")
#
#         # Prepare data
#         X_train, y_train = self._prepare_data(X_train, y_train)
#
#         # Define parameter grid - simplified for calcium imaging data
#         if self.use_pca:
#             param_grid = {
#                 'pca__n_components': [0.85, 0.9, 0.95, 0.99],  # Focused range
#                 'svm__C': [0.1, 1, 10, 100],
#                 'svm__gamma': ['scale', 'auto', 0.001, 0.01],
#                 'svm__kernel': ['rbf', 'linear']  # Removed 'poly' - rarely needed
#             }
#         else:
#             param_grid = {
#                 'svm__C': [0.1, 1, 10, 100],
#                 'svm__gamma': ['scale', 'auto', 0.001, 0.01],
#                 'svm__kernel': ['rbf', 'linear']
#             }
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
#         # Update model parameters
#         if self.use_pca:
#             self.n_components = best_params.get('pca__n_components', self.n_components)
#             self.C = best_params.get('svm__C', self.C)
#             self.gamma = best_params.get('svm__gamma', self.gamma)
#             self.kernel = best_params.get('svm__kernel', self.kernel)
#
#             # Reinitialize pipeline with best parameters
#             self.svm = SVC(
#                 C=self.C,
#                 kernel=self.kernel,
#                 gamma=self.gamma,
#                 class_weight=self.class_weight,
#                 probability=self.probability,
#                 random_state=self.random_state
#             )
#
#             self.model = Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('pca', PCA(n_components=self.n_components, random_state=self.random_state)),
#                 ('svm', self.svm)
#             ])
#         else:
#             self.C = best_params.get('svm__C', self.C)
#             self.gamma = best_params.get('svm__gamma', self.gamma)
#             self.kernel = best_params.get('svm__kernel', self.kernel)
#
#             # Reinitialize SVM with best parameters
#             self.svm = SVC(
#                 C=self.C,
#                 kernel=self.kernel,
#                 gamma=self.gamma,
#                 class_weight=self.class_weight,
#                 probability=self.probability,
#                 random_state=self.random_state
#             )
#
#             self.model = Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('svm', self.svm)
#             ])
#
#         return self
#
#     def fit(self, X_train, y_train, X_val=None, y_val=None):
#         """
#         Train the SVM model.
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
#         logger.info("Training SVM model")
#
#         # Prepare data
#         X_train, y_train = self._prepare_data(X_train, y_train)
#
#         # Optimize hyperparameters if requested
#         if self.optimize_hyperparams:
#             self.optimize_hyperparameters(X_train, y_train)
#
#         # Train the model
#         self.model.fit(X_train, y_train)
#
#         logger.info("SVM model training complete")
#
#         # Log PCA explained variance if applicable
#         if self.use_pca:
#             pca = self.model.named_steps['pca']
#             explained_variance = pca.explained_variance_ratio_.sum()
#             n_components = pca.n_components_
#             logger.info(f"PCA: {n_components} components explain {explained_variance:.2%} of variance")
#
#         # If validation data is provided, report validation score
#         if X_val is not None and y_val is not None:
#             X_val, y_val = self._prepare_data(X_val, y_val)
#             val_score = self.model.score(X_val, y_val)
#             logger.info(f"Validation accuracy: {val_score:.4f}")
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
#         # Prepare data
#         X, _ = self._prepare_data(X)
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
#         # Prepare data
#         X, _ = self._prepare_data(X)
#
#         # Predict probabilities
#         probabilities = self.model.predict_proba(X)
#
#         return probabilities
#
#


"""
Support Vector Machine WITHOUT standardization for calcium imaging data.

This implementation removes all normalization to study how SVM naturally handles
different calcium signal scales through kernel computations.

The key insight: Different signal types should create different decision boundaries
when their natural scales are preserved.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class SVMModel:
    """
    Support Vector Machine WITHOUT standardization.

    This class demonstrates how SVM kernels respond to natural signal scales:
    - Raw calcium (~6000): Creates decision boundaries based on large fluorescence values
    - ΔF/F (~0.15): Creates boundaries based on relative change magnitudes
    - Deconvolved (sparse): Creates boundaries based on spike event patterns

    Each signal type should produce fundamentally different kernel responses.
    """

    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 gamma: str = 'scale',
                 class_weight: Optional[str] = 'balanced',
                 probability: bool = True,
                 random_state: int = 42,
                 optimize_hyperparams: bool = False,
                 use_pca: bool = False,
                 pca_variance: float = 0.95):
        """
        Initialize SVM WITHOUT any preprocessing pipeline.

        Parameters Explained:
        ---------------------
        C : float
            Regularization parameter. Controls the trade-off between achieving low
            training error and low testing error. For raw calcium signals (large values),
            this effectively controls how much the model can be influenced by outliers.

        kernel : str
            The kernel function. 'rbf' (Radial Basis Function) creates circular decision
            boundaries, perfect for complex calcium signal patterns. 'linear' creates
            straight-line boundaries, useful for interpretability.

        gamma : str
            Kernel coefficient for 'rbf'. 'scale' automatically adapts to input variance:
            gamma = 1 / (n_features * X.var()). This is CRUCIAL when signals have different
            scales, as it allows the kernel to adapt to each signal type's characteristics.

        class_weight : str
            'balanced' adjusts weights inversely proportional to class frequencies.
            Essential for calcium imaging where movement events are rare vs. rest periods.

        probability : bool
            Enables probability estimates. Required for ROC/PR curve analysis.
            Adds computational overhead but provides richer evaluation metrics.

        use_pca : bool
            Whether to apply PCA WITHOUT prior standardization. Tests whether
            dimensionality reduction on raw signals preserves discriminative patterns.
        """
        # Store all parameters for reproducibility and debugging
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state
        self.optimize_hyperparams = optimize_hyperparams
        self.use_pca = use_pca
        self.pca_variance = pca_variance

        # Initialize SVM model directly (no pipeline to avoid hidden preprocessing)
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state
        )

        # Initialize optional PCA WITHOUT standardization
        # This tests whether dimensionality reduction on raw data preserves patterns
        self.pca = PCA(n_components=pca_variance, random_state=random_state) if use_pca else None

        logger.info(f"Initialized SVM WITHOUT standardization")
        logger.info(f"  Parameters: C={C}, kernel={kernel}, gamma={gamma}")
        logger.info(f"  PCA: {use_pca} (applied to raw data if enabled)")
        logger.info(f"  Expected behavior:")
        logger.info(f"    Raw calcium (~6000): Kernel will adapt to large-scale patterns")
        logger.info(f"    ΔF/F (~0.15): Kernel will adapt to small-scale relative changes")
        logger.info(f"    Deconvolved (sparse): Kernel will focus on sparse spike events")

    def _prepare_data(self, X, y=None):
        """
        Prepare data WITHOUT any standardization.

        This method is critical for preserving signal characteristics.

        Why Each Signal Type Matters:
        ----------------------------
        Raw calcium: High baseline reflects actual photon detection levels
        ΔF/F: Low values reflect relative changes from baseline
        Deconvolved: Sparse values reflect inferred neural spike timing

        These differences contain biological information that standardization destroys.
        """
        # Handle tensor conversion (interface compatibility with PyTorch datasets)
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Reshape for SVM: (n_samples, n_features)
        # SVM expects flattened feature vectors representing temporal-spatial patterns
        if X.ndim == 3:
            n_samples, window_size, n_neurons = X.shape
            X = X.reshape(n_samples, window_size * n_neurons)

        # Document the preserved characteristics for scientific verification
        logger.info(f"SVM data prepared WITHOUT standardization:")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Mean: {X.mean():.6f}")  # Should differ by orders of magnitude
        logger.info(f"  Std: {X.std():.6f}")  # Natural variability preserved
        logger.info(f"  Min: {X.min():.6f}")  # Baseline characteristics maintained
        logger.info(f"  Max: {X.max():.6f}")  # Peak activity levels preserved
        logger.info(f"  Range: {X.max() - X.min():.6f}")  # Dynamic range preserved

        # Verify we haven't accidentally applied normalization
        if abs(X.mean()) < 0.1 and abs(X.std() - 1.0) < 0.1:
            logger.error("🚨 DATA APPEARS NORMALIZED! This suggests pipeline contamination!")
            logger.error("Expected: Raw calcium ~6000, ΔF/F ~0.15, Deconvolved ~0.004")
        else:
            logger.info("✓ Signal characteristics confirm NO normalization applied")

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train SVM WITHOUT standardization.

        Training Process Analysis:
        -------------------------
        1. Raw calcium: Large input values → SVM learns decision boundaries at high scales
        2. ΔF/F: Small input values → SVM learns boundaries at fractional scales
        3. Deconvolved: Sparse inputs → SVM learns to distinguish sparse event patterns

        The 'scale' gamma parameter automatically adapts to each signal type's variance,
        allowing fair comparison without artificial normalization.
        """
        logger.info("Training SVM WITHOUT standardization on natural signal characteristics")

        # Prepare data while preserving all natural characteristics
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Apply PCA if requested (but WITHOUT prior standardization)
        # This is a controlled experiment: can dimensionality reduction work on raw signals?
        if self.use_pca:
            logger.info("Applying PCA to raw data (no prior standardization)")
            original_shape = X_train.shape
            X_train = self.pca.fit_transform(X_train)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            n_components = self.pca.n_components_
            logger.info(f"PCA on raw data: {original_shape[1]} → {n_components} features")
            logger.info(f"Explained variance: {explained_variance:.2%}")

        # Train SVM directly on unprocessed (or PCA-only) data
        # The kernel will adapt to the natural signal scale through gamma='scale'
        logger.info("Training SVM on natural signal scales...")
        self.model.fit(X_train, y_train)

        # Report successful adaptation to signal characteristics
        logger.info("SVM training complete WITHOUT standardization")
        logger.info(f"Decision boundary optimized for natural signal scale differences")

        # Validate performance if validation data provided
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
            if self.use_pca:
                X_val = self.pca.transform(X_val)
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy on natural scales: {val_score:.4f}")

        return self

    def predict(self, X):
        """Make predictions preserving natural signal characteristics."""
        X, _ = self._prepare_data(X)
        if self.use_pca:
            X = self.pca.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities on natural signal scales."""
        X, _ = self._prepare_data(X)
        if self.use_pca:
            X = self.pca.transform(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Extract feature importance WITHOUT standardization bias.

        For linear kernels, this shows how SVM naturally weights different features
        when processing signals at their original scales.
        """
        logger.info("Extracting SVM feature importance WITHOUT standardization bias")

        if self.kernel == 'linear':
            # Linear SVM provides interpretable coefficients
            if hasattr(self.model, 'coef_'):
                coef = self.model.coef_[0]  # Shape: (n_features,)

                # Handle PCA case: transform coefficients back to original space
                if self.use_pca:
                    # Project PCA coefficients back to original feature space
                    original_coef = np.abs(self.pca.components_.T @ coef)
                    importance = original_coef[:window_size * n_neurons]
                else:
                    importance = np.abs(coef)

                # Normalize to create relative importance scores
                if importance.sum() > 0:
                    importance = importance / importance.sum()

                # Reshape to (window_size, n_neurons) for temporal-spatial interpretation
                n_features = min(len(importance), window_size * n_neurons)
                importance_matrix = importance[:n_features].reshape(window_size, n_neurons)

                logger.info("Linear SVM coefficients reflect natural signal scale influences")
                return importance_matrix

        # For non-linear kernels, feature importance isn't directly interpretable
        logger.warning("Non-linear SVM: returning uniform importance")
        logger.info("Consider linear kernel for interpretable feature weights")
        return np.ones((window_size, n_neurons)) / (window_size * n_neurons)

