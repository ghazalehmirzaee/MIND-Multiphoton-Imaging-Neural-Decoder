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
different calcium signal scales and characteristics through kernel computations.

Key Design Principles:
1. NO standardization - preserve natural signal characteristics
2. Optional PCA without prior standardization
3. Direct kernel computation on raw signal scales
4. Proper parameter separation between SVM and preprocessing
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
    Support Vector Machine WITHOUT standardization for raw signal testing.

    This class removes all data normalization to study how SVM kernels
    naturally respond to the different scales of calcium imaging signals.

    Scientific Rationale:
    - Raw calcium signals (~6000 fluorescence units) represent actual photon counts
    - ΔF/F signals (~0.15) represent normalized change from baseline
    - Deconvolved signals (sparse, ~0.004 mean) represent inferred spike events
    - Each scale carries different biological information that standardization would destroy
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
        Initialize SVM WITHOUT preprocessing pipeline.

        Parameters
        ----------
        C : float, optional
            Regularization parameter - controls the trade-off between achieving
            low training error and low testing error. Higher C = less regularization.
            Default 1.0 works well for most calcium imaging data.

        kernel : str, optional
            Specifies the kernel type to be used in the algorithm.
            'rbf' (Radial Basis Function) is excellent for non-linear calcium patterns.
            'linear' can be useful for debugging and feature importance analysis.

        gamma : str, optional
            Kernel coefficient for 'rbf'. 'scale' uses 1/(n_features * X.var())
            which adapts automatically to the signal scale - crucial when we're
            not standardizing and have different signal magnitudes.

        class_weight : Optional[str], optional
            Weights associated with classes. 'balanced' automatically adjusts
            weights inversely proportional to class frequencies - essential for
            calcium imaging where movement events are typically rare (imbalanced).

        probability : bool, optional
            Whether to enable probability estimates. Required for ROC curves and
            probability-based analysis. Adds some computational overhead but
            provides richer evaluation metrics.

        random_state : int, optional
            Controls the pseudo random number generation for shuffling data for
            probability estimates. Ensures reproducible results across runs.

        optimize_hyperparams : bool, optional
            Whether to perform hyperparameter optimization. Can improve performance
            but significantly increases training time.

        use_pca : bool, optional
            Whether to apply PCA for dimensionality reduction WITHOUT standardization.
            Can help with computational efficiency for high-dimensional data.

        pca_variance : float, optional
            Amount of variance to preserve when using PCA. 0.95 retains 95% of
            the original signal information while reducing dimensionality.
        """
        # Store hyperparameters for later use
        # These parameters define how the SVM will behave during training
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state
        self.optimize_hyperparams = optimize_hyperparams
        self.use_pca = use_pca
        self.pca_variance = pca_variance

        # Initialize SVM model directly (no pipeline wrapper)
        # We avoid sklearn's Pipeline to have full control over preprocessing
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state
        )

        # Initialize optional PCA WITHOUT standardization
        # PCA is applied to raw data to study whether dimensionality reduction
        # without normalization can still capture essential neural patterns
        self.pca = PCA(n_components=pca_variance, random_state=random_state) if use_pca else None

        # REMOVED: Pipeline with StandardScaler
        # REMOVED: Mandatory preprocessing steps
        # SVM will work directly with raw signal characteristics

        logger.info(f"Initialized SVM WITHOUT standardization")
        logger.info(f"  Kernel: {kernel}, C: {C}, Gamma: {gamma}")
        logger.info(f"  PCA: {use_pca} (applied to raw data if enabled)")
        logger.info(f"  SVM will learn decision boundaries at natural signal scales")
        logger.info(f"  Raw calcium (~6000), ΔF/F (~0.15), Deconvolved (sparse) maintain distinct patterns")

    def _prepare_data(self, X, y=None):
        """
        Prepare data WITHOUT any standardization.

        This is the critical method that preserves natural signal characteristics.

        Scientific Importance:
        - Raw calcium: High baseline (~6000) reflects actual fluorescence intensity
        - ΔF/F: Low values (~0.15) reflect relative changes from baseline
        - Deconvolved: Sparse values reflect inferred spike timing

        Each signal type has distinct statistical properties that contain
        biological information. Standardization would destroy these differences.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features, shape (n_samples, window_size, n_neurons)
            Contains the raw neural activity patterns we want to preserve

        y : torch.Tensor or np.ndarray, optional
            Target labels, shape (n_samples,)
            Binary labels: 0 = no movement, 1 = contralateral movement

        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Prepared X and y with original scales preserved
        """
        # Convert torch tensors to numpy arrays if needed
        # This handles the interface between PyTorch datasets and sklearn models
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Reshape X to 2D if needed (n_samples, window_size * n_neurons)
        # SVM expects 2D input: each row is a sample, each column is a feature
        # We flatten the temporal-spatial neural activity into a feature vector
        if X.ndim == 3:
            n_samples, window_size, n_neurons = X.shape
            X = X.reshape(n_samples, window_size * n_neurons)

        # Log the raw data characteristics we're preserving
        # This documentation helps verify we're maintaining signal integrity
        logger.info(f"SVM data prepared WITHOUT standardization:")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Mean: {X.mean():.6f}")  # Should differ dramatically between signal types
        logger.info(f"  Std: {X.std():.6f}")  # Natural variability preserved
        logger.info(f"  Min: {X.min():.6f}")  # Baseline characteristics maintained
        logger.info(f"  Max: {X.max():.6f}")  # Peak activity levels preserved
        logger.info(f"  SVM kernel will compute similarities at these natural scales")

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train SVM WITHOUT standardization.

        This method trains the SVM directly on the natural signal characteristics,
        allowing us to study how different calcium signal types are naturally
        separated by the SVM's decision boundary.

        Training Process:
        1. Prepare data (convert tensors, reshape) WITHOUT normalization
        2. Optionally apply PCA (but still no standardization)
        3. Train SVM directly on the prepared data
        4. Evaluate on validation data if provided

        Parameters
        ----------
        X_train : torch.Tensor or np.ndarray
            Training features with natural signal characteristics preserved
        y_train : torch.Tensor or np.ndarray
            Training labels (0=no movement, 1=contralateral movement)
        X_val : torch.Tensor or np.ndarray, optional
            Validation features for performance monitoring
        y_val : torch.Tensor or np.ndarray, optional
            Validation labels for performance monitoring

        Returns
        -------
        self
            The trained model ready for prediction
        """
        logger.info("Training SVM WITHOUT standardization")

        # Prepare data WITHOUT any scaling
        # This preserves the natural amplitude and baseline differences between signal types
        X_train, y_train = self._prepare_data(X_train, y_train)

        # Apply PCA if requested (but WITHOUT prior standardization)
        # This tests whether dimensionality reduction on raw signals can still capture
        # the essential neural patterns needed for movement decoding
        if self.use_pca:
            logger.info("Applying PCA to raw data (no prior standardization)")
            X_train = self.pca.fit_transform(X_train)
            explained_variance = self.pca.explained_variance_ratio_.sum()
            n_components = self.pca.n_components_
            logger.info(f"PCA on raw data: {n_components} components explain {explained_variance:.2%} of variance")

        # REMOVED: StandardScaler preprocessing
        # REMOVED: Pipeline wrapper
        # Train SVM directly on raw or PCA-transformed (but not standardized) data

        # Train the model on natural signal characteristics
        # The SVM will learn to distinguish movement vs. no-movement based on
        # the natural patterns present in each signal type
        self.model.fit(X_train, y_train)

        logger.info("SVM model training complete WITHOUT standardization")
        logger.info("Decision boundary optimized for natural signal scale differences")

        # If validation data is provided, report validation score
        # This helps us monitor training progress and detect overfitting
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
            if self.use_pca:
                X_val = self.pca.transform(X_val)
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")

        return self

    def predict(self, X):
        """
        Make predictions WITHOUT standardization.

        Uses the trained SVM to classify new neural activity patterns
        while preserving their natural signal characteristics.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features with natural signal scales preserved

        Returns
        -------
        np.ndarray
            Predicted labels (0=no movement, 1=contralateral movement)
        """
        # Prepare data WITHOUT scaling
        X, _ = self._prepare_data(X)

        # Apply PCA if it was used during training (but no standardization)
        if self.use_pca:
            X = self.pca.transform(X)

        # REMOVED: pipeline.predict() which included standardization
        # Make predictions on raw or PCA-only transformed data
        predictions = self.model.predict(X)

        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities WITHOUT standardization.

        Provides probability estimates for each class, which are essential
        for ROC curve analysis and understanding model confidence.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features with natural signal scales preserved

        Returns
        -------
        np.ndarray
            Predicted class probabilities, shape (n_samples, n_classes)
            Column 0: probability of no movement
            Column 1: probability of contralateral movement
        """
        # Prepare data WITHOUT scaling
        X, _ = self._prepare_data(X)

        # Apply PCA if it was used during training (but no standardization)
        if self.use_pca:
            X = self.pca.transform(X)

        # REMOVED: pipeline.predict_proba() which included standardization
        # Predict probabilities on raw or PCA-only transformed data
        probabilities = self.model.predict_proba(X)

        return probabilities

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Get feature importance proxy for SVM WITHOUT standardization effects.

        Note: SVMs don't have direct feature importance like Random Forest,
        but we can approximate using support vector patterns or coefficient
        magnitudes for linear kernels.

        This is especially interesting when no standardization is applied because
        the feature weights reflect how the SVM naturally responds to different
        signal scales and characteristics.

        Parameters
        ----------
        window_size : int
            Size of the sliding window (temporal dimension)
        n_neurons : int
            Number of neurons (spatial dimension)

        Returns
        -------
        np.ndarray
            Approximate feature importance matrix of shape (window_size, n_neurons)
            Higher values indicate more important features for classification
        """
        logger.info("Extracting SVM feature importance WITHOUT standardization bias")

        if self.kernel == 'linear':
            # For linear kernels, use coefficient magnitudes
            # Linear SVM coefficients directly indicate feature importance
            if hasattr(self.model, 'coef_'):
                # Get linear coefficients
                coef = self.model.coef_[0]  # Shape: (n_features,)

                # Handle PCA case
                if self.use_pca:
                    # Transform PCA coefficients back to original space
                    # This shows which original features contributed most through PCA
                    original_coef = np.abs(self.pca.components_.T @ coef)
                    importance = original_coef[:window_size * n_neurons]
                else:
                    importance = np.abs(coef)

                # Normalize to create relative importance scores
                if importance.sum() > 0:
                    importance = importance / importance.sum()

                # Reshape to (window_size, n_neurons) to show temporal-spatial patterns
                n_features = min(len(importance), window_size * n_neurons)
                importance_matrix = importance[:n_features].reshape(window_size, n_neurons)

                logger.info("Linear SVM coefficients reflect natural signal scale influences")
                logger.info("Higher coefficients indicate features that naturally distinguish movement patterns")
                return importance_matrix

        # For non-linear kernels, return uniform importance
        # Non-linear kernels don't provide interpretable feature weights
        logger.warning("Non-linear SVM: returning uniform feature importance")
        logger.info("Consider using linear kernel for interpretable feature importance")
        importance_matrix = np.ones((window_size, n_neurons)) / (window_size * n_neurons)

        return importance_matrix


