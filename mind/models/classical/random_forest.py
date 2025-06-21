# """
# Optimized Random Forest model implementation for calcium imaging data.
#
# This implementation focuses on simplicity and effectiveness for decoding behavior
# from calcium imaging data, with proper preprocessing and feature importance extraction.
# """
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import logging
# from typing import Dict, Any, Optional, Tuple, List, Union
#
# logger = logging.getLogger(__name__)
#
#
# class RandomForestModel:
#     """
#     Optimized Random Forest model for neural decoding.
#
#     This class provides a wrapper around sklearn's RandomForestClassifier with
#     appropriate preprocessing and methods for feature importance extraction.
#     """
#
#     def __init__(self,
#                  n_estimators: int = 300,
#                  max_depth: Optional[int] = None,
#                  min_samples_split: int = 5,
#                  min_samples_leaf: int = 2,
#                  max_features: str = 'sqrt',
#                  class_weight: str = 'balanced_subsample',
#                  n_jobs: int = -1,
#                  random_state: int = 42,
#                  criterion: str = 'gini',
#                  bootstrap: bool = True,
#                  use_pca: bool = False,  # Changed to False by default
#                  pca_variance: float = 0.95,
#                  optimize_hyperparams: bool = False):
#         """
#         Initialize Random Forest model with preprocessing options.
#
#         Parameters
#         ----------
#         n_estimators : int, optional
#             Number of trees in the forest, by default 300
#         max_depth : Optional[int], optional
#             Maximum depth of trees, by default None (unlimited)
#         min_samples_split : int, optional
#             Minimum samples required to split a node, by default 5
#         min_samples_leaf : int, optional
#             Minimum samples required in a leaf node, by default 2
#         max_features : str, optional
#             Number of features to consider for best split, by default 'sqrt'
#         class_weight : str, optional
#             Class weights for imbalanced data, by default 'balanced_subsample'
#         n_jobs : int, optional
#             Number of jobs to run in parallel, by default -1 (all CPUs)
#         random_state : int, optional
#             Random seed for reproducibility, by default 42
#         criterion : str, optional
#             Function to measure quality of a split, by default 'gini'
#         bootstrap : bool, optional
#             Whether to use bootstrap samples, by default True
#         use_pca : bool, optional
#             Whether to use PCA for dimensionality reduction, by default False
#         pca_variance : float, optional
#             Explained variance ratio threshold for PCA, by default 0.95
#         optimize_hyperparams : bool, optional
#             Whether to optimize hyperparameters, by default False
#         """
#         # Store parameters
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.min_samples_split = min_samples_split
#         self.min_samples_leaf = min_samples_leaf
#         self.max_features = max_features
#         self.class_weight = class_weight
#         self.n_jobs = n_jobs
#         self.random_state = random_state
#         self.criterion = criterion
#         self.bootstrap = bootstrap
#         self.use_pca = use_pca
#         self.pca_variance = pca_variance
#         self.optimize_hyperparams = optimize_hyperparams
#
#         # Initialize preprocessing components
#         self.scaler = StandardScaler()
#         self.pca = PCA(n_components=pca_variance, random_state=random_state) if use_pca else None
#
#         # Initialize Random Forest
#         self.model = RandomForestClassifier(
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_samples_split=min_samples_split,
#             min_samples_leaf=min_samples_leaf,
#             max_features=max_features,
#             class_weight=class_weight,
#             n_jobs=n_jobs,
#             random_state=random_state,
#             criterion=criterion,
#             bootstrap=bootstrap,
#             oob_score=bootstrap
#         )
#
#         logger.info(f"Initialized Random Forest with {n_estimators} trees and PCA={use_pca}")
#
#     def _prepare_data(self, X, y=None):
#         """
#         Prepare data for model training or inference.
#
#         This method handles conversion from different formats and reshaping.
#
#         Parameters
#         ----------
#         X : torch.Tensor or np.ndarray
#             Input features
#         y : torch.Tensor or np.ndarray, optional
#             Target labels, by default None
#
#         Returns
#         -------
#         Tuple[np.ndarray, np.ndarray or None]
#             Prepared data
#         """
#         # Convert torch tensors to numpy if needed
#         if hasattr(X, 'numpy'):
#             X = X.numpy()
#         if y is not None and hasattr(y, 'numpy'):
#             y = y.numpy()
#
#         # Reshape if needed (without adding potentially noisy features)
#         if X.ndim == 3:
#             n_samples, window_size, n_neurons = X.shape
#             X = X.reshape(n_samples, window_size * n_neurons)
#
#         return X, y
#
#     def fit(self, X_train, y_train, X_val=None, y_val=None):
#         """
#         Train the Random Forest model.
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
#             Trained model
#         """
#         logger.info("Training Random Forest")
#
#         # Prepare data
#         X_train, y_train = self._prepare_data(X_train, y_train)
#
#         # Apply preprocessing - standardization
#         X_train_scaled = self.scaler.fit_transform(X_train)
#
#         # Apply PCA if requested
#         if self.use_pca:
#             n_components = min(self.pca.n_components, X_train_scaled.shape[1])
#             self.pca.n_components = n_components
#             X_train_processed = self.pca.fit_transform(X_train_scaled)
#             logger.info(f"PCA reduced dimensions from {X_train_scaled.shape[1]} to {X_train_processed.shape[1]} "
#                         f"({self.pca.explained_variance_ratio_.sum():.2%} explained variance)")
#         else:
#             X_train_processed = X_train_scaled
#
#         # Train model
#         self.model.fit(X_train_processed, y_train)
#
#         # Log OOB score if available
#         if hasattr(self.model, 'oob_score_'):
#             logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")
#
#         # Validate if data provided
#         if X_val is not None and y_val is not None:
#             X_val, y_val = self._prepare_data(X_val, y_val)
#             X_val_scaled = self.scaler.transform(X_val)
#
#             if self.use_pca:
#                 X_val_processed = self.pca.transform(X_val_scaled)
#             else:
#                 X_val_processed = X_val_scaled
#
#             val_score = self.model.score(X_val_processed, y_val)
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
#         # Apply preprocessing
#         X_scaled = self.scaler.transform(X)
#
#         if self.use_pca:
#             X_processed = self.pca.transform(X_scaled)
#         else:
#             X_processed = X_scaled
#
#         # Make predictions
#         return self.model.predict(X_processed)
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
#         # Apply preprocessing
#         X_scaled = self.scaler.transform(X)
#
#         if self.use_pca:
#             X_processed = self.pca.transform(X_scaled)
#         else:
#             X_processed = X_scaled
#
#         # Predict probabilities
#         return self.model.predict_proba(X_processed)
#
#     def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
#         """
#         Get feature importance matrix.
#
#         This method extracts feature importance from the trained model and reshapes
#         it to a matrix of shape (window_size, n_neurons).
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
#             Feature importance matrix of shape (window_size, n_neurons)
#         """
#         if not hasattr(self.model, 'feature_importances_'):
#             raise ValueError("Model must be trained before getting feature importance")
#
#         # Get feature importances
#         importances = self.model.feature_importances_
#
#         if self.use_pca:
#             # When using PCA, we can't directly map back to original features
#             # Create an approximate mapping using PCA components
#             try:
#                 # Get PCA components
#                 components = self.pca.components_  # shape: (n_components, n_features)
#
#                 # Weight components by explained variance ratio
#                 weighted_components = components.T * self.pca.explained_variance_ratio_
#
#                 # Sum across components to get importance for original features
#                 original_importances = np.abs(weighted_components).sum(axis=1)
#
#                 # Normalize
#                 original_importances = original_importances / original_importances.sum()
#
#                 # Reshape to (window_size, n_neurons)
#                 importance_matrix = original_importances[:window_size * n_neurons].reshape(window_size, n_neurons)
#
#                 return importance_matrix
#             except:
#                 # Fallback: use equal importance
#                 logger.warning("Could not map PCA feature importance back to original space")
#                 importance_matrix = np.ones((window_size, n_neurons)) / (window_size * n_neurons)
#                 return importance_matrix
#         else:
#             # Direct mapping for non-PCA case
#             # Take only the first window_size * n_neurons features
#             n_features = min(len(importances), window_size * n_neurons)
#             importance_matrix = importances[:n_features].reshape(window_size, n_neurons)
#
#             return importance_matrix
#


"""
Random Forest model WITHOUT standardization to preserve signal characteristics.

This implementation removes all normalization to study how Random Forest handles
different calcium signal scales naturally through tree-based splitting criteria.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    Random Forest model WITHOUT standardization for raw signal testing.

    This class removes all data normalization to study how tree-based algorithms
    naturally handle the different scales and characteristics of calcium signals.
    """

    def __init__(self,
                 n_estimators: int = 300,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 class_weight: str = 'balanced_subsample',
                 n_jobs: int = -1,
                 random_state: int = 42,
                 criterion: str = 'gini',
                 bootstrap: bool = True,
                 optimize_hyperparams: bool = False):
        """
        Initialize Random Forest WITHOUT any preprocessing.

        Parameters
        ----------
        n_estimators : int, optional
            Number of trees in the forest, by default 300
        max_depth : Optional[int], optional
            Maximum depth of trees, by default None (unlimited)
        min_samples_split : int, optional
            Minimum samples required to split a node, by default 5
        min_samples_leaf : int, optional
            Minimum samples required in a leaf node, by default 2
        max_features : str, optional
            Number of features to consider for best split, by default 'sqrt'
        class_weight : str, optional
            Class weights for imbalanced data, by default 'balanced_subsample'
        n_jobs : int, optional
            Number of jobs to run in parallel, by default -1 (all CPUs)
        random_state : int, optional
            Random seed for reproducibility, by default 42
        criterion : str, optional
            Function to measure quality of a split, by default 'gini'
        bootstrap : bool, optional
            Whether to use bootstrap samples, by default True
        optimize_hyperparams : bool, optional
            Whether to optimize hyperparameters, by default False
        """
        # Store parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.optimize_hyperparams = optimize_hyperparams

        # REMOVED: StandardScaler - let Random Forest handle raw signal scales
        # REMOVED: PCA - preserve original feature space

        # Initialize Random Forest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            criterion=criterion,
            bootstrap=bootstrap,
            oob_score=bootstrap
        )

        logger.info(f"Initialized Random Forest WITHOUT preprocessing with {n_estimators} trees")

    def _prepare_data(self, X, y=None):
        """
        Prepare data WITHOUT any normalization or scaling.

        This method only handles tensor conversion and reshaping.
        All signal characteristics and scales preserved completely.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features
        y : torch.Tensor or np.ndarray, optional
            Target labels, by default None

        Returns
        -------
        Tuple[np.ndarray, np.ndarray or None]
            Prepared data with original scales preserved
        """
        # Convert torch tensors to numpy if needed
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Reshape if needed (without adding potentially noisy features)
        if X.ndim == 3:
            n_samples, window_size, n_neurons = X.shape
            X = X.reshape(n_samples, window_size * n_neurons)

        # Log the raw data characteristics we're preserving
        logger.info(f"Random Forest data prepared WITHOUT preprocessing:")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Mean: {X.mean():.6f}")
        logger.info(f"  Std: {X.std():.6f}")
        logger.info(f"  Min: {X.min():.6f}")
        logger.info(f"  Max: {X.max():.6f}")
        logger.info(f"  Tree splits will work directly with these natural scales")

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train Random Forest WITHOUT any preprocessing.

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
            Trained model
        """
        logger.info("Training Random Forest WITHOUT preprocessing")

        # Prepare data WITHOUT any scaling or normalization
        X_train, y_train = self._prepare_data(X_train, y_train)

        # REMOVED: All preprocessing steps
        # No StandardScaler.fit_transform()
        # No PCA transformation
        # Direct training on raw signal characteristics

        # Train model on raw data
        self.model.fit(X_train, y_train)

        # Log OOB score if available
        if hasattr(self.model, 'oob_score_'):
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")

        # Validate if data provided
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
            # No transformation applied to validation data either
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")

        logger.info("Random Forest training complete WITHOUT preprocessing")
        logger.info("Tree splits optimized for natural signal characteristics")

        return self

    def predict(self, X):
        """
        Make predictions WITHOUT preprocessing.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predicted labels
        """
        # Prepare data WITHOUT any scaling
        X, _ = self._prepare_data(X)

        # REMOVED: scaler.transform() - use raw data directly
        # Make predictions on unprocessed data
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities WITHOUT preprocessing.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predicted class probabilities
        """
        # Prepare data WITHOUT any scaling
        X, _ = self._prepare_data(X)

        # REMOVED: scaler.transform() - use raw data directly
        # Predict probabilities on unprocessed data
        return self.model.predict_proba(X)

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Get feature importance WITHOUT preprocessing effects.

        This shows how the Random Forest naturally weights different features
        when working with signals at their original scales.

        Parameters
        ----------
        window_size : int
            Size of the sliding window
        n_neurons : int
            Number of neurons

        Returns
        -------
        np.ndarray
            Feature importance matrix of shape (window_size, n_neurons)
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be trained before getting feature importance")

        # Get feature importances (these reflect natural signal characteristics)
        importances = self.model.feature_importances_

        # Direct mapping since no PCA was applied
        n_features = min(len(importances), window_size * n_neurons)
        importance_matrix = importances[:n_features].reshape(window_size, n_neurons)

        logger.info(f"Random Forest feature importance extracted WITHOUT preprocessing")
        logger.info(f"  Importances reflect natural signal scale differences")
        logger.info(f"  Raw calcium (~6000), ΔF/F (~0.15), Deconvolved (sparse) maintain distinct patterns")

        return importance_matrix

