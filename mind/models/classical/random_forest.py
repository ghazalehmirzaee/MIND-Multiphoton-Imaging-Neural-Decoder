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
Random Forest model WITHOUT any preprocessing to preserve signal characteristics.

FIXED: Completely removes all preprocessing and adds comprehensive data tracking
to ensure different signals produce different results.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
import hashlib

logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    Random Forest model WITHOUT any preprocessing for raw signal testing.

    FIXED: This class now includes comprehensive data tracking to ensure
    that different signal types maintain their unique characteristics
    throughout the entire training process.
    """

    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: Optional[int] = 15,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 class_weight: str = 'balanced_subsample',
                 n_jobs: int = -1,
                 random_state: int = 42,
                 criterion: str = 'gini',
                 bootstrap: bool = True,
                 **kwargs):  # FIXED: Accept extra parameters and ignore them
        """
        Initialize Random Forest WITHOUT any preprocessing.

        FIXED: Now includes comprehensive parameter handling and data integrity checking.
        """
        # Store all parameters
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

        # FIXED: Log any extra parameters that were ignored
        if kwargs:
            logger.info(f"Random Forest ignoring extra parameters: {list(kwargs.keys())}")

        # CRITICAL: NO preprocessing components initialized
        # This ensures absolutely no data modification occurs

        # Initialize Random Forest with parameters optimized for calcium imaging
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

        # FIXED: Track data fingerprints to detect contamination
        self.data_fingerprints = {}

        logger.info(f"Initialized Random Forest WITHOUT preprocessing with {n_estimators} trees")
        logger.info(f"  All preprocessing components DISABLED")
        logger.info(f"  Data fingerprint tracking ENABLED")

    def _create_data_fingerprint(self, X, label=""):
        """
        Create a unique fingerprint for the data to track contamination.

        FIXED: This method helps us verify that different signal types
        maintain their unique characteristics.
        """
        fingerprint = {
            'label': label,
            'shape': X.shape,
            'mean': float(X.mean()),
            'std': float(X.std()),
            'min': float(X.min()),
            'max': float(X.max()),
            'sum': float(X.sum()),
            'hash': hashlib.md5(X.tobytes()).hexdigest()[:16],
            'first_10_values': X.flatten()[:10].tolist(),
            'memory_id': id(X)
        }
        return fingerprint

    def _prepare_data(self, X, y=None):
        """
        Prepare data WITHOUT any normalization or scaling.

        FIXED: Now includes comprehensive fingerprinting to track data integrity.
        """
        # Convert torch tensors to numpy if needed
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Create fingerprint BEFORE any processing
        original_fingerprint = self._create_data_fingerprint(X, "input")

        # Reshape for Random Forest: (n_samples, n_features)
        original_shape = X.shape
        if X.ndim == 3:
            n_samples, window_size, n_neurons = X.shape
            X = X.reshape(n_samples, window_size * n_neurons)

        # Create fingerprint AFTER reshaping
        reshaped_fingerprint = self._create_data_fingerprint(X, "reshaped")

        # FIXED: Comprehensive verification that NO preprocessing occurred
        logger.info(f"Random Forest data preparation WITHOUT preprocessing:")
        logger.info(f"  Shape transformation: {original_shape} -> {X.shape}")
        logger.info(f"  Mean: {reshaped_fingerprint['mean']:.8f}")
        logger.info(f"  Std: {reshaped_fingerprint['std']:.8f}")
        logger.info(f"  Range: [{reshaped_fingerprint['min']:.6f}, {reshaped_fingerprint['max']:.6f}]")
        logger.info(f"  Data hash: {reshaped_fingerprint['hash']}")

        # CRITICAL: Check for signs of normalization
        if abs(reshaped_fingerprint['mean']) < 0.1 and abs(reshaped_fingerprint['std'] - 1.0) < 0.1:
            logger.error("🚨 DATA APPEARS NORMALIZED! This indicates hidden preprocessing!")
            logger.error(f"   Mean: {reshaped_fingerprint['mean']:.8f}, Std: {reshaped_fingerprint['std']:.8f}")
            raise ValueError("Data has been unexpectedly normalized")

        # Store fingerprint for contamination detection
        timestamp = len(self.data_fingerprints)
        self.data_fingerprints[f"prepare_{timestamp}"] = reshaped_fingerprint

        logger.info(f"  ✓ Data maintains original scale characteristics")
        logger.info(f"  ✓ No hidden preprocessing detected")

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train Random Forest WITHOUT any preprocessing.

        FIXED: Enhanced monitoring to ensure different signals produce different trees.
        """
        logger.info("Training Random Forest WITHOUT preprocessing")

        # Prepare data WITHOUT any scaling or normalization
        X_train, y_train = self._prepare_data(X_train, y_train)

        # FIXED: Store training data fingerprint for comparison across runs
        train_fingerprint = self._create_data_fingerprint(X_train, "training")

        # Check if we've seen this exact data before (contamination detection)
        for stored_name, stored_fp in self.data_fingerprints.items():
            if stored_fp['hash'] == train_fingerprint['hash'] and stored_name.startswith("training_"):
                logger.warning(f"🚨 IDENTICAL TRAINING DATA detected!")
                logger.warning(f"   Current data matches previously seen: {stored_name}")
                logger.warning(f"   This may explain identical results across signal types")

        # Store this training fingerprint
        self.data_fingerprints[f"training_{train_fingerprint['hash'][:8]}"] = train_fingerprint

        logger.info(f"Training Random Forest on data with characteristics:")
        logger.info(f"  Mean: {train_fingerprint['mean']:.8f}")
        logger.info(f"  Std: {train_fingerprint['std']:.8f}")
        logger.info(f"  Unique hash: {train_fingerprint['hash']}")

        # REMOVED: All preprocessing steps
        # Train Random Forest directly on raw signal characteristics
        try:
            self.model.fit(X_train, y_train)
            logger.info("Random Forest training completed successfully")
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            raise

        # Log out-of-bag score if available
        if hasattr(self.model, 'oob_score_'):
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")

        # FIXED: Analyze what the trees actually learned
        self._analyze_tree_splits(X_train)

        # Validate if data provided
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")

        return self

    def _analyze_tree_splits(self, X_train):
        """
        FIXED: Analyze the decision tree splits to verify they're using original scales.
        """
        if hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 0:
            # Analyze first few trees
            split_values = []
            for i, tree_estimator in enumerate(self.model.estimators_[:3]):
                tree = tree_estimator.tree_
                if hasattr(tree, 'threshold'):
                    thresholds = tree.threshold[tree.threshold != -2]  # -2 indicates leaf nodes
                    if len(thresholds) > 0:
                        split_values.extend(thresholds)
                        logger.info(f"Tree {i} split range: [{thresholds.min():.6f}, {thresholds.max():.6f}]")

            if split_values:
                all_splits = np.array(split_values)
                logger.info(f"Overall split analysis:")
                logger.info(f"  Split value range: [{all_splits.min():.6f}, {all_splits.max():.6f}]")
                logger.info(f"  Split value mean: {all_splits.mean():.6f}")
                logger.info(f"  These splits should reflect natural signal scale differences")

                # FIXED: Verify splits are appropriate for the data scale
                data_range = X_train.max() - X_train.min()
                split_range = all_splits.max() - all_splits.min()
                if split_range > data_range * 0.1:  # Splits span reasonable portion of data range
                    logger.info(f"  ✓ Splits span appropriate range relative to data")
                else:
                    logger.warning(f"  ⚠️ Splits may not be utilizing full data range effectively")

    def predict(self, X):
        """Make predictions WITHOUT preprocessing."""
        X, _ = self._prepare_data(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities WITHOUT preprocessing."""
        X, _ = self._prepare_data(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Get feature importance WITHOUT preprocessing effects.

        FIXED: Enhanced analysis of importance patterns.
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be trained before extracting feature importance")

        importances = self.model.feature_importances_
        n_features = min(len(importances), window_size * n_neurons)
        importance_matrix = importances[:n_features].reshape(window_size, n_neurons)

        logger.info(f"Random Forest feature importance extracted WITHOUT preprocessing")
        logger.info(f"  Importance statistics:")
        logger.info(f"    Mean: {importance_matrix.mean():.8f}")
        logger.info(f"    Max: {importance_matrix.max():.8f}")
        logger.info(f"    Non-zero features: {np.sum(importance_matrix > 0)}/{importance_matrix.size}")

        # FIXED: Verify importance pattern makes sense
        if importance_matrix.max() < 0.01:
            logger.warning("⚠️ Very low maximum importance detected")
            logger.warning("   This may indicate the model didn't find strong patterns")

        return importance_matrix

    def get_data_fingerprints(self):
        """
        FIXED: Return all data fingerprints for contamination analysis.
        """
        return self.data_fingerprints

