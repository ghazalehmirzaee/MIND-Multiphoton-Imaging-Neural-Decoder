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

This implementation is designed to work directly with the natural scales of different
calcium imaging signals to reveal how tree-based algorithms handle scale differences.

Scientific Rationale:
- Random Forest uses decision trees that split on actual feature values
- Tree splits should naturally adapt to different signal scales
- Raw calcium (~6000), ΔF/F (~0.15), and deconvolved (sparse) signals should produce
  different tree structures and decision boundaries
- NO preprocessing allows us to study the natural discriminative power of each signal type
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    Random Forest model WITHOUT any preprocessing for raw signal testing.

    This class completely removes data normalization to study how tree-based algorithms
    naturally handle the different scales and characteristics of calcium signals.

    Key Design Philosophy:
    1. ZERO preprocessing - preserve every aspect of natural signal characteristics
    2. Let tree splits work directly with actual fluorescence values
    3. Study how different signal types create different decision tree structures
    4. Verify that scale differences alone can provide discriminative power
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
                 bootstrap: bool = True):
        """
        Initialize Random Forest WITHOUT any preprocessing.

        Parameters
        ----------
        n_estimators : int, optional
            Number of trees in the forest. 200 provides good balance between
            performance and computational cost for calcium imaging data.
            More trees = more stable predictions but longer training time.

        max_depth : Optional[int], optional
            Maximum depth of each tree. 15 prevents overfitting while allowing
            sufficient complexity to capture temporal-spatial neural patterns.
            None would allow unlimited depth (risk of overfitting).

        min_samples_split : int, optional
            Minimum samples required to split an internal node. 5 prevents
            the tree from making splits based on very few samples (reduces noise).
            This is crucial for calcium imaging where some patterns might be rare.

        min_samples_leaf : int, optional
            Minimum samples required in a leaf node. 2 ensures each decision
            has statistical support while maintaining sufficient granularity.

        max_features : str, optional
            Number of features to consider for the best split. 'sqrt' means
            sqrt(total_features) are randomly selected for each split.
            This adds randomness and prevents overfitting to dominant features.

        class_weight : str, optional
            'balanced_subsample' automatically balances class weights for each
            tree using bootstrap sample composition. Essential for calcium imaging
            where movement events are typically much rarer than no-movement periods.

        n_jobs : int, optional
            Number of parallel jobs. -1 uses all available CPU cores for faster
            training on the typically high-dimensional calcium imaging data.

        random_state : int, optional
            Controls randomness for reproducible results across different runs.
            Essential for scientific experiments requiring reproducibility.

        criterion : str, optional
            Function to measure split quality. 'gini' measures impurity and works
            well for binary classification of movement vs. no-movement.

        bootstrap : bool, optional
            Whether to use bootstrap samples for training each tree. True enables
            out-of-bag scoring and adds robustness through sample diversity.
        """
        # Store all parameters for potential debugging and hyperparameter analysis
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

        # CRITICAL: NO preprocessing components initialized
        # REMOVED: StandardScaler - would destroy natural signal characteristics
        # REMOVED: PCA - would create linear combinations that obscure signal differences
        # REMOVED: Any normalization - would make different signals artificially similar

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
            oob_score=bootstrap  # Enable out-of-bag scoring for performance monitoring
        )

        logger.info(f"Initialized Random Forest WITHOUT preprocessing with {n_estimators} trees")
        logger.info(f"  Tree splits will work directly on natural signal values:")
        logger.info(f"  - Raw calcium: ~6000 fluorescence units (actual photon counts)")
        logger.info(f"  - ΔF/F: ~0.15 normalized units (relative change from baseline)")
        logger.info(f"  - Deconvolved: sparse values (inferred spike events)")
        logger.info(f"  Each signal type should create distinct tree structures!")

    def _prepare_data(self, X, y=None):
        """
        Prepare data WITHOUT any normalization or scaling.

        This is the most critical method in the entire pipeline. It must preserve
        every aspect of the original signal characteristics to enable proper
        comparison between signal types.

        The Scientific Importance:
        Random Forest decision trees make splits based on actual feature values:
        - Raw calcium: Trees might split on "if fluorescence > 6500 then movement"
        - ΔF/F: Trees might split on "if change > 0.3 then movement"
        - Deconvolved: Trees might split on "if spike_probability > 0.1 then movement"

        These are fundamentally different decision rules that should produce
        different performance characteristics!

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input features preserving natural signal characteristics
        y : torch.Tensor or np.ndarray, optional
            Target labels (0=no movement, 1=contralateral movement)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray or None]
            Prepared data with original scales completely preserved
        """
        # Convert torch tensors to numpy if needed (interface compatibility)
        if hasattr(X, 'numpy'):
            X = X.numpy()
        if y is not None and hasattr(y, 'numpy'):
            y = y.numpy()

        # Reshape for Random Forest: (n_samples, n_features)
        # Random Forest expects 2D input where each row is a sample
        # We flatten temporal-spatial neural activity into feature vectors
        if X.ndim == 3:
            n_samples, window_size, n_neurons = X.shape
            X = X.reshape(n_samples, window_size * n_neurons)

        # CRITICAL VERIFICATION: Log data characteristics to verify signal integrity
        # These statistics should be dramatically different for each signal type
        logger.info(f"Random Forest data prepared WITHOUT any preprocessing:")
        logger.info(f"  Shape: {X.shape}")
        logger.info(f"  Mean: {X.mean():.8f}")  # Should differ by orders of magnitude
        logger.info(f"  Std: {X.std():.8f}")  # Natural variability preserved
        logger.info(f"  Min: {X.min():.8f}")  # Baseline characteristics maintained
        logger.info(f"  Max: {X.max():.8f}")  # Peak activity levels preserved
        logger.info(f"  Range: {X.max() - X.min():.8f}")  # Dynamic range preserved

        # Additional verification: check for any signs of normalization artifacts
        # Normalized data would have mean ≈ 0, std ≈ 1
        if abs(X.mean()) < 0.1 and abs(X.std() - 1.0) < 0.1:
            logger.error("⚠️  DATA APPEARS TO BE NORMALIZED! This suggests preprocessing error!")
            logger.error("   Raw calcium should have mean ~6000, ΔF/F ~0.15, deconvolved ~0.004")
        else:
            logger.info("✓ Data characteristics confirm NO normalization applied")

        logger.info(f"  Tree splits will use these exact values for decision boundaries")

        return X, y

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train Random Forest WITHOUT any preprocessing.

        This method trains the Random Forest directly on natural signal characteristics,
        allowing tree splits to discover the inherent discriminative patterns in each
        signal type without any artificial modifications.

        Training Process Analysis:
        1. Raw calcium: High-amplitude signals (thousands of fluorescence units)
           Trees will split on large thresholds reflecting actual photon count differences
        2. ΔF/F: Low-amplitude signals (fractions)
           Trees will split on small thresholds reflecting relative fluorescence changes
        3. Deconvolved: Sparse signals (mostly zeros with occasional spikes)
           Trees will split primarily on presence/absence of inferred spike events

        These fundamentally different splitting strategies should produce different
        tree structures and therefore different performance characteristics.

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
            Trained model ready for prediction and analysis
        """
        logger.info("Training Random Forest WITHOUT any preprocessing")

        # Prepare data WITHOUT any scaling or normalization
        # This is where we preserve the natural signal characteristics
        X_train, y_train = self._prepare_data(X_train, y_train)

        # VERIFICATION: Double-check that different signal types have different characteristics
        # This verification helps catch data pipeline errors early
        signal_fingerprint = {
            'mean': X_train.mean(),
            'std': X_train.std(),
            'min': X_train.min(),
            'max': X_train.max(),
            'n_zeros': np.sum(X_train == 0),  # Important for deconvolved signals
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1]
        }

        logger.info("Signal fingerprint for Random Forest training:")
        for key, value in signal_fingerprint.items():
            logger.info(f"  {key}: {value}")

        # REMOVED: All preprocessing steps that could homogenize the signals:
        # - No StandardScaler.fit_transform()
        # - No PCA transformation
        # - No normalization of any kind
        # - No feature scaling

        # Train Random Forest directly on raw signal characteristics
        # Each tree will learn to split on the natural feature values
        self.model.fit(X_train, y_train)

        # Log out-of-bag score if available (built-in cross-validation measure)
        if hasattr(self.model, 'oob_score_'):
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")
            logger.info("OOB score reflects performance on natural signal characteristics")

        # Validate if data provided
        if X_val is not None and y_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
            # No transformation applied to validation data either
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")

        logger.info("Random Forest training complete WITHOUT preprocessing")
        logger.info("Tree structure optimized for natural signal characteristics")

        # Additional diagnostic: Check if trees actually used different thresholds
        # This helps verify that different signal types create different tree structures
        if hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 0:
            first_tree = self.model.estimators_[0].tree_
            if hasattr(first_tree, 'threshold'):
                thresholds = first_tree.threshold[first_tree.threshold != -2]  # -2 indicates leaf nodes
                if len(thresholds) > 0:
                    logger.info(f"Sample tree thresholds - Min: {thresholds.min():.6f}, Max: {thresholds.max():.6f}")
                    logger.info("These thresholds should reflect natural signal scale differences")

        return self

    def predict(self, X):
        """
        Make predictions WITHOUT preprocessing.

        Uses trained Random Forest to classify neural activity patterns while
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

        # REMOVED: Any preprocessing transformations
        # Make predictions on completely unprocessed data
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities WITHOUT preprocessing.

        Provides probability estimates based on the natural signal characteristics
        learned during training.

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

        # REMOVED: Any preprocessing transformations
        # Predict probabilities on completely unprocessed data
        return self.model.predict_proba(X)

    def get_feature_importance(self, window_size: int, n_neurons: int) -> np.ndarray:
        """
        Get feature importance WITHOUT preprocessing effects.

        This reveals how Random Forest naturally weights different features when
        working with signals at their original scales. The importance values
        reflect the discriminative power of each feature in its natural units.

        Scientific Insight:
        - For raw calcium: Important features will be those fluorescence values
          that best distinguish movement vs. no-movement states
        - For ΔF/F: Important features will be relative changes that best predict behavior
        - For deconvolved: Important features will be spike patterns most predictive of movement

        These should be fundamentally different patterns!

        Parameters
        ----------
        window_size : int
            Size of the sliding window (temporal dimension)
        n_neurons : int
            Number of neurons (spatial dimension)

        Returns
        -------
        np.ndarray
            Feature importance matrix of shape (window_size, n_neurons)
            Values reflect natural discriminative power without preprocessing bias
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be trained before extracting feature importance")

        # Get feature importances directly from Random Forest
        # These values reflect how much each feature (at its natural scale)
        # contributed to reducing impurity across all trees
        importances = self.model.feature_importances_

        # Direct mapping since no PCA or other transformations were applied
        # Every feature corresponds directly to a (time_step, neuron) combination
        n_features = min(len(importances), window_size * n_neurons)
        importance_matrix = importances[:n_features].reshape(window_size, n_neurons)

        logger.info(f"Random Forest feature importance extracted WITHOUT preprocessing")
        logger.info(f"  Importances reflect natural signal scale differences:")
        logger.info(f"  - High importance = features that naturally distinguish movement patterns")
        logger.info(f"  - Values are based on original signal characteristics, not normalized features")
        logger.info(f"  - Different signal types should show different importance patterns")

        # Additional diagnostic information
        logger.info(f"  Importance statistics:")
        logger.info(f"    Mean: {importance_matrix.mean():.8f}")
        logger.info(f"    Max: {importance_matrix.max():.8f}")
        logger.info(f"    Non-zero features: {np.sum(importance_matrix > 0)}/{importance_matrix.size}")

        return importance_matrix

