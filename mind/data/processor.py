# """
# Data processing utilities for calcium imaging data.
# """
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from typing import Dict, Tuple, List, Optional, Any, Union
# from sklearn.model_selection import train_test_split
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# class SlidingWindowDataset(Dataset):
#     """
#     Dataset that creates sliding windows of neural activity for binary classification.
#
#     This dataset takes calcium imaging signals and creates windows of activity
#     for each frame, along with the corresponding behavior label.
#     """
#
#     def __init__(self,
#                  signal: np.ndarray,
#                  labels: np.ndarray,
#                  window_size: int = 15,
#                  step_size: int = 1,
#                  remove_zero_labels: bool = False):
#         """
#         Initialize a sliding window dataset.
#         """
#         self.signal = signal
#         self.labels = labels
#         self.window_size = window_size
#         self.step_size = step_size
#
#         # Calculate valid indices for windows
#         self.valid_indices = []
#
#         # Create sliding windows
#         n_frames = signal.shape[0]
#
#         for i in range(0, n_frames - window_size + 1, step_size):
#             # Get the label for this window (use the label of the last frame in the window)
#             window_label = labels[i + window_size - 1]
#
#             # If we're removing windows with zero labels, check the label
#             if remove_zero_labels and window_label == 0:
#                 continue
#
#             self.valid_indices.append(i)
#
#         logger.info(f"Created dataset with {len(self.valid_indices)} windows")
#
#     def __len__(self):
#         return len(self.valid_indices)
#
#     def __getitem__(self, idx):
#         # Get the starting index for this window
#         start_idx = self.valid_indices[idx]
#
#         # Extract the window
#         window = self.signal[start_idx:start_idx + self.window_size, :]
#
#         # Get the label for this window (use the label of the last frame in the window)
#         label = self.labels[start_idx + self.window_size - 1]
#
#         # Convert to tensors
#         window_tensor = torch.FloatTensor(window)
#         label_tensor = torch.LongTensor([label])
#
#         return window_tensor, label_tensor.squeeze()
#
#
# def create_datasets(calcium_signals: Dict[str, np.ndarray],
#                     frame_labels: np.ndarray,
#                     window_size: int = 15,
#                     step_size: int = 1,
#                     test_size: float = 0.15,
#                     val_size: float = 0.15,
#                     random_state: int = 42) -> Dict[str, Dict[str, SlidingWindowDataset]]:
#     """Create train, validation, and test datasets for each signal type."""
#     logger.info("Creating datasets from calcium signals")
#
#     # DEBUGGING: Check if signals are actually different
#     signal_ids = {}
#     signal_stats = {}
#
#     for signal_name, signal in calcium_signals.items():
#         if signal is not None:
#             signal_ids[signal_name] = id(signal)
#             signal_stats[signal_name] = {
#                 'mean': float(np.mean(signal)),
#                 'std': float(np.std(signal)),
#                 'shape': signal.shape,
#                 'first_10_elements': signal.flatten()[:10].tolist()
#             }
#
#     logger.info(f"🔍 SIGNAL IDENTITY CHECK:")
#     for name, sig_id in signal_ids.items():
#         logger.info(f"  {name}: id={sig_id}, mean={signal_stats[name]['mean']:.6f}")
#
#     # Check for identical objects
#     signal_names = list(signal_ids.keys())
#     for i in range(len(signal_names)):
#         for j in range(i + 1, len(signal_names)):
#             name1, name2 = signal_names[i], signal_names[j]
#             if signal_ids[name1] == signal_ids[name2]:
#                 logger.error(f"🚨 IDENTICAL OBJECTS: {name1} and {name2} share the same memory!")
#             elif np.array_equal(calcium_signals[name1], calcium_signals[name2]):
#                 logger.error(f"🚨 IDENTICAL ARRAYS: {name1} and {name2} have identical values!")
#             else:
#                 logger.info(f"✓ DIFFERENT: {name1} and {name2} are properly different")
#
#     # Create dataset dictionary
#     datasets = {}
#
#     # Process each signal type
#     for signal_name, signal in calcium_signals.items():
#         if signal is None:
#             logger.warning(f"Skipping {signal_name} because it is None")
#             continue
#
#         logger.info(f"Processing {signal_name}")
#
#         # DEBUGGING: Create a TRUE copy to ensure independence
#         signal_copy = signal.copy()  # Force a deep copy
#
#         logger.info(f"🔍 COPY CHECK for {signal_name}:")
#         logger.info(f"  Original id: {id(signal)}")
#         logger.info(f"  Copy id: {id(signal_copy)}")
#         logger.info(f"  Are same object? {signal is signal_copy}")
#         logger.info(f"  Have same values? {np.array_equal(signal, signal_copy)}")
#
#         # Create windows using the copy
#         full_dataset = SlidingWindowDataset(signal_copy, frame_labels,
#                                             window_size=window_size,
#                                             step_size=step_size)
#
#         # Rest of the function remains the same...
#         indices = np.arange(len(full_dataset))
#         train_val_indices, test_indices = train_test_split(
#             indices, test_size=test_size, random_state=random_state, stratify=None)
#         actual_val_size = val_size / (1 - test_size)
#         train_indices, val_indices = train_test_split(
#             train_val_indices, test_size=actual_val_size, random_state=random_state, stratify=None)
#
#         train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
#         val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
#         test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
#
#         datasets[signal_name] = {
#             'train': train_dataset,
#             'val': val_dataset,
#             'test': test_dataset
#         }
#
#         logger.info(
#             f"{signal_name} split sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
#
#     return datasets
#
#
# #
# # def create_datasets(calcium_signals: Dict[str, np.ndarray],
# #                     frame_labels: np.ndarray,
# #                     window_size: int = 15,
# #                     step_size: int = 1,
# #                     test_size: float = 0.15,
# #                     val_size: float = 0.15,
# #                     random_state: int = 42) -> Dict[str, Dict[str, SlidingWindowDataset]]:
# #     logger.info("Creating datasets from calcium signals")
# #     datasets = {}
# #
# #     for signal_name, signal in calcium_signals.items():
# #         if signal is None:
# #             logger.warning(f"Skipping {signal_name} because it is None")
# #             continue
# #
# #         logger.info(f"Processing {signal_name}")
# #
# #         print(f"\n🔍 DATASET CREATION CHECKPOINT:")
# #         print(f"Original {signal_name}: mean={signal.mean():.6f}, id={id(signal)}")
# #         print(f"Copy {signal_name}: mean={signal.mean():.6f}, id={id(signal)}")
# #         print(f"Are they the same object? {signal is signal}")
# #
# #         # Use the copy instead of the original
# #         full_dataset = SlidingWindowDataset(signal, frame_labels,
# #                                             window_size=window_size,
# #                                             step_size=step_size)
# #
# #         # Create indices for train/val/test split
# #         indices = np.arange(len(full_dataset))
# #
# #         # Split into train+val and test
# #         train_val_indices, test_indices = train_test_split(
# #             indices, test_size=test_size, random_state=random_state, stratify=None)
# #
# #         # Calculate actual validation size as a fraction of train+val
# #         actual_val_size = val_size / (1 - test_size)
# #
# #         # Split train+val into train and val
# #         train_indices, val_indices = train_test_split(
# #             train_val_indices, test_size=actual_val_size, random_state=random_state, stratify=None)
# #
# #         # Create subsets
# #         train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
# #         val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
# #         test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
# #
# #         # Store datasets
# #         datasets[signal_name] = {
# #             'train': train_dataset,
# #             'val': val_dataset,
# #             'test': test_dataset
# #         }
# #
# #         # Log split sizes
# #         logger.info(
# #             f"{signal_name} split sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
# #
# #     return datasets
# #
#
# def create_data_loaders(datasets: Dict[str, Dict[str, torch.utils.data.Dataset]],
#                         batch_size: int = 32,
#                         num_workers: int = 4) -> Dict[str, Dict[str, torch.utils.data.DataLoader]]:
#     """
#     Create DataLoader objects for each dataset.
#     """
#     logger.info(f"Creating DataLoaders with batch_size={batch_size}, num_workers={num_workers}")
#
#     dataloaders = {}
#
#     for signal_name, signal_datasets in datasets.items():
#         dataloaders[signal_name] = {}
#
#         for split_name, dataset in signal_datasets.items():
#             # Use different batch sizes for different splits if needed
#             current_batch_size = batch_size
#
#             # Create DataLoader
#             dataloader = DataLoader(
#                 dataset,
#                 batch_size=current_batch_size,
#                 shuffle=(split_name == 'train'),  # Only shuffle training data
#                 num_workers=num_workers,
#                 drop_last=False,
#                 pin_memory=True
#             )
#
#             dataloaders[signal_name][split_name] = dataloader
#
#             logger.info(f"Created DataLoader for {signal_name}/{split_name} with {len(dataloader)} batches")
#
#     return dataloaders
#
#
# def get_dataset_dimensions(datasets: Dict[str, Dict[str, torch.utils.data.Dataset]]) -> Dict[str, Tuple[int, int]]:
#     """
#     Get the dimensions (window_size, n_features) for each dataset.
#
#     Parameters
#     ----------
#     datasets : Dict[str, Dict[str, torch.utils.data.Dataset]]
#         Dictionary of datasets for each signal type and split
#
#     Returns
#     -------
#     Dict[str, Tuple[int, int]]
#         Dictionary of dimensions (window_size, n_features) for each signal type
#     """
#     dimensions = {}
#
#     for signal_name, signal_datasets in datasets.items():
#         # Get the first dataset (train)
#         dataset = signal_datasets['train']
#
#         # Get the first sample
#         X, _ = dataset[0]
#
#         # Get dimensions
#         if isinstance(X, torch.Tensor):
#             dimensions[signal_name] = (X.shape[0], X.shape[1])
#         else:
#             # Handle case where X is not a tensor (e.g., for classical models)
#             dimensions[signal_name] = (X.shape[0], X.shape[1])
#
#     return dimensions
#


"""
Fixed data processing with guaranteed signal independence.

This implementation ensures that each signal type maintains its unique characteristics
throughout the entire pipeline by using deep copies and signal-specific processing.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional, Any, Union
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class SlidingWindowDataset(Dataset):
    """
    Dataset that creates sliding windows with guaranteed signal independence.

    This dataset ensures that each signal type maintains its unique characteristics
    by storing the original signal properties and preventing any cross-contamination.
    """

    def __init__(self,
                 signal: np.ndarray,
                 labels: np.ndarray,
                 window_size: int = 15,
                 step_size: int = 1,
                 signal_name: str = "unknown",
                 remove_zero_labels: bool = False):
        """
        Initialize a sliding window dataset with signal identification.

        Parameters
        ----------
        signal : np.ndarray
            The calcium signal data (frames x neurons)
        labels : np.ndarray
            Behavioral labels for each frame
        window_size : int
            Size of sliding windows
        step_size : int
            Step size for sliding windows
        signal_name : str
            Name/type of the signal for tracking
        remove_zero_labels : bool
            Whether to remove windows with zero labels
        """
        # Make a DEEP copy to ensure complete independence
        self.signal = signal.copy()
        self.labels = labels.copy()
        self.window_size = window_size
        self.step_size = step_size
        self.signal_name = signal_name

        # Store original signal characteristics for verification
        self.signal_fingerprint = {
            'name': signal_name,
            'shape': self.signal.shape,
            'mean': float(self.signal.mean()),
            'std': float(self.signal.std()),
            'min': float(self.signal.min()),
            'max': float(self.signal.max()),
            'memory_id': id(self.signal)
        }

        # Calculate valid indices for windows
        self.valid_indices = []
        n_frames = signal.shape[0]

        for i in range(0, n_frames - window_size + 1, step_size):
            # Get the label for this window (use the label of the last frame in the window)
            window_label = labels[i + window_size - 1]

            # If removing zero labels, check the label
            if remove_zero_labels and window_label == 0:
                continue

            self.valid_indices.append(i)

        logger.info(f"Created {signal_name} dataset:")
        logger.info(f"  Windows: {len(self.valid_indices)}")
        logger.info(f"  Signal mean: {self.signal_fingerprint['mean']:.8f}")
        logger.info(f"  Signal std: {self.signal_fingerprint['std']:.8f}")
        logger.info(f"  Memory ID: {self.signal_fingerprint['memory_id']}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the starting index for this window
        start_idx = self.valid_indices[idx]

        # Extract the window from our independent signal copy
        window = self.signal[start_idx:start_idx + self.window_size, :]

        # Get the label for this window
        label = self.labels[start_idx + self.window_size - 1]

        # Convert to tensors
        window_tensor = torch.FloatTensor(window)
        label_tensor = torch.LongTensor([label])

        return window_tensor, label_tensor.squeeze()

    def verify_signal_integrity(self):
        """Verify that this dataset's signal maintains its original characteristics."""
        current_mean = float(self.signal.mean())
        current_std = float(self.signal.std())

        logger.info(f"Signal integrity check for {self.signal_name}:")
        logger.info(f"  Original mean: {self.signal_fingerprint['mean']:.8f}")
        logger.info(f"  Current mean:  {current_mean:.8f}")
        logger.info(f"  Original std:  {self.signal_fingerprint['std']:.8f}")
        logger.info(f"  Current std:   {current_std:.8f}")

        # Check if values have changed (allowing for tiny floating point differences)
        mean_unchanged = abs(current_mean - self.signal_fingerprint['mean']) < 1e-10
        std_unchanged = abs(current_std - self.signal_fingerprint['std']) < 1e-10

        if mean_unchanged and std_unchanged:
            logger.info(f"  ✓ Signal integrity PRESERVED for {self.signal_name}")
            return True
        else:
            logger.error(f"  ✗ Signal integrity VIOLATED for {self.signal_name}")
            return False


def create_datasets(calcium_signals: Dict[str, np.ndarray],
                    frame_labels: np.ndarray,
                    window_size: int = 15,
                    step_size: int = 1,
                    test_size: float = 0.15,
                    val_size: float = 0.15,
                    random_state: int = 42) -> Dict[str, Dict[str, SlidingWindowDataset]]:
    """
    Create datasets with GUARANTEED signal independence.

    This function ensures that each signal type maintains its unique characteristics
    by using deep copies, signal-specific random states, and integrity verification.
    """
    logger.info("Creating datasets with GUARANTEED signal independence")

    # First, verify that input signals are actually different
    signal_fingerprints = {}
    for signal_name, signal in calcium_signals.items():
        if signal is not None:
            fingerprint = {
                'mean': float(signal.mean()),
                'std': float(signal.std()),
                'min': float(signal.min()),
                'max': float(signal.max()),
                'id': id(signal)
            }
            signal_fingerprints[signal_name] = fingerprint
            logger.info(f"Input {signal_name}: mean={fingerprint['mean']:.8f}, id={fingerprint['id']}")

    # Check for identical input signals (this would indicate upstream problems)
    signal_names = list(signal_fingerprints.keys())
    for i in range(len(signal_names)):
        for j in range(i + 1, len(signal_names)):
            name1, name2 = signal_names[i], signal_names[j]
            fp1, fp2 = signal_fingerprints[name1], signal_fingerprints[name2]

            if fp1['id'] == fp2['id']:
                logger.error(f"🚨 CRITICAL: {name1} and {name2} share the same memory!")
                raise ValueError(f"Signal contamination detected: {name1} and {name2} are the same object")

            # Check if values are suspiciously similar
            mean_diff = abs(fp1['mean'] - fp2['mean'])
            std_diff = abs(fp1['std'] - fp2['std'])

            if mean_diff < 1e-6 and std_diff < 1e-6:
                logger.error(f"🚨 CRITICAL: {name1} and {name2} have identical values!")
                raise ValueError(f"Signal contamination detected: {name1} and {name2} have identical data")
            else:
                logger.info(f"✓ VERIFIED: {name1} and {name2} are properly different")

    # Create datasets with complete independence
    datasets = {}

    for signal_name, signal in calcium_signals.items():
        if signal is None:
            logger.warning(f"Skipping {signal_name} because it is None")
            continue

        logger.info(f"Processing {signal_name} with complete independence")

        # Create a DEEP, INDEPENDENT copy of the signal
        # This ensures absolutely no cross-contamination between signal types
        signal_deep_copy = signal.copy()

        # Verify the copy is independent but has same values
        assert signal_deep_copy is not signal, f"Copy failed for {signal_name}"
        assert np.array_equal(signal_deep_copy, signal), f"Copy values incorrect for {signal_name}"

        logger.info(f"Deep copy verified for {signal_name}:")
        logger.info(f"  Original ID: {id(signal)}")
        logger.info(f"  Copy ID: {id(signal_deep_copy)}")
        logger.info(f"  Values identical: {np.array_equal(signal_deep_copy, signal)}")
        logger.info(f"  Objects different: {signal_deep_copy is not signal}")

        # Create labels copy for this signal (also independent)
        labels_copy = frame_labels.copy()

        # Create signal-specific random state to ensure complete independence
        # Each signal gets a unique random state based on its characteristics
        signal_hash = hash((
            signal_name,
            signal_deep_copy.shape[0],
            signal_deep_copy.shape[1],
            float(signal_deep_copy.mean()),
            float(signal_deep_copy.std())
        ))
        signal_random_state = (random_state + abs(signal_hash)) % (2 ** 31 - 1)

        logger.info(f"Signal-specific random state for {signal_name}: {signal_random_state}")

        # Create the sliding window dataset with complete independence
        full_dataset = SlidingWindowDataset(
            signal=signal_deep_copy,
            labels=labels_copy,
            window_size=window_size,
            step_size=step_size,
            signal_name=signal_name
        )

        # Verify signal integrity after dataset creation
        full_dataset.verify_signal_integrity()

        # Create train/val/test splits with signal-specific randomization
        indices = np.arange(len(full_dataset))

        # Split into train+val and test using signal-specific random state
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=signal_random_state,  # Signal-specific randomization!
            stratify=None
        )

        # Calculate actual validation size as a fraction of train+val
        actual_val_size = val_size / (1 - test_size)

        # Split train+val into train and val using signal-specific randomization
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=actual_val_size,
            random_state=signal_random_state + 1,  # Different but related random state
            stratify=None
        )

        # Create subsets with complete independence
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        # Store datasets
        datasets[signal_name] = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

        logger.info(f"{signal_name} splits (independent randomization):")
        logger.info(f"  Train: {len(train_dataset)} samples")
        logger.info(f"  Val: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        logger.info(f"  Random state used: {signal_random_state}")

    # Final verification: ensure all datasets maintain signal independence
    logger.info("\n🔍 FINAL SIGNAL INDEPENDENCE VERIFICATION:")

    # Extract first samples from each dataset to verify they're different
    first_samples = {}
    for signal_name, signal_datasets in datasets.items():
        first_sample, _ = signal_datasets['train'][0]
        first_samples[signal_name] = {
            'mean': float(first_sample.mean()),
            'std': float(first_sample.std()),
            'shape': first_sample.shape
        }
        logger.info(f"  {signal_name} first sample: mean={first_samples[signal_name]['mean']:.8f}")

    # Cross-check that samples are actually different
    signal_names_list = list(first_samples.keys())
    for i in range(len(signal_names_list)):
        for j in range(i + 1, len(signal_names_list)):
            name1, name2 = signal_names_list[i], signal_names_list[j]
            sample1, sample2 = first_samples[name1], first_samples[name2]

            mean_diff = abs(sample1['mean'] - sample2['mean'])
            if mean_diff < 1e-6:
                logger.error(f"🚨 FINAL CHECK FAILED: {name1} and {name2} samples are identical!")
                raise ValueError(f"Dataset contamination: {name1} and {name2} have identical samples")
            else:
                logger.info(f"✓ FINAL VERIFIED: {name1} and {name2} samples are different (mean diff: {mean_diff:.8f})")

    logger.info("✓ ALL SIGNALS VERIFIED INDEPENDENT")
    return datasets


def create_data_loaders(datasets: Dict[str, Dict[str, torch.utils.data.Dataset]],
                        batch_size: int = 32,
                        num_workers: int = 4) -> Dict[str, Dict[str, torch.utils.data.DataLoader]]:
    """
    Create DataLoader objects for each dataset with signal independence.
    """
    logger.info(f"Creating DataLoaders with guaranteed signal independence")
    logger.info(f"  Batch size: {batch_size}, Workers: {num_workers}")

    dataloaders = {}

    for signal_name, signal_datasets in datasets.items():
        dataloaders[signal_name] = {}

        for split_name, dataset in signal_datasets.items():
            # Create DataLoader with signal-specific settings
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split_name == 'train'),  # Only shuffle training data
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True
            )

            dataloaders[signal_name][split_name] = dataloader
            logger.info(f"Created DataLoader for {signal_name}/{split_name}: {len(dataloader)} batches")

    return dataloaders


def get_dataset_dimensions(datasets: Dict[str, Dict[str, torch.utils.data.Dataset]]) -> Dict[str, Tuple[int, int]]:
    """
    Get the dimensions for each dataset while preserving signal independence.
    """
    dimensions = {}

    for signal_name, signal_datasets in datasets.items():
        # Get the first dataset (train)
        dataset = signal_datasets['train']

        # Get the first sample
        X, _ = dataset[0]

        # Get dimensions
        if isinstance(X, torch.Tensor):
            dimensions[signal_name] = (X.shape[0], X.shape[1])
        else:
            dimensions[signal_name] = (X.shape[0], X.shape[1])

        logger.info(f"{signal_name} dimensions: {dimensions[signal_name]}")

    return dimensions

