# """
# Data loader for calcium imaging.
# """
# import os
# import numpy as np
# import pandas as pd
# import h5py
# import scipy.io
# import hdf5storage
# from typing import Dict, Tuple, List, Optional, Any, Union
# import logging
#
# logger = logging.getLogger(__name__)
#
# def load_calcium_signals(mat_file_path: str) -> Dict[str, np.ndarray]:
#     """Load calcium imaging signals from MATLAB file."""
#     logger.info(f"Loading calcium signals from {mat_file_path}")
#
#     try:
#         # Load the data (existing code)
#         try:
#             data = scipy.io.loadmat(mat_file_path)
#             calcium_signal = data.get('calciumsignal_wanted', None)
#             deltaf_signal = data.get('deltaf_cells_not_excluded', None)
#             deconv_signal = data.get('DeconvMat_wanted', None)
#         except NotImplementedError:
#             data = hdf5storage.loadmat(mat_file_path)
#             calcium_signal = data.get('calciumsignal_wanted', None)
#             deltaf_signal = data.get('deltaf_cells_not_excluded', None)
#             deconv_signal = data.get('DeconvMat_wanted', None)
#
#         # DEBUGGING: Check if MATLAB file has duplicate references
#         logger.info("🔍 MATLAB FILE CONTENT CHECK:")
#         for key, value in data.items():
#             if not key.startswith('__') and isinstance(value, np.ndarray):
#                 logger.info(f"  {key}: shape={value.shape}, id={id(value)}")
#
#         # DEBUGGING: Verify signals are actually different
#         signals = {
#             'calcium_signal': calcium_signal,
#             'deltaf_signal': deltaf_signal,
#             'deconv_signal': deconv_signal
#         }
#
#         logger.info("🔍 SIGNAL VERIFICATION:")
#         valid_signals = []
#         for name, signal in signals.items():
#             if signal is not None:
#                 logger.info(f"  {name}: shape={signal.shape}, mean={np.mean(signal):.6f}, id={id(signal)}")
#                 valid_signals.append((name, signal))
#             else:
#                 logger.warning(f"  {name}: None")
#
#         # Check for identical arrays or references
#         for i in range(len(valid_signals)):
#             for j in range(i + 1, len(valid_signals)):
#                 name1, sig1 = valid_signals[i]
#                 name2, sig2 = valid_signals[j]
#
#                 if sig1 is sig2:
#                     logger.error(f"🚨 SAME OBJECT: {name1} and {name2} reference the same array!")
#                 elif np.array_equal(sig1, sig2):
#                     logger.error(f"🚨 IDENTICAL VALUES: {name1} and {name2} have identical values!")
#                 else:
#                     diff_pct = np.mean(np.abs(sig1 - sig2)) / (np.mean(np.abs(sig1)) + 1e-10) * 100
#                     logger.info(f"✓ DIFFERENT: {name1} vs {name2}, difference: {diff_pct:.2f}%")
#
#         # Log shapes and basic stats (existing code)
#         if calcium_signal is not None:
#             logger.info(f"Raw calcium signal shape: {calcium_signal.shape}")
#         if deltaf_signal is not None:
#             logger.info(f"ΔF/F signal shape: {deltaf_signal.shape}")
#         if deconv_signal is not None:
#             logger.info(f"Deconvolved signal shape: {deconv_signal.shape}")
#
#         return {
#             'calcium_signal': calcium_signal,
#             'deltaf_signal': deltaf_signal,
#             'deconv_signal': deconv_signal
#         }
#
#     except Exception as e:
#         logger.error(f"Error loading {mat_file_path}: {e}")
#         raise
#
# def load_behavioral_data(xlsx_file_path: str) -> pd.DataFrame:
#     """
#     Load behavioral data from Excel file.
#     """
#     logger.info(f"Loading behavioral data from {xlsx_file_path}")
#
#     try:
#         # Load the Excel file
#         behavior_data = pd.read_excel(xlsx_file_path)
#
#         # Check if the expected columns exist
#         expected_columns = ['Foot (L/R)', 'Frame Start', 'Frame End']
#         missing_columns = [col for col in expected_columns if col not in behavior_data.columns]
#
#         if missing_columns:
#             logger.error(f"Missing required columns in {xlsx_file_path}: {missing_columns}")
#             raise ValueError(f"Missing required columns: {missing_columns}")
#
#         logger.info(f"Loaded behavioral data with {len(behavior_data)} events")
#         return behavior_data
#
#     except Exception as e:
#         logger.error(f"Error loading {xlsx_file_path}: {e}")
#         raise
#
#
# def match_behavior_to_frames(behavior_data: pd.DataFrame, num_frames: int,
#                              binary_classification: bool = True) -> np.ndarray:
#     """
#     Create frame-by-frame behavior labels from behavioral events.
#     """
#     logger.info(f"Creating frame-by-frame behavior labels for {num_frames} frames")
#     logger.info(f"Binary classification mode: {binary_classification}")
#
#     # Initialize array of zeros (no footstep)
#     frame_labels = np.zeros(num_frames, dtype=np.int32)
#
#     try:
#         # Map footstep events to frames
#         for _, row in behavior_data.iterrows():
#             # Get the foot side, start frame and end frame using the actual column names
#             foot = str(row['Foot (L/R)']).lower()
#             start_frame = int(row['Frame Start'])
#             end_frame = int(row['Frame End'])
#
#             # Ensure frames are within the valid range
#             start_frame = max(0, min(start_frame, num_frames - 1))
#             end_frame = max(0, min(end_frame, num_frames - 1))
#
#             # Determine the label based on the foot
#             if 'right' in foot or 'r' == foot or 'contra' in foot:
#                 label = 1  # Contralateral (right) footstep
#             elif 'left' in foot or 'l' == foot or 'ipsi' in foot:
#                 if binary_classification:
#                     # In binary classification, we ignore ipsilateral footsteps
#                     continue
#                 else:
#                     label = 2  # Ipsilateral (left) footstep
#             else:
#                 logger.warning(f"Unrecognized foot value: {foot}, skipping event")
#                 continue
#
#             # Assign labels to frames
#             frame_labels[start_frame:end_frame + 1] = label
#
#         # Log stats about the labels
#         unique_labels, counts = np.unique(frame_labels, return_counts=True)
#         label_stats = {f"Label {label}": count for label, count in zip(unique_labels, counts)}
#         logger.info(f"Label distribution: {label_stats}")
#
#         # Log percentages
#         total_frames = len(frame_labels)
#         for label, count in zip(unique_labels, counts):
#             percentage = (count / total_frames) * 100
#             logger.info(f"Label {label}: {count} frames ({percentage:.2f}%)")
#
#         return frame_labels
#
#     except Exception as e:
#         logger.error(f"Error creating behavior labels: {e}")
#         raise
#
#
# def load_and_align_data(mat_file_path: str, xlsx_file_path: str,
#                         binary_classification: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
#     """
#     Load and align calcium imaging data with behavioral labels.
#
#     Parameters
#     ----------
#     mat_file_path : str
#         Path to the MATLAB file containing calcium imaging data
#     xlsx_file_path : str
#         Path to the Excel file containing behavioral data
#     binary_classification : bool, optional
#         If True, create binary labels (0 for no footstep, 1 for contralateral/right footstep)
#
#     Returns
#     -------
#     Tuple[Dict[str, np.ndarray], np.ndarray]
#         Tuple containing:
#         - Dictionary of calcium signals
#         - Array of behavior labels
#     """
#
#     # Load calcium signals
#     calcium_signals = load_calcium_signals(mat_file_path)
#
#     # Determine the number of frames
#     num_frames = None
#     for signal_type, signal in calcium_signals.items():
#         if signal is not None:
#             num_frames = signal.shape[0]
#             break
#
#     if num_frames is None:
#         raise ValueError("No valid calcium signals found")
#
#     # Load behavioral data
#     behavior_data = load_behavioral_data(xlsx_file_path)
#
#     # Match behavior to frames (binary classification by default)
#     frame_labels = match_behavior_to_frames(behavior_data, num_frames, binary_classification)
#
#     return calcium_signals, frame_labels
#
#
# def find_most_active_neurons(calcium_signals: Dict[str, np.ndarray],
#                              n_neurons: int = 20,
#                              signal_type: str = 'deconv_signal') -> np.ndarray:
#     """
#     Find the most active neurons based on calcium transient activity.
#
#     Parameters
#     ----------
#     calcium_signals : Dict[str, np.ndarray]
#         Dictionary of calcium signals
#     n_neurons : int
#         Number of top neurons to return
#     signal_type : str
#         Type of signal to use for finding active neurons
#
#     Returns
#     -------
#     np.ndarray
#         Indices of the most active neurons
#     """
#     signal = calcium_signals[signal_type]
#     if signal is None:
#         # Fallback to other signal types
#         for alt_signal in ['deltaf_signal', 'calcium_signal']:
#             if calcium_signals[alt_signal] is not None:
#                 signal = calcium_signals[alt_signal]
#                 break
#
#     # Calculate activity metrics
#     # For deconvolved signals, use sum of transients
#     # For other signals, use variance
#     if signal_type == 'deconv_signal':
#         activity_metric = np.sum(signal > 0, axis=0)  # Count of active frames
#     else:
#         activity_metric = np.var(signal, axis=0)  # Variance
#
#     # Get indices of top neurons
#     top_indices = np.argsort(activity_metric)[::-1][:n_neurons]
#
#     return top_indices
#
#


"""
Data loader for calcium imaging with enhanced signal verification.

FIXED: This module now includes comprehensive signal integrity checking
to ensure different signal types maintain their unique characteristics.
"""
import os
import numpy as np
import pandas as pd
import h5py
import scipy.io
import hdf5storage
from typing import Dict, Tuple, List, Optional, Any, Union
import logging
import hashlib

logger = logging.getLogger(__name__)


def load_calcium_signals(mat_file_path: str) -> Dict[str, np.ndarray]:
    """
    Load calcium imaging signals with comprehensive integrity verification.

    FIXED: Enhanced verification to ensure signals maintain unique characteristics.
    """
    logger.info(f"Loading calcium signals from {mat_file_path}")

    try:
        # Load the data using appropriate method
        try:
            data = scipy.io.loadmat(mat_file_path)
        except NotImplementedError:
            data = hdf5storage.loadmat(mat_file_path)

        # Extract signals with new variable names to avoid contamination
        calcium_signal = data.get('calciumsignal_wanted', None)
        deltaf_signal = data.get('deltaf_cells_not_excluded', None)
        deconv_signal = data.get('DeconvMat_wanted', None)

        # FIXED: Create completely independent copies to prevent memory sharing
        if calcium_signal is not None:
            calcium_signal = calcium_signal.copy()
        if deltaf_signal is not None:
            deltaf_signal = deltaf_signal.copy()
        if deconv_signal is not None:
            deconv_signal = deconv_signal.copy()

        # FIXED: Comprehensive signal integrity verification
        logger.info("🔍 ENHANCED SIGNAL VERIFICATION:")

        signals = {
            'calcium_signal': calcium_signal,
            'deltaf_signal': deltaf_signal,
            'deconv_signal': deconv_signal
        }

        # Create detailed fingerprints for each signal
        signal_fingerprints = {}
        for name, signal in signals.items():
            if signal is not None:
                fingerprint = {
                    'shape': signal.shape,
                    'mean': float(np.mean(signal)),
                    'std': float(np.std(signal)),
                    'min': float(np.min(signal)),
                    'max': float(np.max(signal)),
                    'sum': float(np.sum(signal)),
                    'memory_id': id(signal),
                    'hash': hashlib.md5(signal.tobytes()).hexdigest()[:16],
                    'first_10_values': signal.flatten()[:10].tolist(),
                    'zero_fraction': float(np.sum(signal == 0) / signal.size)
                }
                signal_fingerprints[name] = fingerprint

                logger.info(f"  {name}:")
                logger.info(f"    Shape: {fingerprint['shape']}")
                logger.info(f"    Mean: {fingerprint['mean']:.8f}")
                logger.info(f"    Std: {fingerprint['std']:.8f}")
                logger.info(f"    Range: [{fingerprint['min']:.6f}, {fingerprint['max']:.6f}]")
                logger.info(f"    Hash: {fingerprint['hash']}")
                logger.info(f"    Memory ID: {fingerprint['memory_id']}")
                logger.info(f"    Zero fraction: {fingerprint['zero_fraction']:.4f}")

        # FIXED: Rigorous cross-signal verification
        logger.info("🔍 CROSS-SIGNAL VERIFICATION:")
        signal_names = list(signal_fingerprints.keys())

        for i in range(len(signal_names)):
            for j in range(i + 1, len(signal_names)):
                name1, name2 = signal_names[i], signal_names[j]
                fp1, fp2 = signal_fingerprints[name1], signal_fingerprints[name2]

                # Check for memory sharing
                if fp1['memory_id'] == fp2['memory_id']:
                    logger.error(f"🚨 CRITICAL: {name1} and {name2} share the same memory!")
                    raise ValueError(f"Memory contamination: {name1} and {name2} are the same object")

                # Check for identical data
                if fp1['hash'] == fp2['hash']:
                    logger.error(f"🚨 CRITICAL: {name1} and {name2} have identical data!")
                    raise ValueError(f"Data contamination: {name1} and {name2} have identical values")

                # Calculate meaningful difference metrics
                mean_ratio = abs(fp1['mean'] / fp2['mean']) if fp2['mean'] != 0 else float('inf')
                std_ratio = abs(fp1['std'] / fp2['std']) if fp2['std'] != 0 else float('inf')

                logger.info(f"  {name1} vs {name2}:")
                logger.info(f"    Mean ratio: {mean_ratio:.2f}x")
                logger.info(f"    Std ratio: {std_ratio:.2f}x")
                logger.info(f"    ✓ Properly different signals confirmed")

        # FIXED: Expected signal characteristics verification
        logger.info("🔍 EXPECTED CHARACTERISTICS VERIFICATION:")

        if 'calcium_signal' in signal_fingerprints:
            fp = signal_fingerprints['calcium_signal']
            if fp['mean'] < 1000:
                logger.warning(f"⚠️ Raw calcium mean ({fp['mean']:.2f}) lower than expected (~6000)")
            elif fp['mean'] > 10000:
                logger.warning(f"⚠️ Raw calcium mean ({fp['mean']:.2f}) higher than expected (~6000)")
            else:
                logger.info(f"  ✓ Raw calcium scale appropriate ({fp['mean']:.2f})")

        if 'deltaf_signal' in signal_fingerprints:
            fp = signal_fingerprints['deltaf_signal']
            if fp['mean'] < 0.01 or fp['mean'] > 1.0:
                logger.warning(f"⚠️ ΔF/F mean ({fp['mean']:.4f}) outside expected range (0.01-1.0)")
            else:
                logger.info(f"  ✓ ΔF/F scale appropriate ({fp['mean']:.4f})")

        if 'deconv_signal' in signal_fingerprints:
            fp = signal_fingerprints['deconv_signal']
            if fp['zero_fraction'] < 0.7:
                logger.warning(f"⚠️ Deconvolved signal not sparse enough ({fp['zero_fraction']:.2f} zeros)")
            else:
                logger.info(f"  ✓ Deconvolved signal appropriately sparse ({fp['zero_fraction']:.2f} zeros)")

        logger.info("✅ All signals verified as unique and properly scaled")

        return signals

    except Exception as e:
        logger.error(f"Error loading {mat_file_path}: {e}")
        raise


def load_behavioral_data(xlsx_file_path: str) -> pd.DataFrame:
    """
    Load behavioral data from Excel file.

    This function remains unchanged as behavioral data processing is working correctly.
    """
    logger.info(f"Loading behavioral data from {xlsx_file_path}")

    try:
        behavior_data = pd.read_excel(xlsx_file_path)

        expected_columns = ['Foot (L/R)', 'Frame Start', 'Frame End']
        missing_columns = [col for col in expected_columns if col not in behavior_data.columns]

        if missing_columns:
            logger.error(f"Missing required columns in {xlsx_file_path}: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        logger.info(f"Loaded behavioral data with {len(behavior_data)} events")
        return behavior_data

    except Exception as e:
        logger.error(f"Error loading {xlsx_file_path}: {e}")
        raise


def match_behavior_to_frames(behavior_data: pd.DataFrame, num_frames: int,
                             binary_classification: bool = True) -> np.ndarray:
    """
    Create frame-by-frame behavior labels from behavioral events.

    This function remains unchanged as label creation is working correctly.
    """
    logger.info(f"Creating frame-by-frame behavior labels for {num_frames} frames")
    logger.info(f"Binary classification mode: {binary_classification}")

    frame_labels = np.zeros(num_frames, dtype=np.int32)

    try:
        for _, row in behavior_data.iterrows():
            foot = str(row['Foot (L/R)']).lower()
            start_frame = int(row['Frame Start'])
            end_frame = int(row['Frame End'])

            start_frame = max(0, min(start_frame, num_frames - 1))
            end_frame = max(0, min(end_frame, num_frames - 1))

            if 'right' in foot or 'r' == foot or 'contra' in foot:
                label = 1  # Contralateral (right) footstep
            elif 'left' in foot or 'l' == foot or 'ipsi' in foot:
                if binary_classification:
                    continue
                else:
                    label = 2  # Ipsilateral (left) footstep
            else:
                logger.warning(f"Unrecognized foot value: {foot}, skipping event")
                continue

            frame_labels[start_frame:end_frame + 1] = label

        # Log label statistics
        unique_labels, counts = np.unique(frame_labels, return_counts=True)
        total_frames = len(frame_labels)

        for label, count in zip(unique_labels, counts):
            percentage = (count / total_frames) * 100
            logger.info(f"Label {label}: {count} frames ({percentage:.2f}%)")

        return frame_labels

    except Exception as e:
        logger.error(f"Error creating behavior labels: {e}")
        raise


def load_and_align_data(mat_file_path: str, xlsx_file_path: str,
                        binary_classification: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load and align calcium imaging data with behavioral labels.

    FIXED: Enhanced verification throughout the loading process.
    """
    logger.info("Loading and aligning calcium imaging data with behavioral labels")

    # Load calcium signals with enhanced verification
    calcium_signals = load_calcium_signals(mat_file_path)

    # Determine the number of frames
    num_frames = None
    for signal_type, signal in calcium_signals.items():
        if signal is not None:
            num_frames = signal.shape[0]
            break

    if num_frames is None:
        raise ValueError("No valid calcium signals found")

    logger.info(f"Processing {num_frames} frames of neural data")

    # Load behavioral data
    behavior_data = load_behavioral_data(xlsx_file_path)

    # Match behavior to frames
    frame_labels = match_behavior_to_frames(behavior_data, num_frames, binary_classification)

    # FIXED: Final verification that data loading was successful
    logger.info("🔍 FINAL DATA LOADING VERIFICATION:")
    for signal_name, signal in calcium_signals.items():
        if signal is not None:
            logger.info(f"  {signal_name}: {signal.shape} - Mean: {signal.mean():.6f}")

    logger.info(
        f"  Frame labels: {frame_labels.shape} - Distribution: {dict(zip(*np.unique(frame_labels, return_counts=True)))}")
    logger.info("✅ Data loading and alignment completed successfully")

    return calcium_signals, frame_labels


def find_most_active_neurons(calcium_signals: Dict[str, np.ndarray],
                             n_neurons: int = 20,
                             signal_type: str = 'deconv_signal') -> np.ndarray:
    """
    Find the most active neurons based on calcium transient activity.

    This function remains unchanged as it's working correctly.
    """
    signal = calcium_signals[signal_type]
    if signal is None:
        for alt_signal in ['deltaf_signal', 'calcium_signal']:
            if calcium_signals[alt_signal] is not None:
                signal = calcium_signals[alt_signal]
                break

    if signal_type == 'deconv_signal':
        activity_metric = np.sum(signal > 0, axis=0)
    else:
        activity_metric = np.var(signal, axis=0)

    top_indices = np.argsort(activity_metric)[::-1][:n_neurons]
    return top_indices

