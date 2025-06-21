"""
Targeted debugging script for Python pipeline data contamination.
This will trace data through your specific pipeline to find where identical data appears.
"""
import numpy as np
import torch
import hashlib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DataTracker:
    """Class to track data transformations and detect contamination."""

    def __init__(self):
        self.history = []
        self.fingerprints = {}

    def create_fingerprint(self, array, name="", stage=""):
        """Create a detailed fingerprint of array data."""
        if array is None:
            return None

        # Convert torch tensors to numpy for consistent fingerprinting
        if hasattr(array, 'numpy'):
            array = array.numpy()
        elif hasattr(array, 'cpu'):
            array = array.cpu().numpy()

        fingerprint = {
            'name': name,
            'stage': stage,
            'shape': array.shape,
            'dtype': str(array.dtype),
            'mean': float(np.mean(array)),
            'std': float(np.std(array)),
            'sum': float(np.sum(array)),
            'memory_id': id(array),
            'hash': hashlib.md5(array.tobytes()).hexdigest(),
            'first_5_elements': array.flatten()[:5].tolist(),
            'last_5_elements': array.flatten()[-5:].tolist()
        }

        # Store in history
        key = f"{stage}_{name}"
        self.fingerprints[key] = fingerprint
        self.history.append((stage, name, fingerprint))

        return fingerprint

    def track_signals(self, signals_dict, stage=""):
        """Track all signals in a dictionary."""
        tracked = {}
        for name, signal in signals_dict.items():
            if signal is not None:
                fp = self.create_fingerprint(signal, name, stage)
                tracked[name] = fp
                print(f"  📊 {stage} - {name}: mean={fp['mean']:.8f}, hash={fp['hash'][:12]}...")
        return tracked

    def compare_stages(self, stage1, stage2, signal_name):
        """Compare the same signal across different stages."""
        key1 = f"{stage1}_{signal_name}"
        key2 = f"{stage2}_{signal_name}"

        if key1 not in self.fingerprints or key2 not in self.fingerprints:
            print(f"  ❌ Cannot compare {signal_name}: missing fingerprints")
            return False

        fp1 = self.fingerprints[key1]
        fp2 = self.fingerprints[key2]

        print(f"\n🔍 COMPARING {signal_name}: {stage1} → {stage2}")
        print(f"  Hash: {fp1['hash'][:12]}... → {fp2['hash'][:12]}...")
        print(f"  Mean: {fp1['mean']:.8f} → {fp2['mean']:.8f}")
        print(f"  Memory ID: {fp1['memory_id']} → {fp2['memory_id']}")

        if fp1['hash'] == fp2['hash']:
            if fp1['memory_id'] == fp2['memory_id']:
                print(f"  🔄 SAME OBJECT (no copy made)")
            else:
                print(f"  ✅ PROPER COPY (same data, different object)")
            return True
        else:
            print(f"  🚨 DATA CHANGED UNEXPECTEDLY!")
            return False

    def find_contamination(self):
        """Look for signals with identical hashes across different signal types."""
        print(f"\n🕵️ SEARCHING FOR CONTAMINATION:")

        # Group fingerprints by stage
        stages = {}
        for stage, name, fp in self.history:
            if stage not in stages:
                stages[stage] = {}
            stages[stage][name] = fp

        # Check each stage for contamination
        contamination_found = False
        for stage_name, stage_data in stages.items():
            signal_names = list(stage_data.keys())

            print(f"\n  📋 Stage: {stage_name}")
            for i in range(len(signal_names)):
                for j in range(i + 1, len(signal_names)):
                    name1, name2 = signal_names[i], signal_names[j]
                    fp1, fp2 = stage_data[name1], stage_data[name2]

                    if fp1['hash'] == fp2['hash']:
                        print(f"    🚨 CONTAMINATION: {name1} and {name2} have identical data!")
                        print(f"       Hash: {fp1['hash'][:16]}...")
                        print(f"       Mean: {fp1['mean']:.8f}")
                        contamination_found = True
                    else:
                        mean_diff = abs(fp1['mean'] - fp2['mean'])
                        print(f"    ✅ {name1} vs {name2}: properly different (mean diff: {mean_diff:.6f})")

        if not contamination_found:
            print(f"  ✅ No contamination detected in any stage")

        return contamination_found

# Integration with your existing pipeline
def debug_load_and_align_data(mat_file_path, xlsx_file_path):
    """Debug version of your load_and_align_data function."""
    print(f"\n" + "="*80)
    print(f"🔍 DEBUGGING LOAD_AND_ALIGN_DATA")
    print(f"="*80)

    tracker = DataTracker()

    # Import your actual loading function
    from mind.data.loader import load_calcium_signals, load_behavioral_data, match_behavior_to_frames

    # Step 1: Load calcium signals
    print(f"\n📥 Step 1: Loading calcium signals...")
    calcium_signals = load_calcium_signals(mat_file_path)
    tracker.track_signals(calcium_signals, "1_loaded")

    # Step 2: Create frame labels (using first signal to get num_frames)
    print(f"\n📝 Step 2: Creating frame labels...")
    num_frames = None
    for signal_type, signal in calcium_signals.items():
        if signal is not None:
            num_frames = signal.shape[0]
            break

    behavior_data = load_behavioral_data(xlsx_file_path)
    frame_labels = match_behavior_to_frames(behavior_data, num_frames, True)

    # Step 3: Check contamination after loading
    tracker.find_contamination()

    return calcium_signals, frame_labels, tracker

def debug_create_datasets(calcium_signals, frame_labels, tracker, config):
    """Debug version of your create_datasets function."""
    print(f"\n" + "="*80)
    print(f"🔍 DEBUGGING CREATE_DATASETS")
    print(f"="*80)

    # Import your actual dataset creation function
    from mind.data.processor import create_datasets

    # Track signals before dataset creation
    print(f"\n📊 Signals before dataset creation:")
    tracker.track_signals(calcium_signals, "2_before_datasets")

    # Create datasets
    print(f"\n🏗️ Creating datasets...")
    datasets = create_datasets(
        calcium_signals=calcium_signals,
        frame_labels=frame_labels,
        window_size=config["data"]["window_size"],
        step_size=config["data"]["step_size"],
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_state=42
    )

    # Extract actual data from datasets to check
    print(f"\n📊 Extracting first samples from datasets:")
    dataset_samples = {}
    for signal_name, signal_datasets in datasets.items():
        # Get the first sample from training dataset
        first_sample, _ = signal_datasets['train'][0]
        dataset_samples[signal_name] = first_sample
        print(f"  📦 {signal_name} first sample shape: {first_sample.shape}")

    tracker.track_signals(dataset_samples, "3_first_samples")

    # Check for contamination
    tracker.find_contamination()

    return datasets, tracker

def debug_model_training_data(datasets, signal_type, tracker):
    """Debug the data extraction for model training."""
    print(f"\n" + "="*80)
    print(f"🔍 DEBUGGING MODEL TRAINING DATA EXTRACTION")
    print(f"Signal type: {signal_type}")
    print(f"="*80)

    # Extract training data the same way your trainer does
    train_dataset = datasets[signal_type]['train']
    val_dataset = datasets[signal_type]['val']
    test_dataset = datasets[signal_type]['test']

    # Extract data exactly as your trainer does
    X_train = torch.stack([x for x, _ in train_dataset])
    y_train = torch.tensor([y.item() for _, y in train_dataset])
    X_val = torch.stack([x for x, _ in val_dataset])
    y_val = torch.tensor([y.item() for _, y in val_dataset])
    X_test = torch.stack([x for x, _ in test_dataset])
    y_test = torch.tensor([y.item() for _, y in test_dataset])

    # Track the extracted data
    training_data = {
        f"{signal_type}_X_train": X_train,
        f"{signal_type}_X_val": X_val,
        f"{signal_type}_X_test": X_test
    }

    print(f"\n📊 Extracted training data:")
    tracker.track_signals(training_data, "4_training_extraction")

    return X_train, y_train, X_val, y_val, X_test, y_test, tracker

# Main debugging function that runs your entire pipeline
def debug_full_pipeline(mat_file_path, xlsx_file_path, config):
    """Run your entire pipeline with comprehensive debugging."""
    print(f"\n🚀 STARTING FULL PIPELINE DEBUG")
    print(f"="*80)

    # Step 1: Load and align data
    calcium_signals, frame_labels, tracker = debug_load_and_align_data(mat_file_path, xlsx_file_path)

    # Step 2: Create datasets
    datasets, tracker = debug_create_datasets(calcium_signals, frame_labels, tracker, config)

    # Step 3: Extract training data for each signal type
    training_data = {}
    for signal_type in ['calcium_signal', 'deltaf_signal', 'deconv_signal']:
        if signal_type in datasets:
            print(f"\n🎯 Processing {signal_type}...")
            X_train, y_train, X_val, y_val, X_test, y_test, tracker = debug_model_training_data(
                datasets, signal_type, tracker
            )
            training_data[signal_type] = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }

    # Final contamination check across all signal types
    print(f"\n🏁 FINAL CONTAMINATION ANALYSIS")
    print(f"="*50)

    # Compare training data across signal types
    if len(training_data) >= 2:
        signal_types = list(training_data.keys())
        for i in range(len(signal_types)):
            for j in range(i + 1, len(signal_types)):
                sig1, sig2 = signal_types[i], signal_types[j]

                # Compare training data
                X1 = training_data[sig1]['X_train']
                X2 = training_data[sig2]['X_train']

                hash1 = hashlib.md5(X1.numpy().tobytes()).hexdigest()
                hash2 = hashlib.md5(X2.numpy().tobytes()).hexdigest()

                print(f"\n🔍 FINAL CHECK: {sig1} vs {sig2}")
                print(f"  X_train shapes: {X1.shape} vs {X2.shape}")
                print(f"  X_train means: {X1.mean():.8f} vs {X2.mean():.8f}")
                print(f"  X_train hashes: {hash1[:16]}... vs {hash2[:16]}...")

                if hash1 == hash2:
                    print(f"  🚨 TRAINING DATA IS IDENTICAL!")
                    print(f"     This explains your identical model results!")
                else:
                    print(f"  ✅ Training data is properly different")

    return tracker

# Example usage - add this to a separate script:
if __name__ == "__main__":
    from mind.config import get_config

    config = get_config()
    mat_file_path = config["data"]["mat_file"]
    xlsx_file_path = config["data"]["xlsx_file"]

    tracker = debug_full_pipeline(mat_file_path, xlsx_file_path, config)


