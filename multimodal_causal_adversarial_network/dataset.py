from pathlib import Path
import numpy as np
import scipy.io as sio
from loguru import logger
import typer
import tensorflow as tf
from typing import Tuple, List, Dict
from scipy.signal import resample

from multimodal_causal_adversarial_network.config import DATA_DIR

app = typer.Typer()


def resample_data(data: Dict, target_sampling_rate: float) -> Dict:
    """
    Resample data to a new sampling rate.
    """
    original_sampling_rate = 1 / data["fs"][0][0]
    resampling_factor = original_sampling_rate / target_sampling_rate

    # Store original data
    data["fullySampledEEG"] = data["EEG"].copy()
    data["fullySampledfMRI"] = data["fMRI"].copy()

    # Resample both modalities
    data["EEG"] = resample(data["EEG"], num=int(data["EEG"].shape[1] * resampling_factor), axis=1)
    data["fMRI"] = resample(
        data["fMRI"], num=int(data["fMRI"].shape[1] * resampling_factor), axis=1
    )

    return data


def extract_trials(
    data: Dict, trial_period: Tuple[float, float], target_sampling_rate: float
) -> List[Dict[str, np.ndarray]]:
    """Extract individual trials from a run's data."""
    trials = []
    offset_indices = (
        round(trial_period[0] / target_sampling_rate),
        round(trial_period[1] / target_sampling_rate),
    )

    stim_onsets = data["stimOnsetTime"].flatten()

    for i, onset in enumerate(stim_onsets):
        start = round(onset / target_sampling_rate)
        
        # Extract trial data
        fmri_trial = data["fMRI"][:, start + offset_indices[0] : start + offset_indices[1]]
        eeg_trial = data["EEG"][:, start + offset_indices[0] : start + offset_indices[1]]
        
        # Validate trial shapes
        if fmri_trial.shape[1] == 0 or eeg_trial.shape[1] == 0:
            logger.warning(
                f"Trial {i} has empty time dimension:\n"
                f"fMRI shape: {fmri_trial.shape}\n"
                f"EEG shape: {eeg_trial.shape}\n"
                f"Onset time: {onset}\n"
                f"Start index: {start}\n"
                f"Offset indices: {offset_indices}"
            )
            continue
        
        # Add valid trial
        trial = {
            "fmri": fmri_trial,
            "eeg": eeg_trial,
            "stim_onset": onset,
        }
        trials.append(trial)

    return trials


def load_and_preprocess_run(
    file_path: Path,
    target_sampling_rate: float = 0.1,
    trial_period: Tuple[float, float] = (-0.20, 1.00),
) -> List[Dict[str, np.ndarray]]:
    """Load MATLAB file and extract trials."""
    data = sio.loadmat(file_path)
    data = data["data"]
    data = {n: data[n][0, 0] for n in data.dtype.names}

    data = resample_data(data, target_sampling_rate)
    trials = extract_trials(data, trial_period, target_sampling_rate)

    return trials


class BrainDataset:
    def __init__(self, data_dir: Path, shuffle: bool = True, seed: int = 42):
        self.data_dir = data_dir
        self.shuffle = shuffle
        np.random.seed(seed)

        # Get all run files
        self.file_paths = list(data_dir.glob("**/[!.]*.mat"))
        if not self.file_paths:
            raise ValueError(f"No MATLAB files found in {data_dir}")

        # Extract all trials from all runs
        self.trials = []
        invalid_trials = 0
        for file_path in self.file_paths:
            try:
                run_trials = load_and_preprocess_run(file_path)
            
                # Validate trial shapes before adding
                for trial in run_trials:
                    if (trial["fmri"].shape[1] == 0 or 
                        trial["eeg"].shape[1] == 0):
                        invalid_trials += 1
                        continue
                    self.trials.append(trial)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {str(e)}")

        logger.info(
            f"Found {len(self.trials)} valid trials across {len(self.file_paths)} runs\n"
            f"Skipped {invalid_trials} invalid trials"
        )

    def _generator(self):
        """Generator function that yields individual trials"""
        while True:
            if self.shuffle:
                np.random.shuffle(self.trials)

            for trial in self.trials:
                yield trial["fmri"], trial["eeg"]

    def get_dataset(self) -> tf.data.Dataset:
        """Create a tf.data.Dataset that yields individual trials"""
        trial = self.trials[0]

        output_signature = (
            tf.TensorSpec(
                shape=(trial["fmri"].shape[0], trial["fmri"].shape[1]),  # (regions, time)
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(trial["eeg"].shape[0], trial["eeg"].shape[1]),  # (channels, time)
                dtype=tf.float32,
            ),
        )

        dataset = tf.data.Dataset.from_generator(
            self._generator, output_signature=output_signature
        )

        return dataset


@app.command()
def main(data_dir: Path = DATA_DIR):
    """Test data loading pipeline"""
    logger.info("Testing data loading pipeline...")

    try:
        brain_dataset = BrainDataset(data_dir=data_dir, shuffle=True)
        dataset = brain_dataset.get_dataset()

        logger.info("Testing trial iteration...")
        for trial_idx, (fmri, eeg) in enumerate(dataset.take(2)):
            logger.info(f"Trial {trial_idx + 1}:")
            logger.info(f"fMRI shape: {fmri.shape} (regions, time)")
            logger.info(f"EEG shape: {eeg.shape} (channels, time)")

        logger.success("Data pipeline test successful!")

    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")


if __name__ == "__main__":
    app()
