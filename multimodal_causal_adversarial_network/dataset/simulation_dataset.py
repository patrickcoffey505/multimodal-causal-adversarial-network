from pathlib import Path
import numpy as np
import scipy.io as sio
from loguru import logger
import typer
import tensorflow as tf
from typing import Tuple, List, Dict
from scipy.signal import resample, convolve, spectrogram
from nilearn.glm.first_level import spm_hrf
import networkx as nx
import matplotlib.pyplot as plt
from multimodal_causal_adversarial_network.config import DATA_DIR
from multimodal_causal_adversarial_network.utils.causal_graph import create_causal_graph, plot_causal_graph
from scipy.signal import butter, filtfilt

app = typer.Typer()

def generate_directed_connectivity_matrix(n_regions: int) -> np.ndarray:
    """Generate a directed connectivity matrix with random values."""
    return NotImplemented

import numpy as np
from typing import List, Optional

def generate_neural_trials_effective_connectivity(
    A: np.ndarray,
    n_trials: int,
    n_timepoints: int,
    dt: float = 0.01,
    input_scale: float = 0.2,
    noise_scale: float = 0.01,
    use_nonlinearity: bool = True,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Simulates neural activity over time using a directed effective connectivity matrix A.
    
    Args:
        A: (n_regions x n_regions) effective connectivity matrix. Should be stable (e.g. diag(A) < 0).
        n_trials: Number of trials to simulate.
        n_timepoints: Number of timepoints per trial.
        dt: Time step for Euler integration.
        input_scale: Magnitude of external input noise.
        noise_scale: Magnitude of internal system noise.
        use_nonlinearity: Whether to apply tanh nonlinearity on x before applying A.
        seed: Optional random seed for reproducibility.
    
    Returns:
        List of np.ndarray, each of shape (n_timepoints, n_regions).
    """
    if seed is not None:
        np.random.seed(seed)
        
    n_regions = A.shape[0]
    B = np.eye(n_regions)  # Input to each region
    trials = []

    for _ in range(n_trials):
        x = np.zeros((n_timepoints, n_regions))
        x[0] = np.random.randn(n_regions) * 0.1  # small random initial state
        u = np.random.randn(n_timepoints, n_regions) * input_scale + 0.01  # tonic input

        for t in range(1, n_timepoints):
            x_prev = np.tanh(x[t-1]) if use_nonlinearity else x[t-1]
            dx = A @ x_prev + B @ u[t-1] + np.random.randn(n_regions) * noise_scale
            x[t] = x[t-1] + dt * dx

        trials.append(x)

    return trials

def generate_neural_trials(
        A: np.ndarray,
        n_trials: int,
        n_regions: int,
        n_timepoints: int,
        dt: float
    ) -> List[np.ndarray]:
    """Generate a neural trial from a directed connectivity matrix."""
    neural_trials = []
    B = np.eye(n_regions)  # Input directly to each region

    for _ in range(n_trials):
        x = np.zeros((n_timepoints, n_regions))
        x[0] = np.random.randn(n_regions) * 0.01 # random initial state

        u = np.random.randn(n_timepoints, n_regions) * 0.005  # noise input

        # --- Simulate neural state: x[t+1] = x[t] + dt * (Ax[t] + Bu[t]) ---
        for t in range(1, n_timepoints):
            dx = A @ x[t-1] + B @ u[t-1] + np.random.randn(n_regions) * 0.005
            x[t] = x[t-1] + dt * dx  # Euler integration
        
        neural_trials.append(x)

    return neural_trials

import numpy as np
from typing import List, Optional

def generate_neural_trials_with_stimulus(
    A: np.ndarray,
    n_trials: int,
    n_timepoints: int,
    start_regions: List[int],
    stimulus_amplitude: float = 1.0,
    stimulus_duration: int = 10,
    dt: float = 0.01,
    input_scale: float = 0.05,
    noise_scale: float = 0.01,
    use_nonlinearity: bool = True,
    return_spikes: bool = False,
    seed: Optional[int] = None
) -> List[np.ndarray]:
    """
    Simulates neural activity over time using a directed effective connectivity matrix A.
    Stimulus is applied to selected 'start_regions' at the beginning of each trial.
    
    Args:
        A: (n_regions x n_regions) effective connectivity matrix.
        n_trials: Number of trials to simulate.
        n_timepoints: Number of timepoints per trial.
        start_regions: List of region indices to receive stimulus input.
        stimulus_amplitude: Strength of the stimulus.
        stimulus_duration: Number of timepoints stimulus is active (from t=0).
        dt: Time step for Euler integration.
        input_scale: Magnitude of external noise input.
        noise_scale: Magnitude of internal noise.
        use_nonlinearity: Apply tanh nonlinearity to state before passing through A.
        return_spikes: If True, return binary spike trains instead of continuous traces.
        seed: Optional random seed for reproducibility.
    
    Returns:
        List of np.ndarray (n_timepoints x n_regions), either continuous or binary.
    """
    if seed is not None:
        np.random.seed(seed)
        
    n_regions = A.shape[0]
    B = np.eye(n_regions)
    trials = []

    for _ in range(n_trials):
        x = np.zeros((n_timepoints, n_regions))
        output = np.zeros_like(x)  # spikes or continuous signal
        x[0] = np.random.randn(n_regions) * 0.1
        u = np.random.randn(n_timepoints, n_regions) * input_scale

        # Add stimulus to start regions during stimulus_duration
        for t in range(stimulus_duration):
            u[t, start_regions] += stimulus_amplitude

        for t in range(1, n_timepoints):
            x_prev = np.tanh(x[t-1]) if use_nonlinearity else x[t-1]
            dx = A @ x_prev + B @ u[t-1] + np.random.randn(n_regions) * noise_scale
            x[t] = x[t-1] + dt * dx

            if return_spikes:
                # Convert to firing rate in [0,1]
                rate = (np.tanh(x[t]) + 1) / 2
                output[t] = (np.random.rand(n_regions) < rate).astype(int)
            else:
                output[t] = x[t]

        trials.append(output)

    return trials


def generate_fmri_trials(
        neural_trials: List[np.ndarray],
        n_regions: int,
        fmri_timepoints: int,
        fmri_timestep: float
    ) -> List[np.ndarray]:
    """Generate a fmri trial from a directed connectivity matrix."""
    hrf = spm_hrf(fmri_timestep, oversampling=1, time_length=32.0, onset=0.0) # HRF window defined over 32 second range
    fmri_trials = []

    for neural_trial in neural_trials:
        bold = np.zeros((fmri_timepoints, n_regions))

        # downsample neural trial to fmri timestep
        neural_trial_ds = resample(neural_trial, num=int(fmri_timepoints), axis=0)
        for region in range(n_regions):
            bold[:, region] = convolve(neural_trial_ds[:, region], hrf, mode='full')[:fmri_timepoints]
        fmri_trials.append(bold)
    
    return fmri_trials

def generate_eeg_trials(
        neural_trials: List[np.ndarray],
        eeg_channels: int,
        noise_level: float = 0.1,
        eeg_band: Tuple[float, float] = (1, 40),
        sfreq: int = 250,
        seed: int = 42
) -> List[np.ndarray]:
    """
    Simulate EEG data from neural source activity via a linear observation model with noise and bandpass filtering.

    Parameters:
    - neural_trials: List[np.ndarray] of shape (n_trials, n_timepoints, n_sources)
        Simulated neural source activity.
    - eeg_channels: int
        Number of EEG channels (nodes).
    - noise_level: float
        Standard deviation of added Gaussian noise.
    - eeg_band: tuple of float
        Bandpass filter range in Hz (e.g., (1, 40)).
    - sfreq: int
        Sampling frequency (Hz).
    - seed: int or None
        Random seed for reproducibility.

    Returns:
    - eeg_trials: List[np.ndarray] of shape (n_trials, eeg_channels, n_timepoints)
        Simulated EEG data.
    """
    rng = np.random.default_rng(seed)

    eeg_trials = []
    for neural_trial in neural_trials:
        # Transpose neural trial to get (n_sources, n_timepoints)
        neural_trial = neural_trial.T  # Shape: (n_sources, n_timepoints)
        n_sources, n_timepoints = neural_trial.shape

        # Random linear mixing: EEG = L @ sources
        L = rng.standard_normal((eeg_channels, n_sources)) / np.sqrt(n_sources)  # Normalize scale

        eeg_data = L @ neural_trial  # Shape: (eeg_channels, n_timepoints)

        # Add spatially and temporally uncorrelated Gaussian noise
        noise = rng.normal(scale=noise_level, size=eeg_data.shape)
        eeg_data += noise

        # Bandpass filter to simulate EEG frequency characteristics
        if eeg_band is not None:
            low, high = eeg_band
            if low <= 0 or high >= sfreq / 2:
                raise ValueError("eeg_band must be within (0, Nyquist frequency)")
            
            # Calculate filter order and minimum signal length
            filter_order = 4
            min_length = 3 * (filter_order + 1)
            
            # If signal is too short, pad it with zeros
            if eeg_data.shape[1] < min_length:
                pad_length = min_length - eeg_data.shape[1]
                eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_length)), mode='constant')
            
            b, a = butter(filter_order, [low / (sfreq / 2), high / (sfreq / 2)], btype='band')
            eeg_data = filtfilt(b, a, eeg_data, axis=1)
            
            # Remove padding if it was added
            if eeg_data.shape[1] > n_timepoints:
                eeg_data = eeg_data[:, :n_timepoints]
        
        eeg_trials.append(eeg_data)

    return eeg_trials

def extract_band_power_time_series(
    eeg_trials: List[np.ndarray],
    sfreq,
    freq_band=(20, 40),
    nperseg=256,
    noverlap=128,
    window='hann',
    inference_timepoints: int = None
):
    """
    Extracts downsampled EEG features by computing total power in a frequency band over time using spectrogram.

    Parameters:
    - eeg_trials: List[np.ndarray], shape (n_trials, n_channels, n_times)
        EEG signals.
    - sfreq: float
        Sampling frequency (Hz).
    - freq_band: tuple of (fmin, fmax)
        Frequency band to extract power from (e.g., alpha: (8, 12)).
    - nperseg: int
        Length of each spectrogram segment.
    - noverlap: int
        Number of points to overlap between segments.
    - window: str
        Window type to use for STFT (e.g., 'hann').
    - inference_timepoints: int
        Number of timepoints to resample the power series to.

    Returns:
    - band_power: np.ndarray, shape (n_trials, n_channels, n_timepoints)
        Total power in the given band over time for each channel.
    - times: np.ndarray
        Time vector (center of each spectrogram window).
    """
    band_power = []
    times = None

    for eeg_trial in eeg_trials:
        # eeg_trial shape: (n_channels, n_times)
        trial_power = []
        
        for channel in range(eeg_trial.shape[0]):
            f, t, Sxx = spectrogram(
                eeg_trial[channel],
                fs=sfreq,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density',
                mode='psd'  # power spectral density
            )
            
            # Find frequency indices in the desired band
            freq_mask = (f >= freq_band[0]) & (f <= freq_band[1])
            
            # Sum power across those frequencies for each time window
            power_in_band = Sxx[freq_mask, :].sum(axis=0)
            
            # Resample to inference timepoints if specified
            if inference_timepoints is not None:
                power_in_band = resample(power_in_band, inference_timepoints)
            
            trial_power.append(power_in_band)
            
            if times is None:
                if inference_timepoints is not None:
                    times = np.linspace(t[0], t[-1], inference_timepoints)
                else:
                    times = t
        
        band_power.append(np.array(trial_power))

    # Transpose each trial from (n_channels, n_timepoints) → (n_timepoints, n_channels)
    band_power = [trial.T for trial in band_power]  # List of (time, channels)
    print(np.stack(band_power, axis=0).shape)
    return band_power


def upsample_fmri(fmri_trials: List[np.ndarray], inference_timepoints: int) -> List[np.ndarray]:
    """Upsample fmri trials to inference time step."""
    upsampled_fmri_trials = []
    for fmri_trial in fmri_trials:
        upsampled_fmri_trial = resample(fmri_trial, num=inference_timepoints, axis=0)
        upsampled_fmri_trials.append(upsampled_fmri_trial)
    return upsampled_fmri_trials

class SimulationDataset:
    def __init__(
            self,
            directed_connectivity_matrix: np.ndarray = None,
            start_regions: List[int] = None,
            n_trials: int = 200,
            n_regions: int = 5,
            eeg_channels: int = 3,
            trial_time: int = 5,
            inference_time_step: float = 0.2,
            fmri_timestep: float = 1.0,
            eeg_timestep: float = 0.01,
            fmri_only: bool = False,
            shuffle: bool = True,
            seed: int = 42
    ):
        if inference_time_step < eeg_timestep or inference_time_step > fmri_timestep:
            raise ValueError("Inference time step must be in between eeg and fmri timesteps")
        self.inference_time_step = inference_time_step
        self.inference_timepoints = int(trial_time / inference_time_step)

        self.A = None
        # Generate directed connectivity matrix
        if directed_connectivity_matrix is None:
            self.A, self.start_regions = generate_directed_connectivity_matrix(n_regions)
        else:
            self.A = directed_connectivity_matrix
            self.start_regions = start_regions
        
        self.n_trials = n_trials
        self.n_regions = n_regions
        self.trial_time = trial_time

        logger.info(f"Generating neural trials...")

        # Generate neural trials
        self.neural_timestep = eeg_timestep
        self.neural_timepoints = int(trial_time / eeg_timestep)
        self.neural_trials = generate_neural_trials_with_stimulus(
            A=self.A,
            start_regions=self.start_regions,
            n_trials=self.n_trials,
            n_timepoints=self.neural_timepoints,
            stimulus_amplitude=1.5,
            stimulus_duration=trial_time,
            dt=self.neural_timestep,
            input_scale=0.1,
            noise_scale=0.01,
            use_nonlinearity=True,
            return_spikes=False,
            seed=seed
        )
        self.neural_trials = generate_neural_trials(self.A, n_trials, n_regions, self.neural_timepoints, self.neural_timestep)

        logger.info(f"Generating fmri trials...")

        # Generate fmri trials
        self.fmri_timestep = fmri_timestep
        self.fmri_timepoints = int(trial_time / fmri_timestep)
        self.fmri_trials = generate_fmri_trials(self.neural_trials, n_regions, self.fmri_timepoints, self.fmri_timestep)

        logger.info(f"Generating eeg trials...")

        # Generate eeg trials
        self.fmri_only = fmri_only
        self.eeg_timestep = eeg_timestep
        self.eeg_timepoints = int(trial_time / eeg_timestep)
        self.eeg_channels = eeg_channels
        if self.fmri_only:
            self.eeg_trials = [tf.constant([]) for _ in range(self.n_trials)]
        else:
            self.eeg_trials = generate_eeg_trials(
                self.neural_trials,
                self.eeg_channels,
                noise_level=0.1,
                eeg_band=(20, 40),
                sfreq=250,
                seed=42
            )
            self.power_trials = extract_band_power_time_series(
                self.eeg_trials, 
                250, 
                (20, 40),
                inference_timepoints=self.inference_timepoints
            )
            
        neural_trials_stacked = np.stack(self.neural_trials)
        self.neural_mean = np.mean(neural_trials_stacked)
        self.neural_std = np.std(neural_trials_stacked)

        fmri_trials_stacked = np.stack(self.fmri_trials)
        self.fmri_mean = np.mean(fmri_trials_stacked)
        self.fmri_std = np.std(fmri_trials_stacked)

        if not self.fmri_only:
            power_trials_stacked = np.stack(self.power_trials)
            self.power_mean = np.mean(power_trials_stacked)
            self.power_std = np.std(power_trials_stacked)

        self.shuffle = shuffle
        np.random.seed(seed)

    def _generator(self):
        """Generator function that yields individual trials"""
        while True:
            if self.shuffle:
                trial_idxs = np.random.shuffle(range(self.n_trials))

            for trial_idx in trial_idxs:
                yield self.fmri_trials[trial_idx], self.power_trials[trial_idx]

    def get_dataset(self) -> tf.data.Dataset:
        """Create a tf.data.Dataset that yields individual trials"""
        # Instead of using a fixed output signature, we'll create a dataset that
        # can handle variable shapes
        def _generator_wrapper():
            for trial_idx in range(self.n_trials):
                yield (
                    tf.convert_to_tensor(self.fmri_trials[trial_idx], dtype=tf.float32),
                    tf.convert_to_tensor(self.power_trials[trial_idx], dtype=tf.float32),
                )

        # Create a dataset with variable shapes
        output_signature = (
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # fmri
            tf.TensorSpec(shape=(0), dtype=tf.float32) if self.fmri_only else tf.TensorSpec(shape=(None, None), dtype=tf.float32),  # eeg
        )

        dataset = tf.data.Dataset.from_generator(
            _generator_wrapper,
            output_signature=output_signature
        )
        
        return dataset
    
    def plot_causal_graph(self, threshold=0):
        """Plot the causal graph."""
        G = create_causal_graph(tf.expand_dims(self.A, axis=0), threshold)
        plot_causal_graph(G)

    def plot_neural_source(self, trial_idx: int = 0):
        """Plot the neural source."""
        trial = self.neural_trials[trial_idx]
        plt.figure(figsize=(12, 6))
        for i in range(self.n_regions):
            time = np.arange(0, self.neural_timepoints * self.neural_timestep, self.neural_timestep)
            plt.plot(time, trial[:, i] / np.max(abs(trial[:, i])), label=f'Region {i+1}')
        plt.legend()
        plt.show()
    
    def plot_eeg(self, trial_idx: int = 0):
        """Plot the eeg."""
        trial = self.eeg_trials[trial_idx]
        time = np.arange(0, self.eeg_timepoints * self.eeg_timestep, self.eeg_timestep)
        plt.figure(figsize=(12, 6))
        for i in range(self.eeg_channels):  
            plt.plot(time, trial[i] / np.max(abs(trial[i])) + i * 2, label=f'Channel {i+1}')
        plt.legend()
        plt.title('Simulated EEG signals (shifted vertically and resampled)')
        plt.xlabel('Time (s)')
        plt.ylabel('EEG signal (a.u.)')
        plt.show()

    def plot_power(self, trial_idx: int = 0):
        """Plot the power."""
        trial = self.power_trials[trial_idx]
        time = np.arange(0, self.inference_timepoints * self.inference_time_step, self.inference_time_step)
        plt.figure(figsize=(12, 6))
        for i in range(self.eeg_channels):
            plt.plot(time, trial[:, i] / np.max(abs(trial[:, i])) + i * 2, label=f'Channel {i+1}')
        plt.legend()
        plt.title('Simulated EEG power (shifted vertically and resampled)')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (a.u.)')
        plt.show()

    def plot_bold(self, trial_idx: int = 0):
        """Plot the bold signal."""
        trial = self.fmri_trials[trial_idx]
        time = np.arange(0, self.fmri_timepoints * self.fmri_timestep, self.fmri_timestep)
        # --- 6. Plotting ---
        plt.figure(figsize=(12, 6))
        for i in range(self.n_regions):
            y = trial[:, i] / np.max(abs(trial[:, i])) + i * 2
            plt.plot(time, y, label=f'Region {i+1}')
        plt.title('Simulated BOLD signals (shifted vertically and resampled)')
        plt.xlabel('Time (s)')
        plt.ylabel('BOLD signal (a.u.)')
        plt.legend()
        plt.tight_layout()
        plt.show()


@app.command()
def main():
    """Test simulation dataset pipeline"""
    logger.info("Simulation dataset pipeline...")

    try:
        A = np.array([
            [-1.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.4, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.2],
            [0.1, 0.0, 0.0, 0.0, -1.0]
        ])  # Directed loop
        sim_dataset = SimulationDataset(A, fmri_only=True)
        dataset = sim_dataset.get_dataset()

        logger.info("Testing trial iteration...")
        for trial_idx, (fmri, eeg) in enumerate(dataset.take(2)):
            logger.info(f"Trial {trial_idx + 1}:")
            logger.info(f"fMRI shape: {fmri.shape} (time, regions)")
            # logger.info(f"EEG shape: {eeg.shape} (channels, time)")

        logger.success("Data pipeline test successful!")

    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")


if __name__ == "__main__":
    app()
