from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from multimodal_causal_adversarial_network.config import FIGURES_DIR, DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


def plot_fmri_time_series(fmri_data: np.ndarray, title: str = "fMRI Time Series by Region"):
    """
    Plot fMRI time series data for multiple regions.
    
    Args:
        fmri_data: Array of shape (time_points, num_regions)
        title: Title for the plot
    """
    num_regions = fmri_data.shape[1]
    time_points = fmri_data.shape[0]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create time axis
    time = np.arange(time_points)
    
    # Plot each region's time series
    for region in range(num_regions):
        ax.plot(time, fmri_data[:, region], label=f'Region {region + 1}')
    
    # Customize plot
    ax.set_xlabel('Time Points')
    ax.set_ylabel('fMRI Signal')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig, ax


if __name__ == "__main__":
    app()
