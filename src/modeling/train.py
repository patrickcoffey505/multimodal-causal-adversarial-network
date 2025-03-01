from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from multimodal_causal_adversarial_network.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

import tensorflow as tf
import numpy as np
from src.mcan_architecture import MCAN


class MCANTrainer:
    def __init__(self, model, alpha=1.0, beta=1.0, gamma=1.0):
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def discriminator_loss(self, real, generated):
        real_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real), real))
        fake_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(tf.zeros_like(generated), generated)
        )
        return real_loss + fake_loss

    @tf.function
    def train_step(self, x_f, x_e):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate outputs
            outputs = self.model([x_f, x_e], training=True)
            x_f_hat = outputs["x_f_hat"]
            x_e_hat = outputs["x_e_hat"]

            # Discriminator outputs
            d_fmri_real = self.model.fmri_discriminator(x_f)
            d_fmri_fake = self.model.fmri_discriminator(x_f_hat)

            d_eeg_real = self.model.eeg_discriminator(x_e)
            d_eeg_fake = self.model.eeg_discriminator(x_e_hat)

            # Calculate losses
            fmri_disc_loss = self.discriminator_loss(d_fmri_real, d_fmri_fake)
            eeg_disc_loss = self.discriminator_loss(d_eeg_real, d_eeg_fake)
            global_disc_loss = self.discriminator_loss(
                self.model.global_discriminator(outputs["s_hat"]),
                self.model.global_discriminator(tf.concat([x_f, x_e], axis=-1)),
            )

            # Total loss
            total_loss = (
                self.alpha * fmri_disc_loss
                + self.beta * global_disc_loss
                + self.gamma * eeg_disc_loss
            )

        # Calculate gradients
        generator_gradients = gen_tape.gradient(total_loss, self.model.trainable_variables)
        discriminator_gradients = disc_tape.gradient(
            total_loss,
            [
                self.model.fmri_discriminator.trainable_variables,
                self.model.eeg_discriminator.trainable_variables,
                self.model.global_discriminator.trainable_variables,
            ],
        )

        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.model.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients[0], self.model.fmri_discriminator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients[1], self.model.eeg_discriminator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients[2], self.model.global_discriminator.trainable_variables)
        )

        return total_loss

    def train(self, train_dataset, epochs):
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for x_f_batch, x_e_batch in train_dataset:
                loss = self.train_step(x_f_batch, x_e_batch)
                total_loss += loss
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
