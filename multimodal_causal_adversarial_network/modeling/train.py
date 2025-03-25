from pathlib import Path

import typer
from loguru import logger
import tensorflow as tf
import numpy as np

from multimodal_causal_adversarial_network.config import MODELS_DIR, PROCESSED_DATA_DIR
from multimodal_causal_adversarial_network.dataset import BrainDataset
from multimodal_causal_adversarial_network.modeling.mcan_architecture import MCAN

app = typer.Typer()


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
        """Calculate discriminator loss for a single trial"""
        real_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(real), real, from_logits=False
        )
        fake_loss = tf.keras.losses.binary_crossentropy(
            tf.zeros_like(generated), generated, from_logits=False
        )
        return real_loss + fake_loss

    @tf.function
    def train_step(self, x_f, x_e):
        """Train step for a single trial"""
        # Add batch dimension for model
        x_f = tf.expand_dims(x_f, 0)
        x_e = tf.expand_dims(x_e, 0)

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

    def train(self, dataset, epochs):
        """Train the model for specified number of epochs"""
        for epoch in range(epochs):
            total_loss = 0
            num_trials = 0

            # Iterate through individual trials
            for x_f, x_e in dataset:
                loss = self.train_step(x_f, x_e)
                total_loss += loss
                num_trials += 1

                if num_trials % 100 == 0:
                    logger.info(f"Epoch {epoch + 1}, Trial {num_trials}, Loss: {loss:.4f}")

            avg_loss = total_loss / num_trials
            logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")


@app.command()
def main(
    data_dir: Path = PROCESSED_DATA_DIR,
    model_save_path: Path = MODELS_DIR / "mcan_model",
    epochs: int = 10,
):
    """Train the MCAN model"""
    logger.info("Starting model training...")

    try:
        # Load dataset
        brain_dataset = BrainDataset(data_dir=data_dir, shuffle=True)
        dataset = brain_dataset.get_dataset()

        # Get shapes from first trial
        first_trial = next(iter(dataset))
        num_regions = first_trial[0].shape[0]
        num_timepoints_f = first_trial[0].shape[1]
        num_timepoints_e = first_trial[1].shape[1]

        # Initialize model
        model = MCAN(
            num_regions=num_regions,
            num_timepoints_f=num_timepoints_f,
            num_timepoints_e=num_timepoints_e,
        )

        # Initialize trainer
        trainer = MCANTrainer(model)

        # Train model
        logger.info("Training model...")
        trainer.train(dataset, epochs)

        # Save model
        model.save(model_save_path)
        logger.success(f"Model saved to {model_save_path}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")


if __name__ == "__main__":
    app()
