from pathlib import Path

import typer
from loguru import logger
import tensorflow as tf
import numpy as np
from multimodal_causal_adversarial_network.config import MODELS_DIR
from multimodal_causal_adversarial_network.dataset.simulation_dataset import SimulationDataset
from multimodal_causal_adversarial_network.modeling.mcan_architecture import MCANGenerator, Discriminator
from multimodal_causal_adversarial_network.utils.causal_graph import create_causal_graph, plot_causal_graph
from multimodal_causal_adversarial_network.utils.min_max_scaling import scale_to_unit_range
from multimodal_causal_adversarial_network.utils.state_estimation import StateEstimation

app = typer.Typer()


class MCANTrainer:
    def __init__(self, generator, alpha=1.0, beta=1.0, gamma=1.0):
        self.generator_model = generator
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Discriminators
        self.fmri_discriminator = Discriminator()
        self.eeg_discriminator = Discriminator()
        self.global_discriminator = Discriminator()

        # Optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.fmri_discriminator_optimizer = tf.keras.optimizers.Adam(5e-5)
        self.eeg_discriminator_optimizer = tf.keras.optimizers.Adam(5e-5)
        self.global_discriminator_optimizer = tf.keras.optimizers.Adam(5e-5)

        # State estimation
        self.G_mm = None
        self.state_estimation = StateEstimation()

    def discriminator_loss(self, real, generated):
        """Calculate discriminator loss for a single trial"""
        real_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(real), real, from_logits=False
        )
        fake_loss = tf.keras.losses.binary_crossentropy(
            tf.zeros_like(generated), generated, from_logits=False
        )
        return real_loss + fake_loss, real_loss, fake_loss

    def generator_loss(self, generated, real=None):
        """Calculate generator loss including reconstruction loss"""
        adv_loss = tf.keras.losses.binary_crossentropy(
            tf.ones_like(generated), generated, from_logits=False
        )
        
        if real is not None:
            recon_loss = tf.reduce_mean(tf.square(real - generated))
            return adv_loss + recon_loss, adv_loss, recon_loss
        return adv_loss

    @tf.function
    def train_step(self, x_f, x_e = None):
        """Train step for a single trial"""
        # Add batch dimension for model
        x_f = tf.expand_dims(x_f, 0)
        x_e = tf.expand_dims(x_e, 0) if x_e is not None else None

        fmri_scaled, fmri_scale_params = scale_to_unit_range(x_f)
        if x_e is not None:
            eeg_scaled, eeg_scale_params = scale_to_unit_range(x_e)

        with tf.GradientTape(persistent=True) as tape:
            # Generate outputs
            if x_e is None:
                outputs = self.generator_model((fmri_scaled,), training=True)
            else:
                outputs = self.generator_model((fmri_scaled, eeg_scaled), training=True)

            G_mm = outputs["G_mm"]
            losses = {}
            gradient_norms = {}

            # Discriminator outputs
            d_fmri_real = self.fmri_discriminator(fmri_scaled)
            d_fmri_fake = self.fmri_discriminator(outputs["x_f_post"])

            if x_e is not None:
                d_eeg_real = self.eeg_discriminator(eeg_scaled)
                d_eeg_fake = self.eeg_discriminator(outputs["x_e_post"])
                d_g_fmri = self.global_discriminator(outputs["s_f_post"] @ tf.reduce_mean(G_mm, axis=2))
                d_g_eeg = self.global_discriminator(outputs["s_e_post"] @ tf.reduce_mean(G_mm, axis=2))

            # Calculate losses
            fmri_disc_loss, fmri_disc_real_loss, fmri_disc_fake_loss = self.discriminator_loss(d_fmri_real, d_fmri_fake)
            fmri_gen_loss = self.generator_loss(d_fmri_fake)

            losses["fmri_disc_loss"] = fmri_disc_loss
            losses["fmri_disc_real_loss"] = fmri_disc_real_loss
            losses["fmri_disc_fake_loss"] = fmri_disc_fake_loss
            losses["fmri_gen_loss"] = fmri_gen_loss
            
            if x_e is not None:
                eeg_disc_loss, eeg_disc_real_loss, eeg_disc_fake_loss = self.discriminator_loss(d_eeg_real, d_eeg_fake)
                global_disc_loss, global_disc_real_loss, global_disc_fake_loss = self.discriminator_loss(d_g_fmri, d_g_eeg)
                eeg_gen_loss = self.generator_loss(d_eeg_fake)
                global_gen_loss = self.generator_loss(d_g_eeg)
                losses["eeg_disc_loss"] = eeg_disc_loss
                losses["eeg_disc_real_loss"] = eeg_disc_real_loss
                losses["eeg_disc_fake_loss"] = eeg_disc_fake_loss
                losses["eeg_gen_loss"] = eeg_gen_loss
                losses["global_disc_loss"] = global_disc_loss
                losses["global_disc_real_loss"] = global_disc_real_loss
                losses["global_disc_fake_loss"] = global_disc_fake_loss
                losses["global_gen_loss"] = global_gen_loss

            # Total loss
            if x_e is None:
                disc_loss = fmri_disc_loss
                gen_loss = fmri_gen_loss
            else:
                disc_loss = (
                    self.alpha * fmri_disc_loss
                    + self.beta * global_disc_loss
                    + self.gamma * eeg_disc_loss
                )
                gen_loss = (
                    self.alpha * fmri_gen_loss
                    + self.beta * global_gen_loss
                    + self.gamma * eeg_gen_loss
                )
            losses["disc_loss"] = disc_loss
            losses["gen_loss"] = gen_loss

        # Calculate gradients for generator
        generator_gradients = tape.gradient(gen_loss, self.generator_model.trainable_variables)
        generator_grad_norm = tf.linalg.global_norm(generator_gradients)
        gradient_norms["generator_grad_norm"] = generator_grad_norm
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator_model.trainable_variables))

        # Calculate gradients for discriminators
        fmri_disc_gradients = tape.gradient(fmri_disc_loss, self.fmri_discriminator.trainable_variables)
        fmri_disc_grad_norm = tf.linalg.global_norm(fmri_disc_gradients)
        gradient_norms["fmri_disc_grad_norm"] = fmri_disc_grad_norm
        self.fmri_discriminator_optimizer.apply_gradients(zip(fmri_disc_gradients, self.fmri_discriminator.trainable_variables))
        
        if x_e is not None:
            eeg_disc_gradients = tape.gradient(eeg_disc_loss, self.eeg_discriminator.trainable_variables)
            eeg_disc_grad_norm = tf.linalg.global_norm(eeg_disc_gradients)
            gradient_norms["eeg_disc_grad_norm"] = eeg_disc_grad_norm
            self.eeg_discriminator_optimizer.apply_gradients(zip(eeg_disc_gradients, self.eeg_discriminator.trainable_variables))
            global_disc_gradients = tape.gradient(global_disc_loss, self.global_discriminator.trainable_variables)
            global_disc_grad_norm = tf.linalg.global_norm(global_disc_gradients)
            gradient_norms["global_disc_grad_norm"] = global_disc_grad_norm
            self.global_discriminator_optimizer.apply_gradients(zip(global_disc_gradients, self.global_discriminator.trainable_variables))
        
        del tape  # Delete the persistent tape
        return G_mm, losses, gradient_norms

    def train(self, dataset, fmri_only, epochs, state_estimation=False):
        """Train the model for specified number of epochs"""
        G_mm = None
        # dataset mean and std

        for epoch in range(epochs):
            total_gen_loss = 0
            total_disc_loss = 0
            num_trials = 0

            # Iterate through individual trials
            for x_f, x_e in dataset:
                G_mm, losses, gradient_norms = self.train_step(
                    x_f,
                    None if fmri_only else x_e,
                )
                total_gen_loss += losses["gen_loss"]
                total_disc_loss += losses["disc_loss"]
                num_trials += 1
                self.G_mm = G_mm.numpy()
            
            avg_gen_loss = total_gen_loss / num_trials
            avg_disc_loss = total_disc_loss / num_trials
            logger.info(f"Epoch {epoch + 1}, Average Gen Loss: {float(avg_gen_loss.numpy()):.4f}, Average Disc Loss: {float(avg_disc_loss.numpy()):.4f}")

            g = create_causal_graph(G_mm.numpy(), scaling=True)
            # g1 = create_causal_graph(G_mm.numpy(), scaling=True, threshold=0.6)
            plot_causal_graph(g)
            # plot_causal_graph(g1)

    def save_model(self, path):
        self.generator_model.save(path)

    def load_model(self, path):
        self.generator_model = tf.keras.models.load_model(path)
    
    def evaluate(self, metrics = ['precision', 'recall', 'SHD'], state_estimation=False):
        """Evaluate the model"""
        G_mm = self.G_mm
        # Convert symbolic tensor to numpy array
        G_mm_np = tf.keras.backend.get_value(G_mm)
        pred_G = create_causal_graph(G_mm_np, scaling=True)
        plot_causal_graph(pred_G)

@app.command()
def main(
    model_save_path: Path = MODELS_DIR / "mcan_model",
    epochs: int = 10,
):
    """Train the MCAN model"""
    logger.info("Starting model training...")

    try:
        A = np.array([
            [-1.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.4, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.2],
            [0.1, 0.0, 0.0, 0.0, -1.0]
        ])  # Directed loop
        # Load dataset
        sim = SimulationDataset(A, fmri_only=True)
        dataset = sim.get_dataset()

        # Initialize model
        generator = MCANGenerator(
            n_regions=sim.n_regions,
            inference_timepoints=sim.inference_timepoints,
            fmri_timepoints=sim.fmri_timepoints,
            eeg_nodes=sim.eeg_nodes
        )

        # Build model with example input shapes
        # Get first batch from dataset to determine input shapes
        for x_f, x_e in dataset.take(1):
            # Add batch dimension to shapes
            fmri_shape = (1,) + x_f.shape  # (batch_size, fmri_timepoints, n_regions)
            eeg_shape = (1,) + x_e.shape if x_e is not None else None  # (batch_size, inference_timepoints, eeg_nodes)
            
            # Build model with tuple of input shapes
            if eeg_shape is not None:
                generator.build((fmri_shape, eeg_shape))
            else:
                generator.build((fmri_shape,))
            break

        # Initialize trainer
        trainer = MCANTrainer(generator)

        # Train model
        logger.info("Training model...")
        trainer.train(dataset, fmri_only=True, epochs=epochs)

        # Save model
        generator.save(model_save_path)
        logger.success(f"Model saved to {model_save_path}")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")


if __name__ == "__main__":
    app()
