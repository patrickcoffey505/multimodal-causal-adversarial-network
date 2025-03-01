import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np


class SourceEncoder(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(SourceEncoder, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # Initialize trainable matrices
        self.H = self.add_weight(
            name="bold_response_matrix",
            shape=(input_shape[-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.noise_std = self.add_weight(
            name="noise_std", shape=(1,), initializer=tf.constant_initializer(0.01), trainable=True
        )

    def call(self, inputs, training=None):
        # Add Gaussian noise during training
        if training:
            epsilon = tf.random.normal(tf.shape(inputs)) * self.noise_std
        else:
            epsilon = 0

        # Apply encoding transformation
        encoded = tf.matmul(inputs, self.H) + epsilon
        return encoded


class AttentionFusion(layers.Layer):
    def __init__(self, num_heads=8, key_dim=64, **kwargs):
        super(AttentionFusion, self).__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layernorm = layers.LayerNormalization()

    def call(self, s_f, s_e):
        # Concatenate along temporal dimension
        s_mm = tf.concat([s_f, s_e], axis=-1)

        # Self-attention
        attention_output = self.mha(s_mm, s_mm)

        # Add & Normalize
        return self.layernorm(s_mm + attention_output)


class CausalUnit(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CausalUnit, self).__init__(**kwargs)
        self.gru = layers.GRU(units, return_sequences=True)
        self.dense = layers.Dense(units, activation="relu")

    def call(self, inputs, prev_outputs):
        # Process with GRU
        U = self.gru(inputs)

        # Calculate V using dense layer
        V = self.dense(prev_outputs)

        # Combine U and V
        return tf.nn.relu(U + V)


class DynamicCausalNetwork(layers.Layer):
    def __init__(self, num_regions, hidden_dim, **kwargs):
        super(DynamicCausalNetwork, self).__init__(**kwargs)
        self.num_regions = num_regions
        self.causal_units = [CausalUnit(hidden_dim) for _ in range(num_regions)]
        self.W = self.add_weight(
            name="causality_weights",
            shape=(num_regions, num_regions),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Initialize outputs
        outputs = tf.zeros((batch_size, seq_len, self.num_regions))

        # Process each region
        for i in range(self.num_regions):
            region_input = inputs[:, :, i : i + 1]
            other_outputs = outputs[:, :, :i]

            # Apply causal unit
            region_output = self.causal_units[i](region_input, other_outputs)
            outputs = tf.concat(
                [outputs[:, :, :i], region_output, outputs[:, :, i + 1 :]], axis=-1
            )

        return outputs, self.W


class SignalDecoder(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(SignalDecoder, self).__init__(**kwargs)
        self.dense = layers.Dense(output_dim)

    def call(self, inputs):
        return self.dense(inputs)


class Discriminator(layers.Layer):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.conv1d = layers.Conv1D(64, 3, activation="relu")
        self.pool = layers.GlobalAveragePooling1D()
        self.dense = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.pool(x)
        return self.dense(x)


class MCAN(Model):
    def __init__(self, num_regions, num_timepoints_f, num_timepoints_e, hidden_dim=64):
        super(MCAN, self).__init__()

        # Encoders
        self.fmri_encoder = SourceEncoder(num_timepoints_e)
        self.eeg_encoder = SourceEncoder(num_timepoints_e)

        # Fusion
        self.fusion = AttentionFusion()

        # Dynamic Causal Network
        self.dcn = DynamicCausalNetwork(num_regions, hidden_dim)

        # Decoders
        self.fmri_decoder = SignalDecoder(num_timepoints_f)
        self.eeg_decoder = SignalDecoder(num_timepoints_e)

        # Discriminators
        self.fmri_discriminator = Discriminator()
        self.eeg_discriminator = Discriminator()
        self.global_discriminator = Discriminator()

    def call(self, inputs, training=None):
        x_f, x_e = inputs

        # Encoding
        s_f = self.fmri_encoder(x_f, training=training)
        s_e = self.eeg_encoder(x_e, training=training)

        # Fusion
        s_mm = self.fusion(s_f, s_e)

        # Dynamic Causal Network
        s_hat, W = self.dcn(s_mm, training=training)

        # Decoding
        x_f_hat = self.fmri_decoder(s_hat)
        x_e_hat = self.eeg_decoder(s_hat)

        # Compute attention map
        A = tf.matmul(s_mm, tf.transpose(s_mm, perm=[0, 2, 1]))

        # Compute final causality matrix
        G_mm = A + W

        return {"x_f_hat": x_f_hat, "x_e_hat": x_e_hat, "s_hat": s_hat, "G_mm": G_mm}
