import tensorflow as tf
from tensorflow.keras import layers, Model

class FmriSourceEncoder(layers.Layer):
    def __init__(self, fmri_timepoints, inference_timepoints, **kwargs):
        super(FmriSourceEncoder, self).__init__(**kwargs)
        self.fmri_timepoints = fmri_timepoints
        self.inference_timepoints = inference_timepoints

    def build(self, input_shape):
        self.noise_std = self.add_weight(
            name="noise_std",
            shape=(1,),
            initializer=tf.constant_initializer(0.001),
            trainable=True
        )
        # Shared weights between encoder and decoder
        # Encoder: fmri_timepoints -> inference_timepoints
        self.shared_dense = layers.Dense(self.inference_timepoints, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.2))
        
        # Separate batch normalization layers for encoder and decoder
        self.encoder_bn = layers.BatchNormalization(momentum=0.95)
        self.decoder_bn = layers.BatchNormalization(momentum=0.95)
        
        super(FmriSourceEncoder, self).build(input_shape)
    
    def encode(self, inputs, training=True):
        if training:
            epsilon = tf.random.normal([tf.shape(inputs)[0], self.inference_timepoints, tf.shape(inputs)[2]], mean=0, stddev=self.noise_std)
        else:
            epsilon = 0
        
        inputs = tf.transpose(inputs, perm=[0, 2, 1])
        
        # Encoder: fmri_timepoints -> inference_timepoints
        x = self.shared_dense(inputs)
        x = self.encoder_bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = tf.transpose(x, perm=[0, 2, 1])
        return x + epsilon
    
    def decode(self, inputs, training=True):
        if training:
            epsilon = tf.random.normal([tf.shape(inputs)[0], self.fmri_timepoints, tf.shape(inputs)[2]], mean=0, stddev=self.noise_std)
        else:
            epsilon = 0
        
        inputs = tf.transpose(inputs, perm=[0, 2, 1])
        
        # Decoder: inference_timepoints -> fmri_timepoints
        # Use transpose of encoder weights for inverse operation
        x = tf.matmul(inputs, self.shared_dense.kernel, transpose_b=True)
        x = self.decoder_bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = tf.transpose(x, perm=[0, 2, 1])
        return x + epsilon
    
    def call(self, x, mode="encode", training=True):
        if mode == "encode":
            return self.encode(x, training=training)
        elif mode == "decode":
            return self.decode(x, training=training)
        else:
            raise ValueError("mode must be 'encode' or 'decode'")


class EegSourceEncoder(layers.Layer):
    def __init__(self, eeg_nodes, output_regions, **kwargs):
        super(EegSourceEncoder, self).__init__(**kwargs)
        self.eeg_nodes = eeg_nodes
        self.output_regions = output_regions

    def build(self, input_shape):
        # Shared weights between encoder and decoder
        # Encoder: eeg_nodes -> output_regions
        self.shared_dense = layers.Dense(self.output_regions, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.2))
        
        # Separate batch normalization layers for encoder and decoder
        self.encoder_bn = layers.BatchNormalization(momentum=0.95)
        self.decoder_bn = layers.BatchNormalization(momentum=0.95)
        
        self.noise_std = self.add_weight(
            name="noise_std",
            shape=(1,),
            initializer=tf.constant_initializer(0.001),
            trainable=True
        )
        super(EegSourceEncoder, self).build(input_shape)
    
    def encode(self, inputs, training=None):
        if training:
            epsilon = tf.random.normal([tf.shape(inputs)[0], tf.shape(inputs)[1], self.output_regions], mean=0, stddev=self.noise_std)
        else:
            epsilon = 0
            
        # Encoder: eeg_nodes -> output_regions
        x = self.shared_dense(inputs)
        x = self.encoder_bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x + epsilon
    
    def decode(self, inputs, training=None):
        if training:
            epsilon = tf.random.normal([tf.shape(inputs)[0], tf.shape(inputs)[1], self.eeg_nodes], mean=0, stddev=self.noise_std)
        else:
            epsilon = 0
        
        # Decoder: output_regions -> eeg_nodes
        # Use transpose of encoder weights for inverse operation
        x = tf.matmul(inputs, self.shared_dense.kernel, transpose_b=True)
        x = self.decoder_bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x + epsilon
        
    def call(self, x, mode="encode", training=False):
        if mode == "encode":
            return self.encode(x, training=training)
        elif mode == "decode":
            return self.decode(x, training=training)
        else:
            raise ValueError("mode must be 'encode' or 'decode'")


class AttentionFusion(layers.Layer):
    def __init__(self, num_heads=4, key_dim=5, **kwargs):
        super(AttentionFusion, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )
        self.query_norm = layers.LayerNormalization()
        self.value_norm = layers.LayerNormalization()
        
        # Add dropout for attention scores
        self.attention_dropout = layers.Dropout(0.1)

    def call(self, query, value, training=None):
        # Ensure proper shape for attention
        query = tf.transpose(query, perm=[0, 2, 1])
        value = tf.transpose(value, perm=[0, 2, 1])
        query = self.query_norm(query)
        value = self.value_norm(value)
        
        # Multi-head attention
        attention_output, scores = self.mha(
            query,
            value,
            return_attention_scores=True
        )
        
        # Apply dropout to attention scores during training
        if training:
            scores = self.attention_dropout(scores, training=training)
        
        # Add residual connection and normalize output
        attention_output = query + attention_output
        
        # Transpose back to original shape
        output = tf.transpose(attention_output, perm=[0, 2, 1])
        return output, scores


class DynamicCausalNetwork(layers.Layer):
    def __init__(self, num_regions, num_timepoints, **kwargs):
        super(DynamicCausalNetwork, self).__init__(**kwargs)
        self.num_regions = num_regions
        self.num_timepoints = num_timepoints
        
        # Use He initialization for GRU cells with full capacity
        self.gru_cell1 = layers.GRUCell(
            num_regions,
            kernel_initializer='he_normal',
            recurrent_initializer='orthogonal'
        )
        self.gru_cell2 = layers.GRUCell(
            num_regions,
            kernel_initializer='he_normal',
            recurrent_initializer='orthogonal'
        )
        
        # Replace LayerNorm with BatchNorm for better gradient flow
        self.gru_cell1_norm = layers.BatchNormalization(momentum=0.95)
        self.gru_cell2_norm = layers.BatchNormalization(momentum=0.95)
        self.v_norm = layers.BatchNormalization(momentum=0.95)
        self.output_norm = layers.BatchNormalization(momentum=0.95)
        
        # Add dropout for W to help escape local minima
        self.w_dropout = layers.Dropout(0.1)
        
    def build(self, input_shape):
        # Initialize weights with orthogonal initialization for better gradient flow
        # Remove regularization from W to make it more sensitive to updates
        self.W = self.add_weight(
            name="causality_weights",
            shape=(self.num_regions, self.num_regions, self.num_timepoints),
            initializer=tf.keras.initializers.Orthogonal(),
            trainable=True,
        )
        self.time_indices = tf.range(0, self.num_timepoints)
        super(DynamicCausalNetwork, self).build(input_shape)

    def step(self, prev_output, current_input_and_time):
        current_input, t = current_input_and_time
        W_t = self.W[:,:,t]
        
        # First GRU layer
        U1, _ = self.gru_cell1(current_input, prev_output)
        U1 = self.gru_cell1_norm(U1, training=True)
        U1 = tf.nn.leaky_relu(U1, alpha=0.2)
        
        # Second GRU layer
        U2, _ = self.gru_cell2(U1, prev_output)
        U2 = self.gru_cell2_norm(U2, training=True)
        U2 = tf.nn.leaky_relu(U2, alpha=0.2)
        
        # Causal weights path with dropout
        V = tf.matmul(prev_output, W_t)
        V = self.v_norm(V, training=True)
        V = tf.nn.leaky_relu(V, alpha=0.2)
        
        # Combine with residual connection
        combined = U2 + V
        combined = self.output_norm(combined, training=True)
        return combined

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]

        inputs_transposed = tf.transpose(inputs, perm=[1,0,2])
        scan_inputs = (inputs_transposed, self.time_indices)

        initial_state = tf.zeros((batch_size, self.num_regions))
        final_states = tf.scan(
            self.step,
            scan_inputs,
            initializer=initial_state
        )

        post = tf.transpose(final_states, perm=[1,0,2])
        
        # Apply dropout to W during training
        if training:
            W = self.w_dropout(self.W, training=training)
        else:
            W = self.W
            
        return post, W


class MCANGenerator(Model):
    def __init__(
            self,
            n_regions = 5,
            inference_timepoints = 500,
            fmri_timepoints = 100,
            eeg_channels = 3,
    ):
        super(MCANGenerator, self).__init__()
        self.n_regions = n_regions
        self.inference_timepoints = inference_timepoints
        self.fmri_timepoints = fmri_timepoints
        self.eeg_channels = eeg_channels
        
        # Add temperature parameter for slower loss decay
        self.temperature = self.add_weight(
            name="temperature",
            shape=(1,),
            initializer=tf.constant_initializer(0.5),
            trainable=True
        )

    def build(self, input_shape):
        # Handle tuple input shape
        if len(input_shape) == 2:
            fmri_shape, eeg_shape = input_shape
        else:
            fmri_shape, = input_shape
            eeg_shape = None

        # Encoders
        self.fmri_encoder = FmriSourceEncoder(self.fmri_timepoints, self.inference_timepoints)
        self.fmri_encoder.build(fmri_shape)
        
        if eeg_shape is not None:
            self.eeg_encoder = EegSourceEncoder(self.eeg_channels, self.n_regions)
            self.eeg_encoder.build(eeg_shape)

        # Concatenate fmri and eeg
        self.concat = layers.Concatenate(axis=-1)

        # Fusion
        self.fusion_f = AttentionFusion()
        if eeg_shape is not None:
            self.fusion_e = AttentionFusion()

        # Dynamic Causal Network
        self.fmri_dcn = DynamicCausalNetwork(self.n_regions, self.inference_timepoints)
        self.fmri_dcn.build((None, self.inference_timepoints, self.n_regions))
        
        if eeg_shape is not None:
            self.eeg_dcn = DynamicCausalNetwork(self.n_regions, self.inference_timepoints)
            self.eeg_dcn.build((None, self.inference_timepoints, self.n_regions))

        super(MCANGenerator, self).build(input_shape)

    def call(self, inputs, training=None):
        if len(inputs) == 2:
            x_f, x_e = inputs
        elif len(inputs) == 1:
            x_f, = inputs
            x_e = None
        else:
            raise ValueError("Invalid number of inputs")

        # Encoding with gradient dampening
        s_f = self.fmri_encoder(x_f, mode="encode", training=training)
        if x_e is not None:
            s_e = self.eeg_encoder(x_e, mode="encode", training=training)
        
        if x_e is not None:
            o1_f, att_f = self.fusion_f(s_f, s_e, training=training)
            o1_e, att_e = self.fusion_e(s_e, s_f, training=training)
        else:
            o1_f, att_f = self.fusion_f(s_f, s_f, training=training)

        # Dynamic Causal Network with sensitive W
        o2_f, W_f = self.fmri_dcn(s_f, training=training)
        if x_e is not None:
            o2_e, W_e = self.eeg_dcn(s_e, training=training)

        # Combine outputs with reduced sensitivity
        s_f_post = tf.nn.leaky_relu(o1_f + o2_f, alpha=0.1)
        if x_e is not None:
            s_e_post = tf.nn.leaky_relu(o1_e + o2_e, alpha=0.1)

        # Decoding with gradient dampening
        x_f_post = self.fmri_encoder(s_f_post, mode="decode", training=training)
        if x_e is not None:
            x_e_post = self.eeg_encoder(s_e_post, mode="decode", training=training)

        # Process attention scores for fMRI
        # Average attention scores across heads and time
        att_f = tf.reduce_mean(att_f, axis=1)  # Average across heads
        att_f = tf.reduce_mean(att_f, axis=0)  # Average across time
        att_f = tf.reshape(att_f, (self.n_regions, self.n_regions, 1))
        if x_e is not None:
            # Process attention scores for EEG
            att_e = tf.reduce_mean(att_e, axis=1)  # Average across heads
            att_e = tf.reduce_mean(att_e, axis=0)  # Average across time
            att_e = tf.reshape(att_e, (self.n_regions, self.n_regions, 1))
        
        # Make G highly sensitive to updates by:
        # 1. No regularization on W or attention scores
        # 2. Direct multiplication without sigmoid
        # 3. No additional transformations
        # 4. Apply temperature scaling for slower loss decay
        G_f = (W_f * att_f) / (1.0 + self.temperature)
        if x_e is not None:
            G_e = (W_e * att_e) / (1.0 + self.temperature)
            G_mm = (G_f + G_e) / 2
        else:
            G_mm = G_f
        
        if x_e is None:
            return {"x_f_post": x_f_post, "s_f_post": s_f_post, "G_mm": G_mm}
        else:
            return {"x_f_post": x_f_post, "x_e_post": x_e_post, "s_f_post": s_f_post, "s_e_post": s_e_post, "G_mm": G_mm}


class Discriminator(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Reduce network capacity
        self.dense1 = layers.SpectralNormalization(layers.Dense(128))
        self.bn1 = layers.BatchNormalization()
        
        self.dense2 = layers.SpectralNormalization(layers.Dense(64))
        self.bn2 = layers.BatchNormalization()
        
        self.dense3 = layers.SpectralNormalization(layers.Dense(32))
        self.bn3 = layers.BatchNormalization()
        
        self.dense4 = layers.SpectralNormalization(layers.Dense(1))
        self.leaky_relu = layers.LeakyReLU(alpha=0.2)
        self.dropout = layers.Dropout(0.2)  # Reduced dropout
        
        # Add temporal pooling
        self.global_pool = layers.GlobalAveragePooling1D()

    def call(self, inputs, training=None):
        # Reshape input to (batch, timepoints, features)
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, 0)  # Add batch dimension if needed
        
        # Apply temporal pooling to get a single feature vector for the entire sequence
        x = self.global_pool(inputs)
        
        x = self.dense1(x)
        x = self.bn1(x, training=training)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.dense3(x)
        x = self.bn3(x, training=training)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.dense4(x)
        return x  # Remove sigmoid for better gradient flow