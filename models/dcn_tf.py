import tensorflow as tf
from tensorflow.keras import layers

class CausalUnit(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, num_regions, **kwargs):
        """
        Initialize a causal unit for one brain region.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            num_regions (int): Total number of brain regions
        """
        super(CausalUnit, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_regions = num_regions
        
        # GRU cell for temporal information
        self.gru = layers.GRUCell(hidden_size)
        
        # Trainable neurons for inter-region causality
        # We use num_regions-1 because we exclude self-connection
        self.causal_weights = self.add_weight(
            shape=(num_regions - 1, hidden_size),
            initializer='zeros',
            trainable=True,
            name='causal_weights'
        )
        
        # Output layer
        self.output_layer = layers.Dense(input_size)
        
    def call(self, inputs, prev_output, region_idx):
        """
        Forward pass of the causal unit.
        
        Args:
            inputs (tf.Tensor): Input tensor for current time step
            prev_output (tf.Tensor): Previous output from all regions
            region_idx (int): Index of current brain region
            
        Returns:
            tf.Tensor: Output for current region and time step
        """
        # Get GRU output
        gru_output, _ = self.gru(inputs, [prev_output[:, region_idx, :]])
        
        # Calculate causal influence from other regions
        causal_influence = tf.zeros_like(gru_output)
        weight_idx = 0
        
        for r in range(self.num_regions):
            if r != region_idx:  # Skip self-connection
                # Apply causal weight and activation
                causal_influence += tf.tanh(self.causal_weights[weight_idx] * prev_output[:, r, :])
                weight_idx += 1
        
        # Combine GRU output and causal influence
        combined = tf.tanh(gru_output + causal_influence)
        
        # Generate output
        output = self.output_layer(combined)
        
        return output

class DynamicCausalNetwork(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_regions, sequence_length, **kwargs):
        """
        Initialize the Dynamic Causal Network.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            num_regions (int): Number of brain regions
            sequence_length (int): Length of time series
        """
        super(DynamicCausalNetwork, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_regions = num_regions
        self.sequence_length = sequence_length
        
        # Create causal units for each brain region
        self.causal_units = [
            CausalUnit(input_size, hidden_size, num_regions)
            for _ in range(num_regions)
        ]
        
    def call(self, inputs):
        """
        Forward pass of the DCN.
        
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, sequence_length, num_regions, input_size)
            
        Returns:
            tf.Tensor: Generated time series of shape (batch_size, sequence_length, num_regions, input_size)
        """
        batch_size = tf.shape(inputs)[0]
        
        # Initialize outputs
        outputs = tf.zeros((batch_size, self.sequence_length, self.num_regions, self.input_size))
        
        # Process each time step
        for t in range(self.sequence_length):
            # Process each region
            for r in range(self.num_regions):
                # Get current input for region r
                current_input = inputs[:, t, r, :]
                
                # Get previous outputs for all regions
                prev_outputs = outputs[:, t-1, :, :] if t > 0 else tf.zeros_like(inputs[:, 0, :, :])
                
                # Process through causal unit
                outputs = tf.tensor_scatter_nd_update(
                    outputs,
                    [[i, t, r] for i in range(batch_size)],
                    self.causal_units[r](current_input, prev_outputs, r)
                )
        
        return outputs

    def get_causal_weights(self):
        """
        Get the causal weights between brain regions.
        
        Returns:
            tf.Tensor: Causal weights matrix of shape (num_regions, num_regions)
        """
        weights = tf.zeros((self.num_regions, self.num_regions))
        
        for r in range(self.num_regions):
            weight_idx = 0
            for other_r in range(self.num_regions):
                if other_r != r:
                    weights = tf.tensor_scatter_nd_update(
                        weights,
                        [[r, other_r]],
                        [tf.reduce_mean(self.causal_units[r].causal_weights[weight_idx])]
                    )
                    weight_idx += 1
        
        return weights 