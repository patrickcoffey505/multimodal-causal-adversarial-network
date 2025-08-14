from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt

class AttentionWithMap(layers.MultiHeadAttention):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(AttentionWithMap, self).__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            **kwargs
        )
        self.attention_map = None

    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=True, training=None):
        # Call the parent's call method with return_attention_scores=True
        output, attention_scores = super().call(
            query=query,
            value=value,
            key=key,
            attention_mask=attention_mask,
            return_attention_scores=True,
            training=training
        )
        
        # Store the attention map
        self.attention_map = attention_scores
        
        return output, attention_scores

def plot_attention_map(attention_map, head_idx=0, title="Attention Map"):
    """
    Plot the attention map for a specific head.
    
    Args:
        attention_map: tensor of shape (batch_size, num_heads, query_length, key_length)
        head_idx: int, which head to visualize
        title: str, title for the plot
    """
    # Take the first batch element and specified head
    attn = attention_map[0, head_idx].numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.title(f"{title} (Head {head_idx})")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.show()

# Example usage:
# Create the attention layer
# num_heads = 8
# key_dim = 32
# attention_layer = AttentionWithMap(num_heads=num_heads, key_dim=key_dim)

# Apply attention
# s_f_transposed = tf.transpose(s_f, perm=[0,2,1])
# output, attention_map = attention_layer(
#     query=s_f_transposed,
#     value=s_f_transposed
# )

# Plot attention map for the first head
# plot_attention_map(attention_map, head_idx=0) 