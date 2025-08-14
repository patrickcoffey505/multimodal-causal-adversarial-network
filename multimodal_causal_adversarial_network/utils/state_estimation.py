import numpy as np
import tensorflow as tf
from ruptures import Pelt
from tensorflow.keras import layers


class StateEstimation(layers.Layer):
    def __init__(self, penalty=1.0, **kwargs):
        super(StateEstimation, self).__init__(**kwargs)
        self.penalty = penalty
        self.pelt = Pelt(jump=1, min_size=1)
        
    def call(self, G_mm):
        # G_mm shape: (L, L, t)
        G_np = G_mm.numpy()
        L = tf.shape(G_mm)[1]
        t = tf.shape(G_mm)[2]

        G_flat = G_np.reshape(t, L*L)
        self.pelt.fit(G_flat)
        change_points = self.pelt.predict(pen=self.penalty)
        
        if change_points[0] != 0:
            change_points = np.insert(change_points, 0, 0)
        if change_points[-1] != len(G_flat):
            change_points = np.append(change_points, len(G_flat))
        
        segments = []
        for i in range(len(change_points) - 1):
            segments.append(G_np[:,:,change_points[i]:change_points[i+1]])
        
        # Stack segments along a new axis
        return segments