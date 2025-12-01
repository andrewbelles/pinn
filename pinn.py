#/usr/bin/env python 

import numpy as np 
import tensorflow as tf
import keras 

def new_mlp(input_dim, layers, activation=tf.nn.tanh, ):
    '''
    Creates a multi-layer perceptron given an input dimension and number of layers 
    to define a model backbone for a neural network 
    '''

    inputs = keras.Input(shape=(input_dim,))
    x = inputs 
    for width in layers: 
        x = keras.layers.Dense(
            width,
            activation=activation,
            kernel_initializer="glorot_normal"
        )(x)
    outputs = keras.layers.Dense(
        1, activation=None, kernel_initializer="glorot_normal"
    )(x)
    return keras.Model(inputs=inputs, outputs=outputs)


class PINN: 
    def __init__(self, model, residual_terms, dtype=tf.float32): 
        self.model: keras.Model = model
        self.residual_terms = residual_terms 
        self.dtype = dtype 

    def loss(self):
        losses = []
        for rterms in self.residual_terms:
            r = rterms(self.model) 
            # RMSE 
            losses.append(tf.reduce_mean(tf.square(r)))
        return tf.add_n(losses)

    def train_adam_(self, lr=1e-3, epochs=1000, log_every=100):
        opt = keras.optimizers.Adam(lr)

        @tf.function 
        def step():
            with tf.GradientTape() as tape: 
                losses = self.loss() 
            gradients = tape.gradient(losses, self.model.trainable_variables)
            if gradients is None: 
                raise ValueError(f"no gradients computed at current step")

            opt.apply_gradients(zip(gradients, self.model.trainable_variables))
            return losses
        
        for epoch in range(epochs):
            losses = step() 
            if log_every and (epoch % log_every == 0): 
                tf.print(f"Adam Epoch: {epoch}, Loss: {losses}")

    def predict(self, x): 
        xn = np.asarray(x, dtype=self.dtype.as_numpy_dtype)
        if xn.ndim == 1: 
            xn = xn[:, None]

        xtf = tf.convert_to_tensor(xn, dtype=self.dtype)
        return self.model(xtf, training=False).numpy() 
