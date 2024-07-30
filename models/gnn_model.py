import tensorflow as tf
import tensorflow_gnn as tfgnn

class GNNLinkPredictionModel(tf.keras.Model):
    def __init__(self, hidden_units, input_dim):
        super(GNNLinkPredictionModel, self).__init__()
        self.hidden_units = hidden_units
        self.input_dim = input_dim
        
        # Define the layers
        self.dense1 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_units, activation='relu')
        self.dense_out = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Apply dense layers to input features
        x = self.dense1(inputs)
        x = self.dense2(x)
        
        # Predict edge existence
        edge_scores = self.dense_out(x)
        return edge_scores

def build_and_compile_model(hidden_units, input_dim):
    model = GNNLinkPredictionModel(hidden_units=hidden_units, input_dim=input_dim)
    model.build((None, input_dim)) 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    return model

