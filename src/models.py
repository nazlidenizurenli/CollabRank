# models.py

import tensorflow as tf
import tensorflow_gnn as tfgnn

class GCNModel(tf.keras.Model):
    def __init__(self, num_classes, hidden_units):
        super(GCNModel, self).__init__()
        self.gcn_layer_1 = tfgnn.layers.GraphConv(hidden_units)
        self.gcn_layer_2 = tfgnn.layers.GraphConv(hidden_units)
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, graph):
        node_features = graph.node_features
        edge_features = graph.edge_features

        x = self.gcn_layer_1(node_features, edge_features)
        x = tf.keras.activations.relu(x)
        x = self.gcn_layer_2(x, edge_features)
        x = tf.keras.activations.relu(x)
        output = self.dense(x)
        return output

class GATModel(tf.keras.Model):
    def __init__(self, hidden_units, num_heads):
        super(GATModel, self).__init__()
        self.gat_layer = tfgnn.layers.MultiHeadGraphAttention(hidden_units, num_heads)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, graph):
        node_features = graph.node_features
        edge_features = graph.edge_features

        x = self.gat_layer(node_features, edge_features)
        x = tf.keras.activations.relu(x)
        output = self.dense(x)
        return output

class CombinedGNNModel(tf.keras.Model):
    def __init__(self, num_classes, hidden_units, num_heads):
        super(CombinedGNNModel, self).__init__()
        self.gcn_layer_1 = tfgnn.layers.GraphConv(hidden_units)
        self.gcn_layer_2 = tfgnn.layers.GraphConv(hidden_units)
        self.gat_layer = tfgnn.layers.MultiHeadGraphAttention(hidden_units, num_heads)
        self.classification_dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.link_pred_dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, graph):
        node_features = graph.node_features
        edge_features = graph.edge_features

        x = self.gcn_layer_1(node_features, edge_features)
        x = tf.keras.activations.relu(x)
        x = self.gcn_layer_2(x, edge_features)
        x = tf.keras.activations.relu(x)
        node_classification_output = self.classification_dense(x)

        x_gat = self.gat_layer(node_features, edge_features)
        x_gat = tf.keras.activations.relu(x_gat)
        link_prediction_output = self.link_pred_dense(x_gat)

        return node_classification_output, link_prediction_output
