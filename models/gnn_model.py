import tensorflow as tf
import tensorflow_gnn as tfgnn

class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(GCNLayer, self).__init__()
        self.output_dim = output_dim
        self.dense = tf.keras.layers.Dense(output_dim)
    
    def call(self, node_features, adjacency_matrix):
        # Node features aggregation (message passing)
        aggregated_features = tf.sparse.sparse_dense_matmul(adjacency_matrix, node_features)
        # Apply dense layer to aggregated features
        return self.dense(aggregated_features)

class LinkPredictionGNN(tf.keras.Model):
    def __init__(self, hidden_dim, output_dim):
        super(LinkPredictionGNN, self).__init__()
        self.gcn1 = GCNLayer(hidden_dim)
        self.gcn2 = GCNLayer(output_dim)
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, graph_tensor):
        adjacency = graph_tensor.edge_sets['edges'].adjacency
        adjacency_matrix = tfgnn.utils.adjacency_to_dense(adjacency)
        node_features = graph_tensor.node_sets['nodes'].features['embedding']
        
        # Apply GCN layers
        hidden_rep = self.gcn1(node_features, adjacency_matrix)
        node_embeddings = self.gcn2(hidden_rep, adjacency_matrix)
        
        # Aggregate node embeddings for link prediction
        source_indices, target_indices = adjacency.source, adjacency.target
        source_embeddings = tf.gather(node_embeddings, source_indices)
        target_embeddings = tf.gather(node_embeddings, target_indices)
        link_features = tf.concat([source_embeddings, target_embeddings], axis=1)
        
        return self.dense(link_features)

def build_and_compile_gnn(hidden_dim=64, output_dim=32):
    model = LinkPredictionGNN(hidden_dim, output_dim)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
