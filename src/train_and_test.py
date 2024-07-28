# train_and_test.py
import networkx as nx
import tensorflow as tf
import tensorflow_gnn as tfgnn
import os, sys
from models.gnn_model import build_and_compile_gnn
from models.mlp_model import build_and_compile_mlp
import numpy as np
import random
from sklearn.model_selection import train_test_split

def split_data(features, labels, validation_split=0.2):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=validation_split, random_state=42)
    return X_train, X_val, y_train, y_val

def generate_negative_examples(num_nodes, positive_examples_set, num_samples):
    negative_examples = set()
    
    while len(negative_examples) < num_samples:
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        
        if i != j and (i, j) not in positive_examples_set and (j, i) not in positive_examples_set:
            negative_examples.add((i, j))
    
    return list(negative_examples)

def verify_indices_with_features(positive_examples, node_feature_vectors):
    for source, target in positive_examples[:10]:  # Checking the first 10 examples
        source_features = node_feature_vectors[source]
        target_features = node_feature_vectors[target]
        
def prepare_data(tf_graph):
    node_features = tf_graph.node_sets['nodes'].features
        
    # Extract and convert embeddings directly
    embeddings = node_features['embedding'].numpy()  # Assuming this is already a numpy array
    # Ensure embeddings are of type float64
    embeddings = embeddings.astype(np.float64)
    
    # Extract other features
    followers = node_features['followers'].numpy()
    popularity = node_features['popularity'].numpy()
    pagerank = node_features['pagerank'].numpy()
        
    # Combine features into a single vector for each node
    node_feature_vectors = np.concatenate([
        embeddings,
        followers.reshape(-1, 1),
        popularity.reshape(-1, 1),
        pagerank.reshape(-1, 1)
    ], axis=1)
    
    # Get positive examples: Extract edge list
    edge_list = tf_graph.edge_sets['edges'].adjacency
    positive_examples = list(zip(edge_list.source.numpy(), edge_list.target.numpy()))
    positive_examples_set = set(positive_examples)
    verify_indices_with_features(positive_examples, node_feature_vectors)
    positive_labels = np.ones(len(positive_examples), dtype=np.float32)
    
    # Generate negative examples
    all_nodes = list(range(len(node_feature_vectors)))
    num_negative_samples = len(positive_examples)
    negative_examples = generate_negative_examples(len(all_nodes), positive_examples_set, num_negative_samples)
    negative_labels = [0.0] * len(negative_examples)
    
    # Combine positive and negative examples
    all_examples = positive_examples + negative_examples
    all_labels = np.concatenate([positive_labels, negative_labels])
    
    # Feature vectors for each example
    features = np.array([
        np.concatenate([
            node_feature_vectors[source],
            node_feature_vectors[target]
        ])
        for source, target in all_examples
    ], dtype=np.float32)
        
    return features, all_labels

def check_fixed_embedding_size(node_features):
    embedding_lengths = [len(v['embedding']) for v in node_features.values()]
    
    if len(set(embedding_lengths)) > 1:
        print("Error: Inconsistent embedding sizes detected.")
        print("Embedding lengths:", set(embedding_lengths))
        exit(-1)
    return embedding_lengths[0]

def load_graph_data(graph_file):
    # Load graph using NetworkX
    G = nx.read_graphml(graph_file)
    
    # Create a mapping from node IDs to indices
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    index_to_node = {idx: node for node, idx in node_to_index.items()}
    
    # Extract node_features
    node_features = {}
    for node, data in G.nodes(data=True):
        # Extract and convert node features
        name = data.get('name', '')
        followers = float(data.get('followers', 0.0))
        popularity = int(data.get('popularity', 0))
        pagerank = float(data.get('pagerank', 0.0))
        
        # Handle embedding field
        embedding_str = data.get('embedding', '[]')
        try:
            # Convert string representation of list to actual list of floats
            embedding = list(map(float, embedding_str.strip('[]').split(',')))
        except ValueError:
            print("Fix embedding logic")
            embedding = []  # Set to empty list if conversion fails
        
        node_features[node] = {
            'name': name,
            'followers': followers,
            'popularity': popularity,
            'pagerank': pagerank,
            'embedding': embedding
        }
            
    # Extract edge list and convert to indices
    edge_list = list(G.edges())

    # Convert node IDs in edge_list to integer indices
    edge_list_indices = [(node_to_index[source], node_to_index[target]) for source, target in edge_list]
    
    # Check if all embeddings have the same length
    fixed_size = check_fixed_embedding_size(node_features)
    
    # Create the Graph Data Structure for TensorFlow GNN
    tf_graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'nodes': tfgnn.NodeSet.from_fields(
                features={
                    'name': tf.constant([node_features[index_to_node[idx]]['name'] for idx in range(len(G.nodes()))]),
                    'followers': tf.constant([node_features[index_to_node[idx]]['followers'] for idx in range(len(G.nodes()))]),
                    'popularity': tf.constant([node_features[index_to_node[idx]]['popularity'] for idx in range(len(G.nodes()))]),
                    'pagerank': tf.constant([node_features[index_to_node[idx]]['pagerank'] for idx in range(len(G.nodes()))]),
                    'embedding': tf.constant([node_features[index_to_node[idx]]['embedding'] for idx in range(len(G.nodes()))])
                },
                sizes=[len(G.nodes())]
            )
        },
        edge_sets={
            'edges': tfgnn.EdgeSet.from_fields(
                sizes=[len(edge_list_indices)],
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('nodes', tf.constant([source for source, target in edge_list_indices], dtype=tf.int32)),
                    target=('nodes', tf.constant([target for source, target in edge_list_indices], dtype=tf.int32))
                )
            )
        }
    )
    
    # Verify the created GraphTensor
    print("GraphTensor created successfully.")
    print(f"Node features in GraphTensor: {tf_graph.node_sets['nodes'].features.keys()}")
    print(f"Edge sets in GraphTensor: {tf_graph.edge_sets.keys()}")
    
    return tf_graph


def train_model(model, tf_graph, epochs=10):
    # Prepare training data
    # For demonstration purposes, using random labels. Replace with actual labels.
    num_nodes = len(tf_graph.node_sets['nodes'].features['embedding'])
    labels = tf.random.uniform((num_nodes,), minval=0, maxval=2, dtype=tf.int32)  # Binary labels

    # Train the model
    history = model.fit(
        tf_graph,
        labels,
        epochs=epochs
    )
    return history

def main():
    # Step 1: Load and preprocess the graph data
    datadir = os.environ.get("DATA_DIR")
    graphml = os.path.join(datadir, "processed/graph_with_embeddings.graphml")
    tf_graph = load_graph_data(graphml)
    
    # Step 2: Preapre data for training
    features, labels = prepare_data(tf_graph)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train_gnn, X_val, y_train_gnn, y_val = split_data(X_train, y_train, validation_split=0.2)


    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Comment out one of the following two blocks to switch between models

    # GNN Model
    gnn_model = build_and_compile_gnn(hidden_dim=64, output_dim=32)
    gnn_history = gnn_model.fit((X_train_gnn, y_train_gnn), epochs=10, batch_size=32, validation_data=(tf_graph, y_val))
    gnn_test_loss, gnn_test_accuracy = gnn_model.evaluate(X_test, y_test)
    print(f"GNN Test accuracy: {gnn_test_accuracy:.4f}")

    # MLP Model
    mlp_model = build_and_compile_mlp(input_dim=features.shape[1])
    mlp_history = mlp_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
    mlp_test_loss, mlp_test_accuracy = mlp_model.evaluate(X_test, y_test)
    print(f"MLP Test accuracy: {mlp_test_accuracy:.4f}")

    
if __name__ == "__main__":
    main()
