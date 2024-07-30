# train_and_test.py
import networkx as nx
import tensorflow as tf
import tensorflow_gnn as tfgnn
import os, sys
from models.gnn_model import build_and_compile_model
from models.mlp_model import build_mlp_model
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def split_data(features, labels, validation_split=0.2):
    """
    Splits the features and labels into training and validation sets.
    
    Parameters:
    - features: np.array, Feature data for the nodes.
    - labels: np.array, Corresponding labels for the nodes.
    - validation_split: float, Proportion of the data to use for validation.

    Returns:
    - X_train: Training features.
    - X_val: Validation features.
    - y_train: Training labels.
    - y_val: Validation labels.
    """
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=validation_split, random_state=42)
    return X_train, X_val, y_train, y_val
        
        
def generate_negative_examples(num_nodes, positive_examples_set, num_negative_samples):
    """
    Generates negative examples (non-existent edges) for training.

    Parameters:
    - num_nodes: int, Number of nodes in the graph.
    - positive_examples_set: set, Set of positive edges to avoid duplicates.
    - num_negative_samples: int, Number of negative samples to generate.

    Returns:
    - negative_examples: list of tuples, Each tuple represents a negative edge.
    """
    negative_examples = []
    all_nodes = list(range(num_nodes))
    
    while len(negative_examples) < num_negative_samples:
        source = np.random.choice(all_nodes)
        target = np.random.choice(all_nodes)
        
        if source != target and (source, target) not in positive_examples_set:
            negative_examples.append((source, target))
    
    return negative_examples

def prepare_data(tf_graph, node_features):
    """
    Prepares features and labels for training by extracting node features and generating positive/negative examples.

    Parameters:
    - tf_graph: tfgnn.GraphTensor, The TensorFlow Graph representation.
    - node_features: dict, Dictionary mapping node IDs to their features.

    Returns:
    - features: np.array, Feature vectors for the training examples.
    - all_labels: np.array, Labels corresponding to the features.
    """
    # Extract node features
    feature_vectors = []
    for node_id, features in node_features.items():
        # Extract each feature
        embeddings = np.array(features.get('embedding', np.zeros((1,))), dtype=np.float64)
        # Print shape of embeddings for debugging
        if embeddings.shape[0] == 1:
            print(f"Warning: Node {node_id} has embeddings shape {embeddings.shape}. Expected ~700.")
        followers = np.array([features.get('followers', 0)], dtype=np.float64)
        popularity = np.array([features.get('popularity', 0)], dtype=np.float64)
        pagerank = np.array([features.get('pagerank', 0)], dtype=np.float64)

        # Debug: print shapes of individual features
        print(f"Node {node_id} - embeddings shape: {embeddings.shape}")
        print(f"Node {node_id} - followers shape: {followers.shape}")
        print(f"Node {node_id} - popularity shape: {popularity.shape}")
        print(f"Node {node_id} - pagerank shape: {pagerank.shape}")

        # Concatenate features into a single vector
        node_feature_vector = np.concatenate([
            embeddings.flatten(),  # Flatten embeddings to 1D if it's multidimensional
            followers.flatten(),
            popularity.flatten(),
            pagerank.flatten()
        ])
        feature_vectors.append(node_feature_vector)

    node_feature_vectors = np.array(feature_vectors, dtype=np.float64)
    
    # Debug: print shape of the node_feature_vectors
    print(f"Node feature vectors shape: {node_feature_vectors.shape}")
    
    # Get positive examples from edges
    edge_list = tf_graph.edge_sets['edges'].adjacency
    source_nodes = edge_list.source.numpy()
    target_nodes = edge_list.target.numpy()
    
    positive_examples = list(zip(source_nodes, target_nodes))
    positive_examples_set = set(positive_examples)
    positive_labels = np.ones(len(positive_examples), dtype=np.float32)
    
    # Generate negative examples
    num_nodes = len(node_feature_vectors)
    num_negative_samples = len(positive_examples)
    negative_examples = generate_negative_examples(num_nodes, positive_examples_set, num_negative_samples)
    negative_labels = np.zeros(len(negative_examples), dtype=np.float32)
    
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
    
    # Debug: print shapes of features and labels
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    return features, all_labels

def verify_indices_with_features(examples, feature_vectors):
    """
    Verifies that the node indices in the examples are valid and within the range of feature vectors.

    Parameters:
    - examples: list of tuples, Each tuple represents an edge with source and target nodes.
    - feature_vectors: np.array, Array of node feature vectors.

    Raises:
    - ValueError: If any node index is out of bounds.
    """
    for source, target in examples:
        if source >= len(feature_vectors) or target >= len(feature_vectors):
            raise ValueError(f"Invalid node index: source={source}, target={target}")

def check_fixed_embedding_size(node_features):
    """
    Checks if all node embeddings have the same size and returns the embedding size.

    Parameters:
    - node_features: dict, Dictionary mapping node IDs to their features.

    Returns:
    - embedding_size: int, Size of each embedding vector.

    Raises:
    - SystemExit: If inconsistent embedding sizes are detected.
    """
    embedding_lengths = [len(v['embedding']) for v in node_features.values()]
    
    if len(set(embedding_lengths)) > 1:
        print("Error: Inconsistent embedding sizes detected.")
        print("Embedding lengths:", set(embedding_lengths))
        exit(-1)
    return embedding_lengths[0]

def load_graph_data(graph_file):
    """
    Loads graph data from a file and converts it into TensorFlow GraphTensor format.

    Parameters:
    - graph_file: str, Path to the graph file.

    Returns:
    - graph_tensor: tfgnn.GraphTensor, TensorFlow Graph representation.
    - node_features: dict, Dictionary mapping node IDs to their features.
    """
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
    embedding_dim = check_fixed_embedding_size(node_features)
    
    print(f"All data is: {len(node_features)}")
    print(f"Embedding type is: {type(node_features[index_to_node[0]]['embedding'][0])}")
    
    node_ids = list(node_features.keys())
    num_nodes = len(node_ids)

    # Extract features and convert to tensors
    followers = tf.constant([node_features[node_id]['followers'] for node_id in node_ids], dtype=tf.float32)
    popularity = tf.constant([node_features[node_id]['popularity'] for node_id in node_ids], dtype=tf.float32)
    pagerank = tf.constant([node_features[node_id]['pagerank'] for node_id in node_ids], dtype=tf.float32)
    
    # Convert embeddings (list of floats) to tensors
    embedding_dim = len(next(iter(node_features.values()))['embedding'])  # Get embedding dimension from the first entry
    embeddings = tf.constant([node_features[node_id]['embedding'] for node_id in node_ids], dtype=tf.float32)

    # Combine features into a single tensor
    node_feature_matrix = tf.concat([embeddings, tf.expand_dims(followers, axis=1),
                                     tf.expand_dims(popularity, axis=1),
                                     tf.expand_dims(pagerank, axis=1)], axis=1)
    
    # Construct TensorFlow GraphTensor
    edge_set = tfgnn.EdgeSet.from_edges(
        source=tf.constant([source for source, target in edge_list_indices], dtype=tf.int64),
        target=tf.constant([target for source, target in edge_list_indices], dtype=tf.int64),
        num_nodes=num_nodes
    )
    
    graph_tensor = tfgnn.GraphTensor.from_pieces(
        node_sets={'nodes': tfgnn.NodeSet.from_tensor_slices(node_feature_matrix)},
        edge_sets={'edges': edge_set}
    )
    
    return graph_tensor, node_features

def main():
    # Step 1: Load and preprocess the graph data
    datadir = os.environ.get("DATA_DIR")
    graphml = os.path.join(datadir, "processed/graph_with_embeddings.graphml")
    tf_graph, node_features = load_graph_data(graphml)
    
    # Get and print the number of nodes
    num_nodes = sum(size for size in tf_graph.node_sets['nodes'].sizes)
    print(f"Number of nodes: {num_nodes}")
    
    # Get and print the number of edges
    num_edges = sum(size for size in tf_graph.edge_sets['edges'].sizes)
    print(f"Number of edges: {num_edges}")
    
    features, labels = prepare_data(tf_graph, node_features)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Print some examples from features and labels
    print("\nSample of features and labels:")
    
    num_samples_to_print = 5  # Number of samples you want to print
    
    train_size = int(0.8 * len(labels))
    train_features, train_labels = features[:train_size], labels[:train_size]
    test_features, test_labels = features[train_size:], labels[train_size:]
    
    # Define the number of hidden units based on feature dimensions
    hidden_units = 64
    input_dim = train_features.shape[1]  # The input dimension should match the feature size
    
    # Build and compile the model
    model = build_and_compile_model(hidden_units, input_dim)
        
    # Train the model
    history = model.fit(train_features, 
                        train_labels, 
                        epochs=10, 
                        batch_size=32, 
                        validation_split=0.1)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_features, test_labels)
    print(f"\nTest Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    
        
if __name__ == "__main__":
    main()
