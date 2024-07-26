# train_and_test.py

import tensorflow as tf
import tensorflow_gnn as tfgnn
from src.models import GCNModel, GATModel, CombinedGNNModel

def load_graph_data(graph_file):
    # This function should be replaced with actual code to read and preprocess graph_with_embeddings.graphml
    # Example:
    graph = tfgnn.read_graph(graph_file)
    return graph

def preprocess_graph(graph):
    # Apply preprocessing to the graph if needed
    # For example: normalization, splitting into training/validation sets, etc.
    return graph

def train_model(model, graph):
    # Compile the model (adjust loss and metrics as needed)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model (adjust epochs and batch_size as needed)
    model.fit(graph, epochs=10, batch_size=32)

def evaluate_model(model, graph):
    # Evaluate the model
    results = model.evaluate(graph)
    print('Evaluation results:', results)

def main():
    # Load and preprocess the graph data
    graph = load_graph_data('graph_with_embeddings.graphml')
    graph = preprocess_graph(graph)

    # Define model parameters
    num_classes = 10  # Adjust as needed
    hidden_units = 64
    num_heads = 4

    # Instantiate models
    gcn_model = GCNModel(num_classes=num_classes, hidden_units=hidden_units)
    gat_model = GATModel(hidden_units=hidden_units, num_heads=num_heads)
    combined_model = CombinedGNNModel(num_classes=num_classes, hidden_units=hidden_units, num_heads=num_heads)

    # Train and evaluate models
    print("Training GCN Model...")
    train_model(gcn_model, graph)
    print("Evaluating GCN Model...")
    evaluate_model(gcn_model, graph)

    print("Training GAT Model...")
    train_model(gat_model, graph)
    print("Evaluating GAT Model...")
    evaluate_model(gat_model, graph)

    print("Training Combined GNN Model...")
    train_model(combined_model, graph)
    print("Evaluating Combined GNN Model...")
    evaluate_model(combined_model, graph)

if __name__ == "__main__":
    main()
