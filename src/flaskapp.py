from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_gnn as tfgnn
from ml_ops import create_graph_tensor_for_pair, predict_collaboration, check_existing_collaboration, get_artist_features
import json

app = Flask(__name__)

# Load your model
model = tf.keras.models.load_model('path_to_your_model')

@app.route('/check_collaboration', methods=['POST'])
def check_collaboration():
    data = request.json
    artist1 = data.get('artist1')
    artist2 = data.get('artist2')

    # Check if artists are in the dataset
    if not check_existing_collaboration(artist1, artist2):
        return jsonify({"error": "One or both artists not found in the dataset or collaboration already exists"}), 404

    # Check if they have already collaborated
    if check_existing_collaboration(artist1, artist2):
        return jsonify({"message": "Artists have already collaborated"}), 200

    # Get features for both artists
    node1_features = get_artist_features(artist1)
    node2_features = get_artist_features(artist2)
    
    # Print the features
    print(f"Features for {artist1}: {node1_features}")
    print(f"Features for {artist2}: {node2_features}")

    # Assert the dimensions
    expected_length = 769 # 768 for embedding + 1 for pagerank + 1 for popularity + 1 for followers

    assert len(node1_features) == expected_length, f"Unexpected feature length for {artist1}: {len(node1_features)}"
    assert len(node2_features) == expected_length, f"Unexpected feature length for {artist2}: {len(node2_features)}"

    print(f"Both nodes have the expected feature length of {expected_length}.")

    # Predict likelihood of collaboration
    prediction = predict_collaboration(node1_features, node2_features, model)

    return jsonify({"likelihood": float(prediction.numpy())}), 200

if __name__ == '__main__':
    app.run(debug=True)
