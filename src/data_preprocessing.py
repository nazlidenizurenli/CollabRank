"""
data_preprocessing.py

This script performs various data preprocessing tasks for a graph-based data analysis project. 
It involves loading and cleaning data, constructing a graph, calculating PageRank, 
visualizing the graph using different tools, processing genre embeddings with BERT, 
and saving the processed graph with updated embeddings.

Imports:
- Data manipulation: pandas, numpy
- Graph handling: networkx, pyvis
- Visualization: matplotlib
- Utilities: click, json, os, sys, random, ast, scipy.sparse, concurrent.futures, tqdm

Functions:
- convert_ndarray_to_list: Converts NumPy arrays to lists.
- convert_list_to_string: Converts lists to JSON strings.
- preprocess_graph: Converts node and edge attributes to ensure compatibility with GraphML format.
- compute_node_embedding: Computes embedding vectors for nodes based on genre embeddings.
- load_and_clean_data: Loads and filters data, processes genres, and saves cleaned data.
- build_graph: Constructs a graph from node and edge data.
- calculate_pagerank: Calculates PageRank for nodes in the graph.
- visualize_graph_matplotlib: Visualizes the graph using Matplotlib.
- visualize_graph_pyvis: Visualizes the graph using PyVis.
- print_graph_info: Prints information about the graph and its nodes.
- get_genre_embedding_vector: Generates a mean embedding vector for a list of genres.
- convert_2d_to_1d_embedding_dict: Converts 2D embeddings to a 1D format.
- main: Main function that orchestrates the data preprocessing steps.
"""

from utils.BERTGenreProcessor import BERTGenreProcessor
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import click
import os
import json
from collections import defaultdict
import numpy as np
import concurrent.futures
from tqdm import tqdm

def convert_ndarray_to_list(data: np.ndarray) -> list:
    """
    Converts a NumPy array to a list. If the data is not a NumPy array, returns the data as is.

    Args:
        data (np.ndarray): The NumPy array to be converted.

    Returns:
        list: The converted list.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


def convert_list_to_string(data: list) -> str:
    """
    Converts a list to a JSON string. If the data is not a list, returns the data as is.

    Args:
        data (list): The list to be converted.

    Returns:
        str: The JSON string representation of the list.
    """
    if isinstance(data, list):
        return json.dumps(data)
    return data


def preprocess_graph(G: nx.Graph) -> nx.Graph:
    """
    Preprocesses the graph to ensure that all attributes are compatible with GraphML format.
    Converts ndarray and list attributes to JSON strings.

    Args:
        G (nx.Graph): The graph to be preprocessed.

    Returns:
        nx.Graph: The preprocessed graph.
    """
    for node in G.nodes(data=True):
        node_id, attrs = node
        for key, value in attrs.items():
            value = convert_ndarray_to_list(value)
            attrs[key] = convert_list_to_string(value)
    
    for edge in G.edges(data=True):
        u, v, attrs = edge
        for key, value in attrs.items():
            value = convert_ndarray_to_list(value)
            attrs[key] = convert_list_to_string(value)
    
    return G


def compute_node_embedding(node_id: str, genre_embeddings_dict: dict, embedding_dim: int, genres_dictionary: dict) -> tuple:
    """
    Computes the embedding vector for a node based on its genres.

    Args:
        node_id (str): The ID of the node.
        genre_embeddings_dict (dict): A dictionary of genre embeddings.
        embedding_dim (int): The dimensionality of the embeddings.

    Returns:
        tuple: A tuple containing the node ID and the computed embedding vector.
    """
    node_genres = genres_dictionary[node_id]
    if node_genres:
        embedding_vector = get_genre_embedding_vector(node_genres, genre_embeddings_dict)
        
        # Ensure embedding vector has the correct length
        assert len(embedding_vector) == embedding_dim, "Dimension of embeddings shouldn't change, check logic"
        
        return node_id, embedding_vector
    else:
        return node_id, None


def load_and_clean_data(nodes_path: str, edges_path: str, datadir: str) -> tuple:
    """
    Loads and cleans node and edge data. Filters out nodes based on empty genres or low followers,
    processes genres, and saves the cleaned data.

    Args:
        nodes_path (str): Path to the CSV file containing node data.
        edges_path (str): Path to the CSV file containing edge data.
        datadir (str): Directory to save the processed files.

    Returns:
        tuple: DataFrames for nodes and edges, and a set of all unique genres.
    """
    # Load data
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # Print initial counts
    print(f"Initial node count: {len(nodes_df)}")
    print(f"Initial edge count: {len(edges_df)}")

    # Drop the 'chart_hits' column from the nodes dataframe if it exists
    if 'chart_hits' in nodes_df.columns:
        nodes_df = nodes_df.drop(columns=['chart_hits'])

    # Threshold for followers
    followers_threshold = 400000

    # Step 1: Record IDs where genres are empty
    empty_genres_ids = set(nodes_df[nodes_df['genres'].apply(lambda x: len(eval(x)) == 0)]['spotify_id'])

    # Step 2: Record IDs where followers are below the threshold
    low_followers_ids = set(nodes_df[nodes_df['followers'] < followers_threshold]['spotify_id'])

    # Combine IDs to be dropped
    dropped_ids = empty_genres_ids.union(low_followers_ids)

    # Print the number of IDs to be dropped
    print(f"Number of nodes to be dropped due to empty genres: {len(empty_genres_ids)}")
    print(f"Number of nodes to be dropped due to low followers: {len(low_followers_ids)}")
    print(f"Total number of nodes to be dropped: {len(dropped_ids)}")

    # Print IDs that are in both categories
    both_categories_ids = empty_genres_ids.intersection(low_followers_ids)
    print(f"Number of nodes in both categories: {len(both_categories_ids)}")

    # Step 3: Drop rows from the nodes DataFrame
    nodes_df = nodes_df[~nodes_df['spotify_id'].isin(dropped_ids)]

    # Print updated node count
    print(f"Node count after filtering: {len(nodes_df)}")

    # Step 4: Update edges DataFrame
    edges_df = edges_df[~edges_df['id_0'].isin(dropped_ids)]
    edges_df = edges_df[~edges_df['id_1'].isin(dropped_ids)]
    
    valid_node_ids = set(nodes_df['spotify_id'])
    edges_df = edges_df[edges_df['id_0'].isin(valid_node_ids) & edges_df['id_1'].isin(valid_node_ids)]

    # Print updated edge count
    print(f"Edge count after filtering: {len(edges_df)}")

    # Step 5: Process and export unique genres
    # Normalize the genres column to ensure it's a list of strings
    def parse_genres(genres: str) -> list:
        """
        Converts genre strings to lists and ensures genres are valid strings.

        Args:
            genres (str): The genre string to be converted.

        Returns:
            list: A list of valid genres.
        """
        try:
            genres_list = eval(genres) if isinstance(genres, str) else genres
            if isinstance(genres_list, list):
                return [genre for genre in genres_list if isinstance(genre, str) and genre.strip()]
        except (SyntaxError, ValueError):
            return []
        return []

    nodes_df['genres'] = nodes_df['genres'].apply(parse_genres)

    # Flatten the list of genres and collect unique genres
    all_genres = set()
    for genres_list in nodes_df['genres']:
        all_genres.update(genres_list)

    # Export all unique genres to a text file
    genres_file_path = os.path.join(datadir, 'processed', 'allgenres.txt')
    with open(genres_file_path, 'w') as f:
        for genre in sorted(all_genres):
            f.write(f"{genre}\n")

    print(f"Unique genres have been exported to: {genres_file_path}")

    # Ensure the 'processed' directory exists
    processed_dir = os.path.join(datadir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Save the cleaned data to CSV files
    processed_nodes_path = os.path.join(processed_dir, 'nodes_filtered.csv')
    processed_edges_path = os.path.join(processed_dir, 'edges_filtered.csv')

    nodes_df.to_csv(processed_nodes_path, index=False)
    edges_df.to_csv(processed_edges_path, index=False)

    print(f"Filtered nodes have been saved to: {processed_nodes_path}")
    print(f"Filtered edges have been saved to: {processed_edges_path}")

    return nodes_df, edges_df, all_genres


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """
    Constructs a graph from the node and edge DataFrames.

    Args:
        nodes_df (pd.DataFrame): DataFrame containing node information.
        edges_df (pd.DataFrame): DataFrame containing edge information.

    Returns:
        nx.Graph: The constructed graph.
    """
    G = nx.Graph()
    genres_dictionary = defaultdict(list)

    for index, row in nodes_df.iterrows():
        G.add_node(row['spotify_id'], 
                   name=row['name'], 
                   followers=row['followers'], 
                   popularity=row['popularity'],
                   embedding="",
                   pagerank=0.0)
        
        genres_dictionary[row['spotify_id']] = genres=row['genres']
        
    
    for index, row in edges_df.iterrows():
        G.add_edge(row['id_0'], row['id_1'])

    return G, genres_dictionary


def calculate_pagerank(G: nx.Graph) -> dict:
    """
    Calculates the PageRank of nodes in the graph.

    Args:
        G (nx.Graph): The graph for which PageRank is to be computed.

    Returns:
        dict: A dictionary with nodes as keys and their PageRank as values.
    """
    pagerank_dict = nx.pagerank(G)
    nx.set_node_attributes(G, pagerank_dict, 'pagerank')
    return pagerank_dict


def visualize_graph_matplotlib(G: nx.Graph, datadir: str):
    """
    Visualizes the graph using Matplotlib.

    Args:
        G (nx.Graph): The graph to be visualized.
        save_path (str): The path to save the visualization image.
    """
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15)
    node_color = [G.nodes[n]['pagerank'] for n in G.nodes]
    click.secho("Drawing graph with matplotlib...", fg='green', bold=True)
    nx.draw(G, pos, with_labels=False, node_size=20, node_color=node_color, cmap=plt.cm.Blues, edge_color="gray")
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), ax=plt.gca(), orientation="horizontal")
    savepath = os.path.join(datadir, "processed/collabrank_graph.png")
    plt.savefig(savepath)
    plt.show()


def visualize_graph_pyvis(G: nx.Graph, datadir: str):
    """
    Visualizes the graph using PyVis.

    Args:
        G (nx.Graph): The graph to be visualized.
        save_path (str): The path to save the visualization HTML file.
    """
    net = Network(notebook=True)
    click.secho("Drawing graph with pyvis...", fg='green', bold=True)

    for node in G.nodes(data=True):
        name = node[1].get('name', 'Unknown')
        popularity = node[1].get('popularity', 'N/A')
        pagerank = node[1].get('pagerank', 0)

        net.add_node(node[0], label=name,
                     title=f"Popularity: {popularity}<br>PageRank: {pagerank:.4f}",
                     value=pagerank)

    for edge in G.edges():
        net.add_edge(edge[0], edge[1])

    savepath = os.path.join(datadir, "processed/collabrank_graph.html")
    net.show(savepath)


def print_graph_info(G: nx.Graph):
    """
    Prints information about the graph including the number of nodes, edges,
    and a sample of nodes and edges.

    Args:
        G (nx.Graph): The graph to be analyzed.
    """
    click.secho("Graph Information:", fg="green", bold=True)
    click.secho(f"Number of nodes: {G.number_of_nodes()}", fg="green")
    click.secho(f"Number of edges: {G.number_of_edges()}", fg="green")

    click.secho("Node Features:", fg="green", bold=True)

    for node, data in G.nodes(data=True):
        node_id = node
        features = {k: v for k, v in data.items() if k != 'spotify_id'}
        features_str = ', '.join(f"{k}: {v}" for k, v in features.items())
        
        click.secho(f"Node ID: {node_id}, Features: {features_str}", fg="green")


def get_genre_embedding_vector(genres: list, embedding_dict: dict, max_length=5) -> np.ndarray:
    """
    Generates a mean embedding vector for a list of genres.

    Args:
        genres (list): A list of genres.
        genre_embeddings_dict (dict): A dictionary mapping genres to their embedding vectors.

    Returns:
        np.ndarray: The mean embedding vector for the given genres.
    """
    # Determine the dimensionality of the embeddings
    example_embedding = next(iter(embedding_dict.values()))
    embedding_dim = example_embedding.shape[0]
    
    # Create embeddings for genres
    embeddings = []
    for genre in genres:
        embedding = embedding_dict.get(genre, np.zeros(embedding_dim))
        embeddings.append(embedding)
    
    # Pad or truncate the list of embeddings
    if len(embeddings) > max_length:
        embeddings = embeddings[:max_length]
    else:
        embeddings += [np.zeros(embedding_dim)] * (max_length - len(embeddings))
    
    # Stack embeddings into a matrix and compute the mean vector
    embedding_matrix = np.vstack(embeddings)
    result_vector = np.mean(embedding_matrix, axis=0)
    
    return result_vector


def convert_2d_to_1d_embedding_dict(embedding_dict: dict) -> dict:
    """
    Converts a dictionary of 2D embeddings to 1D format for easier storage or processing.

    Args:
        embedding_dict (dict): A dictionary where keys are nodes and values are 2D embeddings.

    Returns:
        dict: A dictionary where values are 1D embeddings.
    """
    return {genre: np.array([embedding_dict[genre][i] for i in sorted(embedding_dict[genre])]) for genre in embedding_dict}


def main():
    """
    Main function that orchestrates the data preprocessing steps including loading,
    cleaning, processing genres, building the graph, calculating PageRank, 
    and visualizing the graph.
    """
    datadir = os.environ.get("DATA_DIR")
    if not os.path.isdir(datadir):
        click.secho("Please set the DATA_DIR environment variable and ensure the directory exists.", fg='red')
        
    nodes_path = os.path.join(datadir, 'raw', 'nodes.csv')
    edges_path = os.path.join(datadir, 'raw', 'edges.csv')

    # Load and clean data
    click.secho("Step 1: Cleaning input data...", fg="yellow", bold=True)
    nodes_df, edges_df, all_genres = load_and_clean_data(nodes_path, edges_path, datadir)
    
    # Build graph
    click.secho("Step 2: Building graph...", fg="yellow", bold=True)
    G, genres_dictionary = build_graph(nodes_df, edges_df)

    # Calculate PageRank
    click.secho("Step 3: Calculating page rank for nodes...", fg="yellow", bold=True)
    pagerank_dict = calculate_pagerank(G)

    # Print graph info and sample PageRank scores
    print_graph_info(G)
    click.secho("Calculating Sample PageRank scores:", fg="blue", bold=True)
    for node, pagerank in list(pagerank_dict.items())[:10]:
        print(f"Node {node}: PageRank {pagerank}")

    # Visualize graph with matplotlib
    # click.secho("Building graph Visualizations", fg="blue", bold=True)
    # visualize_graph_matplotlib(G, datadir)
    # visualize_graph_pyvis(G, datadir)
    
    # Process genres with BERT
    click.secho("Building BERT Processor", fg="blue", bold=True)
    processor = BERTGenreProcessor()
    click.secho("Creating embeddings for genres...", fg="yellow", bold=True)
    genre_embeddings = processor.process_genres(nodes_df)
    click.secho("Completed Genre Embeddings!", fg="green", bold=True)
    genre_embeddings.to_csv(os.path.join(datadir, 'processed', 'genre_embeddings.csv'))
    
    # Data Verification for embeddings:
    example_embedding = genre_embeddings.iloc[0].values
    embedding_dim = example_embedding.shape[0]
    print(f"Dimension of a single genre embedding: {embedding_dim}")  
    consistent_embedding_size = all(embedding.shape[0] == embedding_dim for embedding in genre_embeddings.values)
    genre_embeddings = genre_embeddings.apply(pd.to_numeric)
    if not consistent_embedding_size:
        raise ValueError("Not all genre embeddings have the same size!")
    else:
        print("All genre embeddings have the same size.")
    
    # Convert genre embeddings to the correct format
    genre_embeddings_dict = genre_embeddings.to_dict(orient='index')
    genresdict = convert_2d_to_1d_embedding_dict(genre_embeddings_dict)

    # Integrate embeddings into graph nodes with parallel processing
    click.secho("Adding Genre embeddings as a feature...", fg="blue", bold=True)
    
    count = 0
    embedding_dim = example_embedding.shape[0]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(compute_node_embedding, node_id, genresdict, embedding_dim, genres_dictionary): node_id for node_id, data in G.nodes(data=True)}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Nodes"):
            node_id, embedding_vector = future.result()
            if embedding_vector is not None:
                G.nodes[node_id]['embedding'] = embedding_vector
            count += 1
            click.secho(f"Processed {count}/{len(G.nodes)} Nodes", fg="blue")
    
    click.secho("Added Genre embeddings as a feature!", fg="green", bold=True)
    
    # Save Graph
    graph_path = os.path.join(datadir, "processed/graph_with_embeddings.graphml")
    G = preprocess_graph(G)
    nx.write_graphml(G, graph_path)
    click.secho(f"Graph with embeddings saved to {graph_path}", fg='green')


if __name__ == "__main__":
    main()
