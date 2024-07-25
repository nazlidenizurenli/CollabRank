from utils.BERTGenreProcessor import BERTGenreProcessor
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import click
import os, sys
import random
import ast
import scipy.sparse
import numpy as np
from tqdm import tqdm


def load_and_clean_data(nodes_path, edges_path, datadir):

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

    # Print updated edge count
    print(f"Edge count after filtering: {len(edges_df)}")

    # Step 5: Process and export unique genres
    # Normalize the genres column to ensure it's a list of strings
    def parse_genres(genres):
        try:
            # Convert string representation of list to actual list
            genres_list = eval(genres) if isinstance(genres, str) else genres
            # Ensure genres_list is a list of strings
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

def build_graph(nodes_df, edges_df):
    G = nx.Graph()

    # Add nodes with metadata
    for index, row in nodes_df.iterrows():
        G.add_node(row['spotify_id'], 
                   name=row['name'], 
                   followers=row['followers'], 
                   popularity=row['popularity'],
                   genres=row['genres'])
    
    # Add edges
    for index, row in edges_df.iterrows():
        G.add_edge(row['id_0'], row['id_1'])

    return G

def calculate_pagerank(G):
    pagerank_dict = nx.pagerank(G)
    nx.set_node_attributes(G, pagerank_dict, 'pagerank')
    return pagerank_dict

def visualize_graph_matplotlib(G, datadir):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15)
    node_color = [G.nodes[n]['pagerank'] for n in G.nodes]
    click.secho("Drawing graph with matplotlib...", fg='green', bold=True)
    nx.draw(G, pos, with_labels=False, node_size=20, node_color=node_color, cmap=plt.cm.Blues, edge_color="gray")
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), ax=plt.gca(), orientation="horizontal")
    savepath = os.path.join(datadir, "processed/collabrank_graph.png")
    plt.savefig(savepath)
    plt.show()

def visualize_graph_pyvis(G, datadir):
    net = Network(notebook=True)
    click.secho("Drawing graph with pyvis...", fg='green', bold=True)

    # Add nodes and edges
    for node in G.nodes(data=True):
        name = node[1].get('name', 'Unknown')
        popularity = node[1].get('popularity', 'N/A')
        pagerank = node[1].get('pagerank', 0)

        net.add_node(node[0], label=name,
                     title=f"Popularity: {popularity}<br>PageRank: {pagerank:.4f}",
                     value=pagerank)

    for edge in G.edges():
        net.add_edge(edge[0], edge[1])

    # Generate the HTML file
    savepath = os.path.join(datadir, "processed/collabrank_graph.html")
    net.show(savepath)

def print_graph_info(G):
    click.secho("Graph Information:", fg="green", bold=True)
    click.secho(f"Number of nodes: {G.number_of_nodes()}", fg="green")
    click.secho(f"Number of edges: {G.number_of_edges()}", fg="green")

    click.secho("Node Features:", fg="green", bold=True)

    for node, data in G.nodes(data=True):
        # 'node' is the node ID itself
        node_id = node
        features = {k: v for k, v in data.items() if k != 'spotify_id'}  # Exclude 'spotify_id' from features
        features_str = ', '.join(f"{k}: {v}" for k, v in features.items())
        
        click.secho(f"Node ID: {node_id}, Features: {features_str}", fg="green")
        
def get_genre_embedding_vector(genres, embedding_dict, max_length=5):
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

def convert_2d_to_1d_embedding_dict(embedding_dict):
    return {genre: np.array([embedding_dict[genre][i] for i in sorted(embedding_dict[genre])]) for genre in embedding_dict}

def main():
    datadir = os.environ.get("DATA_DIR")

    # Check if the directory exists
    if not os.path.isdir(datadir):
        click.secho("Please set the DATA_DIR environment variable and ensure the directory exists.", fg='red')
        
    # Define paths to the CSV files
    nodes_path = os.path.join(datadir, 'raw', 'nodes.csv')
    edges_path = os.path.join(datadir, 'raw', 'edges.csv')

    # Load and clean data
    click.secho("Step 1: Cleaning input data...", fg="yellow", bold=True)
    nodes_df, edges_df, all_genres = load_and_clean_data(nodes_path, edges_path, datadir)

    # Build graph
    click.secho("Step 2: Building graph...", fg="yellow", bold=True)
    G = build_graph(nodes_df, edges_df)

    # Calculate PageRank
    click.secho("Step 3: Calculating page rank for nodes...", fg="yellow", bold=True)
    pagerank_dict = calculate_pagerank(G)

    # Print graph info and sample PageRank scores
    print_graph_info(G)
    click.secho("Calculating Sample PageRank scores:", fg="blue", bold=True)
    for node, pagerank in list(pagerank_dict.items())[:10]:
        print(f"Node {node}: PageRank {pagerank}")

    # Visualize graph with matplotlib
    click.secho("Building graph Visualizations", fg="blue", bold=True)
    
    # Visualize the graph
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
    
    # Integrate embeddings into graph nodes
    # Get total node count
    # Get total node count
    totalcount = len(G.nodes)
    count = 0

    # Initialize tqdm progress bar
    with tqdm(total=totalcount, desc="Processing Nodes") as pbar:
        for node, data in G.nodes(data=True):
            node_id = node
            if 'genres' in G.nodes[node_id]:
                genres = G.nodes[node_id]['genres']
                genre_embeddings_dict = genre_embeddings.to_dict(orient='index')
                genresdict = convert_2d_to_1d_embedding_dict(genre_embeddings_dict)
                embedding_vector = get_genre_embedding_vector(genres, genresdict)
                
                # Ensure embedding vector has the correct length
                assert len(embedding_vector) == embedding_dim, "Dimension of embeddings shouldn't change, check logic"
                
                G.nodes[node_id]['embedding'] = embedding_vector
            else:
                click.secho("Genres list not found", fg="red", bold=True)
                exit(-1)
            
            # Update the progress bar
            pbar.update(1)
    click.secho("Added Genre embeddings as a feature!", fg="green", bold=True)
    # print_graph_info(G)

if __name__ == "__main__":
    main()
