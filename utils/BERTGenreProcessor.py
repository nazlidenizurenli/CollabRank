# genre_processing.py
from transformers import BertTokenizer, BertModel
import torch
from ast import literal_eval
import pandas as pd
import numpy as np

class BERTGenreProcessor:
    def __init__(self):
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def get_embeddings(self, texts):
        # Tokenize and get embeddings
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Return the embeddings of [CLS] token
        return outputs.last_hidden_state[:, 0, :].numpy()

    def process_genres(self, genres_df):
        genres_df['genre_list'] = genres_df['genres'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
        unique_genres = set()
        for genre_list in genres_df['genre_list']:
            unique_genres.update(genre_list)
        unique_genres = list(unique_genres)
        
        # Generate embeddings
        embeddings = self.get_embeddings(unique_genres)
        genre_embeddings = pd.DataFrame(embeddings, index=unique_genres)
        
        return genre_embeddings
