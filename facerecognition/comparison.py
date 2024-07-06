import numpy as np
import os

from facerecognition.utils import bray_curtis_distance, euclidean_distance


class One2Many:
    """
    A class to handle one-to-many comparisons of embeddings using specified distance metrics.

    Attributes:
        distance (function): The distance metric function to use for comparisons.
        db_path (str): Path to the directory containing .npy files with embeddings.
        db (list): List of tuples containing embeddings and their corresponding filenames.
        file_list (set): Set of filenames in the database directory.
    """
    
    def __init__(self, distance, db_path):
        """
        Initialize the One2Many class with the specified distance metric and database path.

        Parameters:
            distance (str): The name of the distance metric ('euclidean' or 'bray_curtis').
            db_path (str): Path to the directory containing .npy files with embeddings.
        """
        self.db_path = db_path
        self.db = self.load_db()
        self.distance = euclidean_distance if distance == 'euclidean' else bray_curtis_distance

    def load_db(self):
        """
        Load embeddings from .npy files in the specified directory.

        Returns:
            list: List of tuples containing embeddings and their corresponding filenames.
        """
        embeddings = []
        self.file_list = set()
        
        for filename in os.listdir(self.db_path):
            if filename.endswith('.npy'):
                filepath = os.path.join(self.db_path, filename)
                embedding = np.load(filepath)
                embeddings.append((embedding, filename.split('.')[0]))
                self.file_list.add(filename)
        
        return embeddings

    def update_db(self):
        """
        Update the database with any new .npy files in the directory.
        """
        new_files = set(os.listdir(self.db_path)) - self.file_list
        
        for filename in new_files:
            if filename.endswith('.npy'):
                filepath = os.path.join(self.db_path, filename)
                embedding = np.load(filepath)
                self.db.append((embedding, filename.split('.')[0]))
                self.file_list.add(filename)

    def compare(self, embedding):
        """
        Compare the given embedding with all embeddings in the database using the specified distance metric.

        Parameters:
            embedding (numpy.ndarray): The embedding to compare.

        Returns:
            list: List of tuples containing distances and corresponding filenames, sorted by distance.
        """
        distances = [(self.distance(embedding, db_embedding[0]), db_embedding[1]) for db_embedding in self.db]
        distances = sorted(distances, key=lambda x: x[0])
        return distances
