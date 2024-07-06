import cv2
import uuid
import numpy as np
import pandas as pd
from facerecognition.utils import find_largest_bbox_index
from facerecognition.inference import FaceRecognition


class DBRegistration:
    """
    A class for registering faces in a database.

    Attributes:
        db_path (str): Path to the database where embeddings will be stored.
        raw_db_path (str): Path to the raw images.
        csv (str): Path to the CSV file that records registered faces.
        face_recognition (FaceRecognition): Face recognition model.
    """
    def __init__(self, db_path, raw_db_path, csv, **kwargs):
        """
        Initialize the DBRegistration class.

        Parameters:
            db_path (str): Path to the database where embeddings will be stored.
            raw_db_path (str): Path to the raw images.
            csv (str): Path to the CSV file that records registered faces.
            kwargs: Additional keyword arguments for the FaceRecognition model.
        """
        self.db_path = db_path
        self.raw_db_path = raw_db_path
        self.csv = csv
        self.face_recognition = FaceRecognition(**kwargs)

    def already_exists(self, df, image_name):
        """
        Check if a subject already exists in the database.

        Parameters:
            df (pandas.DataFrame): DataFrame containing the database records.
            image_name (str): Name of the image file.

        Returns:
            bool: True if the subject already exists, False otherwise.
        """
        if df['image_name'].str.contains(image_name).any():
            print("Subject already in database")
            return True
        return False

    def register(self, image_name):
        """
        Register a new subject in the database.

        Parameters:
            image_name (str): Name of the image file.
        """
        df = pd.read_csv(self.csv)
        if not self.already_exists(df, image_name):
            img = cv2.imread(f"{self.raw_db_path}/{image_name}")
            faces, bboxes = self.face_recognition.detect_faces(img)
            if len(bboxes) == 0:
                print("No face detected!")
            else:
                posic = find_largest_bbox_index(bboxes)
                subject_id = uuid.uuid1()
                embedding = self.face_recognition.image_embedding(img, faces[posic])
                np.save(f"{self.db_path}/{subject_id}.npy", embedding)
                new_row_df = pd.DataFrame([{'subject_id': subject_id, 'image_name': image_name}])
                df = pd.concat([df, new_row_df], ignore_index=True)
                df.to_csv(self.csv, index=False)
                print(f"Subject registered with subject id: {subject_id}")
