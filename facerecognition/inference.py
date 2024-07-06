import torch
from facenet_pytorch import InceptionResnetV1

from facerecognition.detector import MyMTCNN
from facerecognition.utils import tensor_to_numpy


class FaceRecognition:
    """
    A class for face detection and embedding extraction using MyMTCNN and InceptionResnetV1.
    
    Attributes:
        device (torch.device): The device to run the model on (GPU if available, else CPU).
        detector (MyMTCNN): An instance of the MyMTCNN face detector.
        embedding (InceptionResnetV1): An instance of the InceptionResnetV1 model for face embedding.
    """

    def __init__(self, **kwargs):
        """
        Initialize the FaceRecognition class with specified keyword arguments.

        Parameters:
            **kwargs: Arbitrary keyword arguments to configure MyMTCNN.
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        kwargs.update({'device': self.device})
        self.detector = MyMTCNN(**kwargs)
        self.embedding = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def detect_faces(self, img, bbox=True):
        """
        Detect faces in the given image.

        Parameters:
            img (numpy.ndarray): The input image.
            bbox (bool): Whether to return bounding boxes of detected faces. Default is True.

        Returns:
            tuple: A tuple containing:
                - cropped_faces (list): A list of cropped face images.
                - bb (list): A list of bounding boxes for the detected faces (if bbox is True).
        """
        cropped_faces, bb = self.detector(img)
        if bbox:
            return cropped_faces, bb
        else:
            return cropped_faces

    def image_embedding(self, img, cropped_face):
        """
        Get the embedding for a single cropped face image.

        Parameters:
            img (numpy.ndarray): The input image.
            cropped_face (numpy.ndarray): The cropped face image.

        Returns:
            numpy.ndarray: The face embedding or None if no face is detected.
        """
        if cropped_face is not None:
            cropped_face = torch.from_numpy(cropped_face).float().permute(2, 0, 1)
            cropped_face = cropped_face.to(self.device)
            face_embedding = self.embedding(cropped_face.unsqueeze(0))
            face_embedding = tensor_to_numpy(face_embedding)
            return face_embedding
        else:
            print("No face detected in the image.")
            return None

    def image_embeddings(self, img, cropped_faces=None):
        """
        Get embeddings for multiple cropped face images.

        Parameters:
            img (numpy.ndarray): The input image.
            cropped_faces (list): List of cropped face images. If not provided, faces will be detected.

        Returns:
            list: A list of face embeddings or None if no face is detected.
        """
        if cropped_faces is None:
            cropped_faces = self.detect_faces(img, bbox=False)
        
        face_embeddings = []
        for cropped_face in cropped_faces:
            if cropped_face is not None:
                cropped_face = torch.from_numpy(cropped_face).float().permute(2, 0, 1)
                cropped_face = cropped_face.to(self.device)
                face_embedding = self.embedding(cropped_face.unsqueeze(0))
                face_embeddings.append(tensor_to_numpy(face_embedding))

        if len(face_embeddings) > 0:
            return face_embeddings
        else:
            print("No face detected in the image.")
            return None
