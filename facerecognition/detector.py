from facenet_pytorch import MTCNN

class MyMTCNN:
    """
    A wrapper class for the MTCNN face detector from the facenet_pytorch library.

    Attributes:
        mtcnn (MTCNN): An instance of the MTCNN face detector.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the MyMTCNN class with the given arguments and keyword arguments for the MTCNN instance.

        Parameters:
            *args: Variable length argument list for MTCNN.
            **kwargs: Arbitrary keyword arguments for MTCNN.
        """
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, image):
        """
        Detect faces in the given image and return cropped face images and bounding boxes.

        Parameters:
            image (numpy.ndarray): The input image in which to detect faces.

        Returns:
            tuple: A tuple containing:
                - images (list): A list of cropped face images.
                - bboxes (list): A list of bounding boxes for the detected faces.
        """
        boxes, _ = self.mtcnn.detect(image)
        images = []
        bboxes = []
        if boxes is not None:
            for box in boxes:
                int_box = [int(b) for b in box]
                images.append(image[int_box[1]:int_box[3], int_box[0]:int_box[2]])
                bboxes.append(box)

        return images, bboxes
