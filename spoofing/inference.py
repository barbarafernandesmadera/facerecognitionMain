import os
import numpy as np
from spoofing.predict import AntiSpoofPredict
from spoofing.generate_patches import CropImage
from spoofing.utility import parse_model_name

class Spoofing:
    """
    A class to handle spoofing detection using pre-trained models.
    """
    def __init__(self, model_dir):
        """
        Initialize the Spoofing class.

        Parameters:
            model_dir (str): Directory containing the pre-trained models.
        """
        self.model = AntiSpoofPredict()
        self.image_cropper = CropImage()
        self.model_dir = model_dir

    def check_image(self, image):
        """
        Check if the image has the appropriate aspect ratio (4:3).

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            bool: True if the image has the appropriate aspect ratio, False otherwise.
        """
        height, width, _ = image.shape
        if width / height != 3 / 4:
            print(f"Image aspect ratio is {width/height:.2f}, expected 4/3.")
            print("Image is not appropriate! Height/Width should be 4/3.")
            return False
        else:
            return True

    def eval_spoofing(self, image, image_bbox):
        """
        Evaluate the spoofing probability of the given image.

        Parameters:
            image (numpy.ndarray): Input image.
            image_bbox (tuple): Bounding box coordinates (x, y, width, height).

        Returns:
            float: Probability of the image being genuine.
        """
        prediction = np.zeros((1, 3))
        # Sum the prediction from single model's result
        for model_name in os.listdir(self.model_dir):
            h_input, w_input, _, scale = parse_model_name(model_name)
            crop = False if scale is None else True
            param = {
                "org_img": image, "bbox": image_bbox, "scale": scale,
                "out_w": w_input, "out_h": h_input, "crop": crop,
            }
            img = self.image_cropper.crop(**param)
            pred = self.model.predict(img, os.path.join(self.model_dir, model_name))
            prediction += pred

        label = 1  # Position of the genuine class
        value = prediction[0][label] / 2  # Probability of being genuine

        return value

    def is_spoofing(self, image, image_bbox, th):
        """
        Determine if the image is spoofing based on a threshold.

        Parameters:
            image (numpy.ndarray): Input image.
            image_bbox (tuple): Bounding box coordinates (x, y, width, height).
            th (float): Threshold for spoofing detection.

        Returns:
            bool: True if the image is spoofing, False otherwise.
        """
        value = self.eval_spoofing(image, image_bbox)
        return value <= th
