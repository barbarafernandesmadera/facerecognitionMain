import os
import torch
import torch.nn.functional as F
from .MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
import spoofing.transform as trans
from spoofing.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

class AntiSpoofPredict:
    """
    A class for predicting spoofing using various MiniFASNet models.
    """
    def __init__(self):
        """
        Initialize the AntiSpoofPredict class.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_path):
        """
        Load the model from the specified path.

        Parameters:
            model_path (str): Path to the model file.
        """
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = next(keys)
        if 'module.' in first_layer_name:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def predict(self, img, model_path):
        """
        Predict whether the given image is a spoof or not using the specified model.

        Parameters:
            img (PIL.Image or numpy.ndarray): Input image.
            model_path (str): Path to the model file.

        Returns:
            numpy.ndarray: Prediction probabilities.
        """
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result
