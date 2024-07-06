import cv2
import numpy as np
from scipy.spatial.distance import braycurtis


def in_cuda(tensor):
    return tensor.is_cuda

def tensor_to_numpy(tensor):
    if tensor.is_cuda:  # Verificar se o tensor está na GPU
        tensor = tensor.cpu().detach()  # Mover o tensor para a CPU
    return tensor.numpy()  # Converter para NumPy

def calculate_area(bbox):
    """
    Calcula a área de uma bounding box.
    Args:
    bbox (list or tuple): Bounding box na forma [x1, y1, x2, y2]
    
    Returns:
    int: Área da bounding box
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def find_largest_bbox_index(bboxes):
    """
    Encontra a bounding box com a maior área.
    Args:
    bboxes (list of list or tuples): Lista de bounding boxes
    
    Returns:
    list: Bounding box com a maior área
    """
    if not bboxes:
        return None
    
    return max(enumerate(bboxes), key=lambda x: calculate_area(x[1]))[0]


def is_bounding_box_smaller(bbox, kernel_size):
    """
    Check if the bounding box is smaller than the kernel size.

    Parameters:
    bbox (tuple): Coordinates of the bounding box (x_min, y_min, x_max, y_max).
    kernel_size (tuple): Dimensions of the kernel (height, width).

    Returns:
    bool: True if the bounding box is smaller than the kernel size, False otherwise.
    """
    x_min, y_min, x_max, y_max = bbox
    bbox_height = y_max - y_min
    bbox_width = x_max - x_min

    kernel_height, kernel_width = kernel_size
    
    return bbox_height < kernel_height or bbox_width < kernel_width


# Função de distância exemplo (Euclidean)
def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1[0] - embedding2[0])

# Exemplo de uso com distância de Bray-Curtis
def bray_curtis_distance(embedding1, embedding2):
    return braycurtis(embedding1[0], embedding2[0])


def get_frames(path_do_video, frame_skip):
    frames = []
    cap = cv2.VideoCapture(path_do_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print(f"Erro ao abrir o arquivo de vídeo: {path_do_video}")
        return frames
    frame_count = 0
    for frame_number in range(0, total_frames, frame_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frames.append(frame)
    cap.release()

    return frames