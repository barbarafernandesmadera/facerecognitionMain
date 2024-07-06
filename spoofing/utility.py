# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午2:13
# @Author : zhuying
# @Company : Minivision
# @File : utility.py
# @Software : PyCharm

from datetime import datetime
import os

def get_time():
    """
    Get the current time as a formatted string.

    Returns:
        str: Current time formatted as 'YYYY-MM-DD-HH-MM-SS'.
    """
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')

def get_kernel(height, width):
    """
    Calculate the kernel size based on the height and width.

    Parameters:
        height (int): Height of the input.
        width (int): Width of the input.

    Returns:
        tuple: Kernel size.
    """
    kernel_size = ((height + 15) // 16, (width + 15) // 16)
    return kernel_size

def get_width_height(patch_info):
    """
    Extract the width and height from the patch information string.

    Parameters:
        patch_info (str): Patch information string in the format 'HxW'.

    Returns:
        tuple: Width and height as integers.
    """
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input, h_input

def parse_model_name(model_name):
    """
    Parse the model name to extract input dimensions, model type, and scale.

    Parameters:
        model_name (str): Name of the model file.

    Returns:
        tuple: Input height, input width, model type, and scale.
    """
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale

def make_if_not_exist(folder_path):
    """
    Create the directory if it does not exist.

    Parameters:
        folder_path (str): Path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
