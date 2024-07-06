# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm
"""
Create patch from original input image by using bbox coordinate.
"""

import cv2
import numpy as np

class CropImage:
    """
    A class used to crop images based on bounding box coordinates.
    """
    
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        """
        Calculate a new bounding box with the specified scale.

        Parameters:
            src_w (int): Source image width.
            src_h (int): Source image height.
            bbox (tuple): Bounding box (x, y, width, height).
            scale (float): Scale factor for resizing the bounding box.

        Returns:
            tuple: New bounding box coordinates (left_top_x, left_top_y, right_bottom_x, right_bottom_y).
        """
        x, y, box_w, box_h = bbox

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1

        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):
        """
        Crop the original image using the bounding box and scale.

        Parameters:
            org_img (numpy.ndarray): Original image.
            bbox (tuple): Bounding box (x, y, width, height).
            scale (float): Scale factor for resizing the bounding box.
            out_w (int): Output width.
            out_h (int): Output height.
            crop (bool): Whether to crop the image. Default is True.

        Returns:
            numpy.ndarray: Cropped and resized image.
        """
        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)

            img = org_img[left_top_y: right_bottom_y + 1, left_top_x: right_bottom_x + 1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img
