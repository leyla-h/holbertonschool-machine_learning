#!/usr/bin/env python3
"""
Initialize Yolo class for object detection using YOLOv3.
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    Uses the YOLO v3 algorithm to perform object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor for Yolo.
        """
        self.model = K.models.load_model(model_path)
        
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
            
        self.class_t = float(class_t)
        self.nms_t = float(nms_t)
        self.anchors = anchors
