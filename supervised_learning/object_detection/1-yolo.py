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

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(confidence)

            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)

            c_x = np.arange(grid_width)
            c_y = np.arange(grid_height)
            c_x = np.expand_dims(c_x, axis=0)
            c_x = np.repeat(c_x, grid_height, axis=0)
            c_x = np.expand_dims(c_x, axis=2)
            c_x = np.repeat(c_x, anchor_boxes, axis=2)
            c_x = np.expand_dims(c_x, axis=3)

            c_y = np.expand_dims(c_y, axis=1)
            c_y = np.repeat(c_y, grid_width, axis=1)
            c_y = np.expand_dims(c_y, axis=2)
            c_y = np.repeat(c_y, anchor_boxes, axis=2)
            c_y = np.expand_dims(c_y, axis=3)

            b_xy = (1 / (1 + np.exp(-t_xy))) + np.concatenate([c_x, c_y], axis=-1)
            b_xy = b_xy / np.array([grid_width, grid_height], dtype=np.float32)

            anchor_w = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes, 1)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes, 1)
            b_wh = np.exp(t_wh) * np.concatenate([anchor_w, anchor_h], axis=-1)

            input_shape = np.array(
                [self.model.input.shape[1], self.model.input.shape[2]],
                dtype=np.float32
            )
            b_wh = b_wh / input_shape

            b_x1y1 = b_xy - (b_wh / 2)
            b_x2y2 = b_xy + (b_wh / 2)

            processed_boxes = np.concatenate([b_x1y1, b_x2y2], axis=-1)

            orig_height, orig_width = image_size[0], image_size[1]
            processed_boxes[..., 0] *= orig_width
            processed_boxes[..., 1] *= orig_height
            processed_boxes[..., 2] *= orig_width
            processed_boxes[..., 3] *= orig_height

            boxes.append(processed_boxes)

        return boxes, box_confidences, box_class_probs
