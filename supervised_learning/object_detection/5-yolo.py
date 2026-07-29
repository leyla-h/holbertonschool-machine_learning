#!/usr/bin/env python3
"""
Initialize Yolo class for object detection using YOLOv3.
"""

import os
import cv2
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

            sig_xy = 1 / (1 + np.exp(-t_xy))
            b_xy = sig_xy + np.concatenate([c_x, c_y], axis=-1)
            b_xy = b_xy / np.array(
                [grid_width, grid_height], dtype=np.float32
            )

            anchor_w = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes, 1)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes, 1)
            b_wh = np.exp(t_wh) * np.concatenate(
                [anchor_w, anchor_h], axis=-1
            )

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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filter boxes based on box score threshold.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for b, c, p in zip(boxes, box_confidences, box_class_probs):
            scores = c * p

            max_scores = np.max(scores, axis=-1)
            classes = np.argmax(scores, axis=-1)

            mask = max_scores >= self.class_t

            filtered_boxes.append(b[mask])
            box_classes.append(classes[mask])
            box_scores.append(max_scores[mask])

        if filtered_boxes:
            filtered_boxes = np.concatenate(filtered_boxes, axis=0)
            box_classes = np.concatenate(box_classes, axis=0)
            box_scores = np.concatenate(box_scores, axis=0)
        else:
            filtered_boxes = np.array([])
            box_classes = np.array([])
            box_scores = np.array([])

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Perform non-max suppression on filtered boxes.
        """
        unique_classes = np.unique(box_classes)

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            cls_indices = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]
            cls_classes = box_classes[cls_indices]

            sort_indices = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[sort_indices]
            cls_scores = cls_scores[sort_indices]
            cls_classes = cls_classes[sort_indices]

            keep = []
            while len(cls_boxes) > 0:
                keep.append(0)
                if len(cls_boxes) == 1:
                    break

                current_box = cls_boxes[0]
                other_boxes = cls_boxes[1:]

                x1 = np.maximum(current_box[0], other_boxes[:, 0])
                y1 = np.maximum(current_box[1], other_boxes[:, 1])
                x2 = np.minimum(current_box[2], other_boxes[:, 2])
                y2 = np.minimum(current_box[3], other_boxes[:, 3])

                intersection = np.maximum(0, x2 - x1) * \
                    np.maximum(0, y2 - y1)
                current_area = (current_box[2] - current_box[0]) * \
                               (current_box[3] - current_box[1])
                other_area = (other_boxes[:, 2] - other_boxes[:, 0]) * \
                             (other_boxes[:, 3] - other_boxes[:, 1])

                union = current_area + other_area - intersection
                iou = intersection / union

                indices = np.where(iou <= self.nms_t)[0]
                cls_boxes = cls_boxes[indices + 1]
                cls_scores = cls_scores[indices + 1]
                cls_classes = cls_classes[indices + 1]

            if len(keep) > 0:
                box_predictions.append(cls_boxes[keep])
                predicted_box_classes.append(cls_classes[keep])
                predicted_box_scores.append(cls_scores[keep])

        if box_predictions and any(len(b) > 0 for b in box_predictions):
            box_predictions = np.concatenate(box_predictions, axis=0)
            predicted_box_classes = np.concatenate(
                predicted_box_classes, axis=0
            )
            predicted_box_scores = np.concatenate(
                predicted_box_scores, axis=0
            )
        else:
            box_predictions = np.array([])
            predicted_box_classes = np.array([])
            predicted_box_scores = np.array([])

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Load all images from a given folder path.
        """
        images = []
        image_paths = []

        valid_extensions = ['.jpg', '.jpeg', '.png']
        for root, dirs, files in os.walk(folder_path):
            dirs.sort()
            for file in sorted(files):
                if any(
                        file.lower().endswith(ext)
                        for ext in valid_extensions
                ):
                    path = os.path.join(root, file)
                    img = cv2.imread(path)
                    if img is not None:
                        images.append(img)
                        image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes and rescales images for input to the Darknet model.
        """
        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            image_shapes.append(img.shape[:2])

            resized = cv2.resize(
                img,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )
            rescaled = resized / 255.0
            pimages.append(rescaled)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes
