#!/usr/bin/env python3
"""Yolo class - task 5: preprocess_images"""
import numpy as np
import tensorflow.keras as K
import cv2
import glob
import os


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path: path to Darknet Keras model
        classes_path: path to list of class names used for the model
        class_t: box score threshold for initial filtering
        nms_t: IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        outputs: list of numpy.ndarrays containing predictions
        image_size: numpy.ndarray containing image's original size
            [image_height, image_width]

        Returns: tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            box = np.zeros(output[..., 0:4].shape)

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.tile(np.arange(grid_width), grid_height)
            c_x = c_x.reshape(grid_height, grid_width, 1)

            c_y = np.tile(np.arange(grid_width), grid_height)
            c_y = c_y.reshape(grid_width, grid_height).T.reshape(
                grid_height, grid_width, 1)

            b_x = (self.sigmoid(t_x) + c_x) / grid_width
            b_y = (self.sigmoid(t_y) + c_y) / grid_height

            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            input_width = self.model.input.shape[1]
            input_height = self.model.input.shape[2]

            b_w = (anchor_width * np.exp(t_w)) / input_width
            b_h = (anchor_height * np.exp(t_h)) / input_height

            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boxes based on box score threshold

        Returns: tuple of (filtered_boxes, box_classes, box_scores)
        """
        box_scores_list = []
        box_classes_list = []
        boxes_list = []

        for box, box_confidence, box_class_prob in zip(
                boxes, box_confidences, box_class_probs):
            box_scores = box_confidence * box_class_prob
            box_class = np.argmax(box_scores, axis=-1)
            box_score = np.max(box_scores, axis=-1)

            boxes_list.append(box.reshape(-1, 4))
            box_classes_list.append(box_class.reshape(-1))
            box_scores_list.append(box_score.reshape(-1))

        boxes_all = np.concatenate(boxes_list, axis=0)
        box_classes_all = np.concatenate(box_classes_list, axis=0)
        box_scores_all = np.concatenate(box_scores_list, axis=0)

        filtering_mask = box_scores_all >= self.class_t

        filtered_boxes = boxes_all[filtering_mask]
        box_classes = box_classes_all[filtering_mask]
        box_scores = box_scores_all[filtering_mask]

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-max suppression to filtered boxes

        Returns: tuple of (box_predictions, predicted_box_classes,
                            predicted_box_scores)
        """
        box_predictions_list = []
        predicted_box_classes_list = []
        predicted_box_scores_list = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]

            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            keep = []

            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)

                inter = w * h
                union = areas[i] + areas[order[1:]] - inter
                iou = inter / union

                inds = np.where(iou <= self.nms_t)[0]
                order = order[inds + 1]

            keep = np.array(keep)

            box_predictions_list.append(cls_boxes[keep])
            predicted_box_classes_list.append(
                np.full((len(keep),), cls, dtype=box_classes.dtype))
            predicted_box_scores_list.append(cls_scores[keep])

        box_predictions = np.concatenate(box_predictions_list, axis=0)
        predicted_box_classes = np.concatenate(
            predicted_box_classes_list, axis=0)
        predicted_box_scores = np.concatenate(
            predicted_box_scores_list, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        folder_path: path to a folder holding all images to load

        Returns: tuple of (images, image_paths)
        """
        image_paths = glob.glob(os.path.join(folder_path, '*'))
        images = []

        for path in image_paths:
            image = cv2.imread(path)
            images.append(image)

        return images, image_paths

    def preprocess_images(self, images):
        """
        images: a list of images as numpy.ndarrays

        Returns: tuple of (pimages, image_shapes)
            pimages: numpy.ndarray of shape (ni, input_h, input_w, 3)
                containing all preprocessed images
            image_shapes: numpy.ndarray of shape (ni, 2) containing the
                original height and width of the images
        """
        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]

        pimages_list = []
        image_shapes_list = []

        for img in images:
            image_shapes_list.append(img.shape[:2])

            resized = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)

            rescaled = resized / 255.0

            pimages_list.append(rescaled)

        pimages = np.array(pimages_list)
        image_shapes = np.array(image_shapes_list)

        return pimages, image_shapes
