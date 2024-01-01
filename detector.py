# For machine learning
import torch
# For array computations
import numpy as np
# For image decoding / editing
import cv2
# For environment variables
import os
# For detecting which ML Devices we can use
import platform
# For actually using the YOLO models
from ultralytics import YOLO


class YoloV8ImageObjectDetection:

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.50):
        """Initializes a yolov8 detector

        Arguments:
            model_path (str): A path to a pretrained model file or one on torchub
            conf_threshold (float): Confidence threshold for detections

        Default Model Supports The Following:

        {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
            4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 
            18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 
            26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 
            34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 
            37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
            41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 
            47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 
            52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
            57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
            66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 
            79: 'toothbrush'
        }      
        """
        self.conf_threshold = conf_threshold
        self.model = self._load_model(model_path)
        self.device = self._get_device()
        self.classes = self.model.names

    def _load_model(self, model_path):
        """Loads Yolo8 model from pytorch hub or a path on disk

        Arguments:
            model_path (str):  A path to a pretrained model file or one on torchub
        Returns:
            model (Model) - Trained Pytorch model
        """
        model = YOLO(model_path)
        return model
    
    def _get_device(self):
        """Gets best device for your system

        Returns:
            device (str): The device to use for YOLO for your system
        """
        if platform.system().lower() == "darwin":
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def is_detectable(self, classname):
        """Sees if a desired class is in our models known labels

        Arguments:
            classname (str): Class name to search our model for
        
        Returns:
            exists: True if class is in model, False otherwise
        """
        looking_for = self.classname_to_id(classname)

        if (looking_for < 0):
            return False
        
        return True

    def classname_to_id(self, classname):
        """Searches our model's values for a given class name

        Arguments:
            classname (str): Class name to search our model for
        
        Returns:
            idx (int): Index of the class if it exists, -1 otherwise
        """
        return list(self.classes.keys())[list(self.classes.values()).index(classname)]
    
    def detect(self, frame, classname):
        """Analyze a frame using a YOLOv8 model to find any of the classes
        in question

        Arguments:
            frame (numpy.ndarray): The frame to analyze (from cv2)
            classname (str): Class name to search our model for
        
        Returns:
            plotted (numpy.ndarray): Frame with bounding boxes and labels ploted on it.
            boxes (torch.Tensor): A set of bounding boxes
            tracks (list): A list of box IDs
        """
        looking_for = self.classname_to_id(classname)
        results = self.model.track(frame, persist=True, conf=self.conf_threshold, classes = [looking_for])

        plotted = results[0].plot()
        boxes   = results[0].boxes.xywh.cpu()
        tracks  = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id else []
        return plotted, boxes, tracks 