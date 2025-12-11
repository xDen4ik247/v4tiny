# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os


def load_models():
    """
    Загрузка YOLOv4-tiny через OpenCV DNN (Darknet format)
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        cfg_path = os.path.join(current_dir, 'yolov4-tiny.cfg')
        weights_path = os.path.join(current_dir, 'yolov4-tiny.weights')
        
        net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        return net
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def detect_cars(image, model) -> list:
    """
    Детектирование автомобиля на изображении.
    """
    default_box = [0, 0, 0, 0]
    
    if model is None:
        return default_box
    
    try:
        net = model
        orig_h, orig_w = image.shape[:2]
        
        # Create blob (416x416 is standard for YOLOv4-tiny)
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1/255.0,
            size=(416, 416),
            swapRB=True,
            crop=False
        )
        
        net.setInput(blob)
        
        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        # Run inference
        outputs = net.forward(output_layers)
        
        # COCO classes: car=2, bus=5, truck=7
        vehicle_classes = {2, 5, 7}
        
        best_box = None
        best_score = 0.0
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if class_id in vehicle_classes and confidence > best_score:
                    best_score = confidence
                    
                    # YOLOv4 outputs: center_x, center_y, width, height (normalized)
                    cx = detection[0]
                    cy = detection[1]
                    w = detection[2]
                    h = detection[3]
                    
                    # Convert to pixel coordinates
                    x1 = int((cx - w / 2) * orig_w)
                    y1 = int((cy - h / 2) * orig_h)
                    x2 = int((cx + w / 2) * orig_w)
                    y2 = int((cy + h / 2) * orig_h)
                    
                    # Clamp
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(orig_w, x2)
                    y2 = min(orig_h, y2)
                    
                    best_box = [x1, y1, x2, y2]
        
        if best_box is not None:
            return best_box
        
        return default_box
        
    except Exception as e:
        print(f"Error: {e}")
        return default_box

