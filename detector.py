from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8l.pt')

def detect_objects(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    frame_height, frame_width = frame.shape[:2]

    results = model(frame, conf=0.55)  # raised from 0.4 → 0.55

    detected  = []
    seen_labels = {}

    for result in results:
        for box in result.boxes:
            label      = result.names[int(box.cls)]
            confidence = float(box.conf)
            coords     = box.xyxy[0].tolist()

            # Keep only highest confidence instance per label
            if label not in seen_labels or confidence > seen_labels[label]['confidence']:
                seen_labels[label] = {
                    'label':      label,
                    'confidence': round(confidence * 100, 1),
                    'box':        coords
                }

    detected = list(seen_labels.values())
    return detected, frame, frame_width, frame_height