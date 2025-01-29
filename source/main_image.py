import cv2
import numpy as np
import os

# Threshold settings
thres = 0.45  # Confidence threshold
nms_threshold = 0.5  # Non-Maximum Suppression threshold

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height
cap.set(10, 150)  # Brightness

# Load class names
classNames = []
classFile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model configuration and weights
configPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
weightsPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb'))

# Load DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # Capture frame
    success, image = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    # Detect objects
    classIds, confs, bbox = net.detect(image, confThreshold=thres)

    # Ensure proper formatting
    if isinstance(classIds, tuple):  # If no detections, OpenCV returns empty tuple
        classIds = np.array([]) 
        confs = np.array([]) 
        bbox = []

    classIds = classIds.flatten().astype(int) if classIds.size > 0 else []
    confs = confs.flatten() if confs.size > 0 else []
    bbox = list(bbox)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    # Draw bounding boxes if detections exist
    if len(indices) > 0:
        for i in indices.flatten():
            if 0 <= i < len(classIds):  # Validate index
                x, y, w, h = bbox[i]
                class_index = classIds[i] - 1  # Ensure correct label mapping

                # Ensure class_index is within bounds
                if 0 <= class_index < len(classNames):
                    label = classNames[class_index]
                else:
                    label = "Unknown"

                # Draw rectangle and label
                cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
                cv2.putText(image, label, (x + 10, y + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Object Detection", image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
