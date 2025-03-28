import cv2
import numpy as np
import socket
import struct
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import torch
import time

# ✅ TCP Connection (MATLAB communication)
HOST = '127.0.0.1'     # MATLAB IP
PORT = 5005             # Port to communicate with MATLAB
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

# ✅ Define the Custom Classes to Pick
custom_pick_classes = [
    'Mask', 'can', 'cellphone', 'electronics', 'gbottle',
    'glove', 'metal', 'misc', 'net', 'pbag',
    'pbottle', 'plastic', 'rod', 'sunglasses', 'tire'
]

# ✅ Load YOLO Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_pretrained = YOLO('../models/yolov8n.pt').to(device)
model_custom = YOLO('../models/best.pt').to(device)

# ✅ Open Webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit...")

# ✅ Real-Time Detection Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # ✅ YOLO Inference
    results1 = model_pretrained(frame)[0]
    results2 = model_custom(frame)[0]

    # ✅ Prepare Detections
    boxes1, scores1, labels1 = [], [], []
    boxes2, scores2, labels2 = [], [], []

    # ✅ Pretrained Model Detections
    for det in results1.boxes.data.cpu().numpy():
        xmin, ymin, xmax, ymax, conf, cls = det
        boxes1.append([xmin, ymin, xmax, ymax])
        scores1.append(conf)
        labels1.append(int(cls))

    # ✅ Custom Model Detections
    for det in results2.boxes.data.cpu().numpy():
        xmin, ymin, xmax, ymax, conf, cls = det
        boxes2.append([xmin, ymin, xmax, ymax])
        scores2.append(conf)
        labels2.append(int(cls))

    # ✅ Normalize the Boxes
    h, w = frame.shape[:2]
    boxes1 = [[x/w, y/h, x2/w, y2/h] for x, y, x2, y2 in boxes1]
    boxes2 = [[x/w, y/h, x2/w, y2/h] for x, y, x2, y2 in boxes2]

    # ✅ Apply Weighted Box Fusion (WBF)
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [boxes1, boxes2],
        [scores1, scores2],
        [labels1, labels2],
        iou_thr=0.5,              # Intersection-over-Union threshold
        skip_box_thr=0.01         # Minimum confidence threshold
    )

    # ✅ Draw Fused Boxes with Accuracy Only
    for (xmin, ymin, xmax, ymax), score, label in zip(fused_boxes, fused_scores, fused_labels):
        xmin, ymin, xmax, ymax = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)

        # ✅ Determine if it is a garbage class
        is_custom_model = label in labels2
        custom_label_name = results2.names[label] if is_custom_model else results1.names[label]
        is_pickable = custom_label_name in custom_pick_classes

        # ✅ Color Coding
        color = (0, 255, 0) if is_custom_model and is_pickable else (0, 0, 255)

        # ✅ Draw Box and Display Confidence Only
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f'{score:.2%}', (xmin, ymin - 5),  # Display accuracy only
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ✅ Send Message to MATLAB
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        x_norm = x_center / w
        y_norm = y_center / h
        data = struct.pack('ff', x_norm, y_norm)
        sock.sendall(data)

    # ✅ Display the Fused Detection
    cv2.imshow('Fused Detection (Webcam)', frame)

    # ✅ Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Clean up
cap.release()
cv2.destroyAllWindows()
sock.close()
