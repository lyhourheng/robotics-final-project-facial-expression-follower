"""
Test YOLOv8-face ONNX Model on a Single Image
Quick verification without camera
"""

import cv2
import numpy as np
import onnxruntime as ort
import os

print("=" * 60)
print("YOLOV8-FACE ONNX - SINGLE IMAGE TEST")
print("=" * 60)

# Configuration
MODEL_PATH = "../yolov8n-face.onnx"
INPUT_SIZE = 320
CONFIDENCE_THRESHOLD = 0.5

# Check model
if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    exit(1)

print(f"Model: {MODEL_PATH}")

# Load model
print("\nLoading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print("Model loaded")

# Get model info
input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print(f"\nModel Info:")
print(f"  Input: {input_info.name} {input_info.shape}")
print(f"  Output: {output_info.name} {output_info.shape}")

# Create a test image (black with white circle as "face")
print("\nCreating test image...")
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(test_img, (320, 240), 100, (255, 255, 255), -1)
cv2.putText(test_img, "Test Face", (270, 240), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Preprocess
img = cv2.resize(test_img, (INPUT_SIZE, INPUT_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

print(f"Input blob shape: {img.shape}")

# Run inference
print("\nRunning inference...")
outputs = session.run(None, {input_name: img})

print(f"Output shape: {outputs[0].shape}")

# Parse output
predictions = outputs[0][0].T
print(f"Predictions shape: {predictions.shape}")
print(f"Number of predictions: {predictions.shape[0]}")

# Check confidence scores
scores = predictions[:, 4]
high_conf = np.sum(scores > CONFIDENCE_THRESHOLD)

print(f"\nDetections with confidence > {CONFIDENCE_THRESHOLD}: {high_conf}")

if high_conf > 0:
    print("Model is working (detecting something)")
    max_score = np.max(scores)
    print(f"  Max confidence: {max_score:.4f}")
else:
    print("No detections (expected for synthetic test image)")
    print("  This is normal - the model needs real faces")

print("\n" + "=" * 60)
print("MODEL VERIFICATION COMPLETE")
print("=" * 60)
print("\nThe model loaded and ran successfully!")
print("Next step: Test with real camera using test_yolov8_face_onnx.py")
print("=" * 60)
