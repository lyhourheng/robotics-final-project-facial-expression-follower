"""
Test YOLOv8-face ONNX Model
Verify face detection works correctly with ONNX Runtime
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import os

print("=" * 60)
print("YOLOV8-FACE ONNX MODEL TEST")
print("=" * 60)

# Configuration
MODEL_PATH = "../final-model/yolov8n-face.onnx"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
INPUT_SIZE = 320 

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    exit(1)

print(f"Model: {MODEL_PATH}")
print(f"Input size: {INPUT_SIZE}×{INPUT_SIZE}")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"IOU threshold: {IOU_THRESHOLD}")
print("=" * 60)

# Load ONNX model
print("\nLoading ONNX model...")
try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("✓ Model loaded successfully")
    
    # Get model input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"\nModel Info:")
    print(f"  Input name: {input_info.name}")
    print(f"  Input shape: {input_info.shape}")
    print(f"  Output name: {output_info.name}")
    print(f"  Output shape: {output_info.shape}")
    
    input_name = input_info.name
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

def preprocess_image(frame, input_size=INPUT_SIZE):
    """Preprocess image for YOLOv8"""
    # Resize to input size
    img = cv2.resize(frame, (input_size, input_size))
    
    # Convert BGR to RGB and normalize to [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # Transpose to CHW format (channels first)
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def nms(boxes, scores, iou_threshold=IOU_THRESHOLD):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), 
        scores.tolist(), 
        CONFIDENCE_THRESHOLD, 
        iou_threshold
    )
    
    return indices.flatten() if len(indices) > 0 else []

def postprocess_detections(outputs, frame_shape, input_size=INPUT_SIZE):
    """Post-process YOLOv8 outputs to get face boxes"""
    h, w = frame_shape[:2]
    
    # YOLOv8 output format: [batch, 84, num_predictions]
    # Need to transpose to [num_predictions, 84]
    predictions = outputs[0][0].T
    
    # Extract boxes and confidence scores
    # Format: [x_center, y_center, width, height, confidence, ...]
    boxes_xywh = predictions[:, :4]
    scores = predictions[:, 4]
    
    # Filter by confidence
    mask = scores > CONFIDENCE_THRESHOLD
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    
    if len(boxes_xywh) == 0:
        return [], []
    
    # Convert from center format to corner format
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2
    
    # Apply NMS
    keep_indices = nms(boxes_xyxy, scores)
    
    if len(keep_indices) == 0:
        return [], []
    
    # Scale boxes to original frame size
    scale_x = w / input_size
    scale_y = h / input_size
    
    final_boxes = []
    final_scores = []
    
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        score = scores[idx]
        
        # Scale to original size
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        
        # Clip to frame boundaries
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        final_boxes.append([x1, y1, x2, y2])
        final_scores.append(float(score))
    
    return final_boxes, final_scores

def detect_ivcam():
    """Detect iVCam camera index"""
    print("\nDetecting cameras...")
    
    available_cameras = []
    
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                camera_info = {
                    'index': i,
                    'width': int(width),
                    'height': int(height),
                    'is_ivcam': width >= 640 and i > 0
                }
                available_cameras.append(camera_info)
                
                print(f"  Camera {i}: {int(width)}x{int(height)}")
            cap.release()
    
    # Find iVCam
    ivcam_cameras = [cam for cam in available_cameras if cam['is_ivcam']]
    
    if ivcam_cameras:
        selected = ivcam_cameras[0]
        print(f"\n✓ iVCam detected at index {selected['index']} ({selected['width']}x{selected['height']})")
        return selected['index']
    elif available_cameras:
        selected = available_cameras[0]
        print(f"\n⚠ iVCam not found, using camera {selected['index']} ({selected['width']}x{selected['height']})")
        return selected['index']
    else:
        return None
        

# Detect and open camera
camera_index = detect_ivcam()
if camera_index is None:
    print("❌ No cameras detected")
    exit(1)

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Cannot open camera, trying without DSHOW...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit(1)

print("\n✓ Camera opened")
print("\nPress 'q' to quit")
print("=" * 60)
print("\nStarting face detection...\n")

# FPS calculation
fps_list = []
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Preprocess
        input_blob = preprocess_image(frame, INPUT_SIZE)
        
        # Run inference
        outputs = session.run(None, {input_name: input_blob})
        
        # Post-process
        boxes, scores = postprocess_detections(outputs, frame.shape, INPUT_SIZE)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time
        fps_list.append(fps)
        
        # Draw results
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Face: {score:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw FPS and detection info
        avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
        info_text = f"FPS: {avg_fps:.1f} | Faces: {len(boxes)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('YOLOv8-face ONNX Test', frame)
        
        # Print stats every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {avg_fps:.1f} FPS | {len(boxes)} face(s) detected")
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    if fps_list:
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {np.mean(fps_list):.2f}")
        print(f"Min FPS: {np.min(fps_list):.2f}")
        print(f"Max FPS: {np.max(fps_list):.2f}")
        print(f"\n✓ YOLOv8-face ONNX model verified successfully!")
        print("=" * 60)
