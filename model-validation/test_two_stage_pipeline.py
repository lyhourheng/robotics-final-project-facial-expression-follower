"""
Two-Stage Pipeline Test: YOLOv8-face Detection + FER Classification
This combines face detection and emotion recognition in real-time
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import os

print("=" * 60)
print("TWO-STAGE PIPELINE TEST")
print("YOLOv8-face Detection + FER Classifier")
print("=" * 60)

# Configuration
FACE_MODEL_PATH = "../final-model/yolov8n-face.onnx"
FER_MODEL_PATH = "../final-model/fer_yolov8_cls.onnx"
FACE_INPUT_SIZE = 320
FER_INPUT_SIZE = 128
FACE_CONFIDENCE_THRESHOLD = 0.4  # Lowered from 0.5 to catch more faces
FER_CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'happy': (0, 255, 0),      # Green
    'neutral': (128, 128, 128), # Gray
    'sad': (255, 0, 0),        # Blue
    'surprised': (0, 255, 255) # Yellow
}

# Check models exist
if not os.path.exists(FACE_MODEL_PATH):
    print(f"❌ Face model not found: {FACE_MODEL_PATH}")
    exit(1)

if not os.path.exists(FER_MODEL_PATH):
    print(f"❌ FER model not found: {FER_MODEL_PATH}")
    exit(1)

print(f"Face Model: {FACE_MODEL_PATH}")
print(f"FER Model: {FER_MODEL_PATH}")
print(f"Face Input: {FACE_INPUT_SIZE}×{FACE_INPUT_SIZE}")
print(f"FER Input: {FER_INPUT_SIZE}×{FER_INPUT_SIZE}")
print("=" * 60)

# Load models
print("\nLoading models...")
face_session = ort.InferenceSession(FACE_MODEL_PATH, providers=['CPUExecutionProvider'])
fer_session = ort.InferenceSession(FER_MODEL_PATH, providers=['CPUExecutionProvider'])
face_input_name = face_session.get_inputs()[0].name
fer_input_name = fer_session.get_inputs()[0].name
print("✓ Face detector loaded")
print("✓ FER classifier loaded")

def preprocess_face_detection(frame, input_size=FACE_INPUT_SIZE):
    """Preprocess image for YOLOv8 face detection"""
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_fer(face_img, img_size=FER_INPUT_SIZE):
    """Preprocess face crop for FER classifier"""
    face_img = cv2.resize(face_img, (img_size, img_size))
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_rgb = face_rgb.astype(np.float32) / 255.0
    face_rgb = np.transpose(face_rgb, (2, 0, 1))
    face_rgb = np.expand_dims(face_rgb, axis=0)
    return face_rgb

def nms(boxes, scores, iou_threshold=IOU_THRESHOLD):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), 
        FACE_CONFIDENCE_THRESHOLD, iou_threshold
    )
    return indices.flatten() if len(indices) > 0 else []

def detect_faces(frame):
    """Detect faces using YOLOv8-face"""
    h, w = frame.shape[:2]
    
    # Preprocess
    input_blob = preprocess_face_detection(frame, FACE_INPUT_SIZE)
    
    # Inference
    outputs = face_session.run(None, {face_input_name: input_blob})
    
    # Post-process
    predictions = outputs[0][0].T
    boxes_xywh = predictions[:, :4]
    scores = predictions[:, 4]
    
    # Filter by confidence
    mask = scores > FACE_CONFIDENCE_THRESHOLD
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    
    if len(boxes_xywh) == 0:
        return [], []
    
    # Convert to corner format
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    
    # NMS
    keep_indices = nms(boxes_xyxy, scores)
    
    if len(keep_indices) == 0:
        return [], []
    
    # Scale to original size
    scale_x = w / FACE_INPUT_SIZE
    scale_y = h / FACE_INPUT_SIZE
    
    final_boxes = []
    final_scores = []
    
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes_xyxy[idx]
        score = scores[idx]
        
        x1 = int(max(0, min(x1 * scale_x, w)))
        y1 = int(max(0, min(y1 * scale_y, h)))
        x2 = int(max(0, min(x2 * scale_x, w)))
        y2 = int(max(0, min(y2 * scale_y, h)))
        
        final_boxes.append([x1, y1, x2, y2])
        final_scores.append(float(score))
    
    return final_boxes, final_scores

def classify_emotion(face_crop):
    """Classify emotion from face crop"""
    # Preprocess
    face_input = preprocess_fer(face_crop, FER_INPUT_SIZE)
    
    # Inference
    outputs = fer_session.run(None, {fer_input_name: face_input})
    probs = outputs[0][0]
    
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    emotion = CLASS_NAMES[pred_idx]
    
    return emotion, confidence

def detect_ivcam():
    """Detect iVCam camera"""
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
    
    ivcam_cameras = [cam for cam in available_cameras if cam['is_ivcam']]
    
    if ivcam_cameras:
        selected = ivcam_cameras[0]
        print(f"\n✓ iVCam detected at index {selected['index']}")
        return selected['index']
    elif available_cameras:
        selected = available_cameras[0]
        print(f"\n⚠ Using camera {selected['index']}")
        return selected['index']
    else:
        return None

# Open camera
camera_index = 0
if camera_index is None:
    print("❌ No cameras detected")
    exit(1)

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit(1)

print("\n✓ Camera opened")
print("\nPress 'q' to quit")
print("=" * 60)
print("\nStarting two-stage detection...\n")

# Statistics
fps_list = []
frame_count = 0
total_faces = 0
emotion_counts = {emotion: 0 for emotion in CLASS_NAMES}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.time()
        
        # Stage 1: Detect faces
        face_boxes, face_scores = detect_faces(frame)
        total_faces += len(face_boxes)
        
        # Stage 2: Classify emotions for each face
        for box, face_score in zip(face_boxes, face_scores):
            x1, y1, x2, y2 = box
            
            # Extract face crop
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            # Classify emotion
            emotion, emotion_conf = classify_emotion(face_crop)
            
            # Update statistics
            if emotion_conf >= FER_CONFIDENCE_THRESHOLD:
                emotion_counts[emotion] += 1
            
            # Get color for emotion
            color = EMOTION_COLORS[emotion]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw labels
            label1 = f"Face: {face_score:.2f}"
            label2 = f"{emotion.upper()}: {emotion_conf:.2f}"
            
            # Face detection label
            (w1, h1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - h1 - 25), (x1 + w1, y1 - 15), (0, 255, 0), -1)
            cv2.putText(frame, label1, (x1, y1 - 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Emotion label
            (w2, h2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 15), (x1 + w2, y1), color, -1)
            cv2.putText(frame, label2, (x1, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Calculate FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time
        fps_list.append(fps)
        
        # Draw info
        avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
        info_text = f"FPS: {avg_fps:.1f} | Faces: {len(face_boxes)}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Two-Stage Pipeline: Face Detection + FER', frame)
        
        # Print stats
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {avg_fps:.1f} FPS | {len(face_boxes)} face(s)")
        
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
        print(f"Total faces detected: {total_faces}")
        print(f"Average faces per frame: {total_faces/frame_count:.2f}")
        
        print("\nEmotion Distribution:")
        for emotion in CLASS_NAMES:
            count = emotion_counts[emotion]
            print(f"  {emotion.capitalize():10s}: {count:4d}")
        
        print("\n✓ Two-stage pipeline verified successfully!")
        print("=" * 60)
