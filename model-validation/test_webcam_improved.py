"""
Real-time webcam test for improved YOLOv8 FER model
Press 'q' to quit
"""

import cv2
import numpy as np
import onnxruntime as ort
import time

# Configuration
MODEL_PATH = "../final-model/fer_yolov8_cls.onnx"
IMG_SIZE = 128
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# Emotion to color mapping
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'happy': (0, 255, 0),      # Green
    'neutral': (128, 128, 128), # Gray
    'sad': (255, 0, 0),        # Blue
    'surprised': (0, 255, 255) # Yellow
}

print("=" * 60)
print("IMPROVED YOLOv8 FER - WEBCAM TEST")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Input size: {IMG_SIZE}×{IMG_SIZE}")
print("Press 'q' to quit")
print("=" * 60)

# Load ONNX model
print("\nLoading model...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Model loaded")

# Load face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img, img_size=IMG_SIZE):
    """Preprocess face for YOLOv8"""
    # Resize
    face_img = cv2.resize(face_img, (img_size, img_size))
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Normalize
    face_rgb = face_rgb.astype(np.float32) / 255.0
    
    # Transpose to CHW
    face_rgb = np.transpose(face_rgb, (2, 0, 1))
    
    # Add batch dimension
    face_rgb = np.expand_dims(face_rgb, axis=0)
    
    return face_rgb

def detect_ivcam():
    """Detect iVCam camera index"""
    print("\nDetecting cameras...")
    
    available_cameras = []
    
    for i in range(10):  # Check indices 0-9
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
        print(f"\niVCam detected at index {selected['index']} ({selected['width']}x{selected['height']})")
        return selected['index']
    elif available_cameras:
        selected = available_cameras[0]
        print(f"\niVCam not found, using camera {selected['index']} ({selected['width']}x{selected['height']})")
        return selected['index']
    else:
        return None

# Detect and open iVCam
camera_index = detect_ivcam()
if camera_index is None:
    print("No cameras detected")
    exit(1)

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow for better Windows support
if not cap.isOpened():
    print("Cannot open camera, trying without DSHOW...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        exit(1)

print("Starting real-time detection...\n")

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
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess
            face_input = preprocess_face(face_roi, IMG_SIZE)
            
            # Predict
            outputs = session.run([output_name], {input_name: face_input})
            probs = outputs[0][0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            emotion = CLASS_NAMES[pred_idx]
            
            # Get color
            color = EMOTION_COLORS[emotion]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label background
            label = f"{emotion.upper()}: {confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
        
        # Calculate FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        fps_list.append(fps)
        
        # Keep only last 30 FPS values
        if len(fps_list) > 30:
            fps_list.pop(0)
        
        avg_fps = np.mean(fps_list)
        
        # Display FPS
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Display info
        info_text = f"Faces: {len(faces)} | Model: {IMG_SIZE}x{IMG_SIZE}"
        cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Improved YOLOv8 FER - Webcam Test', frame)
        
        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("SESSION STATISTICS")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    if fps_list:
        print(f"Average FPS: {np.mean(fps_list):.2f}")
        print(f"Min FPS: {np.min(fps_list):.2f}")
        print(f"Max FPS: {np.max(fps_list):.2f}")
    print("=" * 60)
