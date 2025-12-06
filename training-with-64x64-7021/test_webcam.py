"""
Real-time emotion detection with webcam
Tests the model in real-world conditions
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Configuration
MODEL_PATH = "fer_yolov8_cls_best.pt"
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# Colors for each emotion (BGR format)
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'happy': (0, 255, 0),       # Green
    'neutral': (255, 255, 0),   # Cyan
    'sad': (255, 0, 0),         # Blue
    'surprised': (0, 165, 255)  # Orange
}

# Load face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def test_webcam(camera_index=None):
    """Test model with webcam feed"""
    print("=" * 60)
    print("REAL-TIME EMOTION DETECTION")
    print("=" * 60)
    
    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded")
    
    # Detect available cameras if not specified
    if camera_index is None:
        print("\nDetecting available cameras...")
        available_cameras = []
        for i in range(5):  # Check first 5 indices
            cap_test = cv2.VideoCapture(i)
            if cap_test.isOpened():
                available_cameras.append(i)
                cap_test.release()
        
        if not available_cameras:
            print("ERROR: No cameras detected")
            return
        
        print(f"Available camera indices: {available_cameras}")
        
        # If multiple cameras, let user choose
        if len(available_cameras) > 1:
            print("\nAvailable cameras:")
            for idx in available_cameras:
                print(f"  {idx}: Camera {idx} {'(iVCam?)' if idx > 0 else '(Built-in?)'}")
            
            choice = input(f"\nSelect camera index (default: {available_cameras[0]}): ").strip()
            camera_index = int(choice) if choice else available_cameras[0]
        else:
            camera_index = available_cameras[0]
    
    # Open selected camera
    print(f"\nOpening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_index}")
        print("Trying iVCam common indices (1, 2, 3)...")
        for idx in [1, 2, 3]:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                camera_index = idx
                print(f"✓ Successfully opened camera {idx}")
                break
        else:
            print("ERROR: Could not open any camera")
            return
    
    # Set camera properties for better quality (if available)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"✓ Camera {camera_index} opened")
    print(f"  Resolution: {int(actual_width)}×{int(actual_height)}")
    print("\nInstructions:")
    print("  • Press 'q' to quit")
    print("  • Press 's' to save screenshot")
    print("  • Press 'c' to change camera")
    print("  • Try different emotions!")
    
    fps_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame")
            break
        
        # Mirror frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            try:
                # Resize to model input size (64x64)
                face_resized = cv2.resize(face_roi, (64, 64))
                
                # Predict emotion
                results = model(face_resized, verbose=False)
                probs = results[0].probs.data.cpu().numpy()
                
                # Get top prediction
                top1_idx = probs.argmax()
                top1_conf = probs[top1_idx]
                emotion = CLASS_NAMES[top1_idx]
                
                # Get color for emotion
                color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Draw emotion label with background
                label = f"{emotion.upper()}: {top1_conf*100:.1f}%"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Background rectangle
                cv2.rectangle(frame, (x, y-30), (x + label_size[0], y), color, -1)
                
                # Text
                cv2.putText(frame, label, (x, y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show top 3 predictions (small text)
                y_offset = y + h + 20
                for i in np.argsort(probs)[-3:][::-1]:
                    prob_label = f"{CLASS_NAMES[i]}: {probs[i]*100:.0f}%"
                    cv2.putText(frame, prob_label, (x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 15
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        # Calculate FPS
        fps = 1 / (time.time() - fps_time)
        fps_time = time.time()
        
        # Display FPS and camera info
        cv2.putText(frame, f"FPS: {fps:.1f} | Camera: {camera_index}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to save | 'c' to change camera", (10, frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Emotion Detection - iVCam/Webcam Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved screenshot: {filename}")
        elif key == ord('c'):
            # Change camera
            cap.release()
            cv2.destroyAllWindows()
            print("\n" + "="*60)
            print("Switching camera...")
            print("="*60)
            new_index = input("Enter camera index (0-4): ").strip()
            if new_index.isdigit():
                test_webcam(int(new_index))
            return
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Webcam test complete")

if __name__ == "__main__":
    test_webcam()
