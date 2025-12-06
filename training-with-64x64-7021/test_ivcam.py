"""
Test YOLOv8 FER model with iVCam
Automatically detects iVCam and provides better camera control
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

# Robot actions for each emotion
ROBOT_ACTIONS = {
    'happy': '⬆️ FORWARD',
    'angry': '⬇️ BACKWARD',
    'surprised': '⬅️ TURN LEFT',
    'sad': '➡️ TURN RIGHT',
    'neutral': '⏹️ STOP'
}

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_ivcam():
    """Detect iVCam camera index"""
    print("=" * 60)
    print("DETECTING CAMERAS")
    print("=" * 60)
    
    available_cameras = []
    
    for i in range(10):  # Check indices 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Read a test frame
            ret, frame = cap.read()
            if ret:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                # iVCam typically has higher resolution
                camera_info = {
                    'index': i,
                    'width': int(width),
                    'height': int(height),
                    'is_ivcam': width >= 640 and i > 0  # Heuristic: iVCam is usually not index 0
                }
                available_cameras.append(camera_info)
                
                print(f"Camera {i}: {int(width)}×{int(height)} {'(Likely iVCam ✓)' if camera_info['is_ivcam'] else '(Built-in?)'}")
            cap.release()
    
    if not available_cameras:
        print("❌ No cameras detected!")
        return None
    
    print(f"\n✓ Found {len(available_cameras)} camera(s)")
    
    # Find iVCam (prefer higher index with good resolution)
    ivcam_candidates = [c for c in available_cameras if c['is_ivcam']]
    
    if ivcam_candidates:
        # Use the one with highest resolution
        best_cam = max(ivcam_candidates, key=lambda c: c['width'] * c['height'])
        print(f"\n🎥 Auto-selected: Camera {best_cam['index']} (iVCam)")
        return best_cam['index']
    else:
        # Just use first available
        return available_cameras[0]['index']

def test_ivcam(camera_index=None):
    """Test model with iVCam"""
    print("=" * 60)
    print("YOLOV8 FER - iVCAM TEST")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded")
    
    # Detect iVCam if not specified
    if camera_index is None:
        camera_index = detect_ivcam()
        if camera_index is None:
            return
    else:
        print(f"\nUsing camera index: {camera_index}")
    
    # Open camera
    print(f"\nOpening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use DirectShow for better Windows support
    
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        print("\nTrying alternative methods...")
        cap = cv2.VideoCapture(camera_index)  # Try without DSHOW
        
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    # Get actual properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✓ Camera opened successfully")
    print(f"  Resolution: {width}×{height}")
    print(f"  FPS: {fps_cap}")
    
    print("\n" + "=" * 60)
    print("CONTROLS")
    print("=" * 60)
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'c' - Change camera")
    print("  'f' - Toggle face detection")
    print("  '+' - Increase confidence threshold")
    print("  '-' - Decrease confidence threshold")
    print("=" * 60)
    
    fps_time = time.time()
    fps = 0
    show_face_detection = True
    confidence_threshold = 0.5
    
    # Statistics
    emotion_counts = {name: 0 for name in CLASS_NAMES}
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Could not read frame")
            break
        
        frame_count += 1
        
        # Mirror frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        if show_face_detection:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        else:
            # Use whole frame
            faces = [(0, 0, frame.shape[1], frame.shape[0])]
        
        detected_emotions = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
            
            try:
                # Resize to model input size
                face_resized = cv2.resize(face_roi, (64, 64))
                
                # Predict emotion
                results = model(face_resized, verbose=False)
                probs = results[0].probs.data.cpu().numpy()
                
                # Get top prediction
                top1_idx = probs.argmax()
                top1_conf = probs[top1_idx]
                emotion = CLASS_NAMES[top1_idx]
                
                # Only show if confidence is above threshold
                if top1_conf >= confidence_threshold:
                    detected_emotions.append(emotion)
                    emotion_counts[emotion] += 1
                    
                    # Get color and robot action
                    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                    action = ROBOT_ACTIONS.get(emotion, '')
                    
                    # Draw bounding box
                    if show_face_detection:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Draw emotion label with background
                    label = f"{emotion.upper()}: {top1_conf*100:.1f}%"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    # Background rectangle
                    cv2.rectangle(frame, (x, y-35), (x + label_size[0] + 10, y), color, -1)
                    
                    # Text
                    cv2.putText(frame, label, (x+5, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Show robot action
                    action_y = y + h + 25
                    cv2.putText(frame, f"Robot: {action}", (x, action_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Show top 3 predictions (small text)
                    y_offset = action_y + 25
                    for i in np.argsort(probs)[-3:][::-1]:
                        prob_label = f"{CLASS_NAMES[i]}: {probs[i]*100:.0f}%"
                        cv2.putText(frame, prob_label, (x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 20
                
            except Exception as e:
                print(f"⚠ Error: {e}")
                continue
        
        # Calculate FPS
        fps = 1 / (time.time() - fps_time)
        fps_time = time.time()
        
        # Draw info panel (top-left)
        cv2.rectangle(frame, (0, 0), (300, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f} | Camera: {camera_index}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)} | Threshold: {confidence_threshold:.2f}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display instructions (bottom)
        cv2.putText(frame, "q: Quit | s: Save | c: Change cam | f: Toggle face | +/-: Threshold", 
                   (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show current emotion if detected
        if detected_emotions:
            main_emotion = max(set(detected_emotions), key=detected_emotions.count)
            emotion_color = EMOTION_COLORS.get(main_emotion, (255, 255, 255))
            cv2.circle(frame, (width-30, 30), 15, emotion_color, -1)
        
        # Show frame
        cv2.imshow('iVCam Emotion Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ivcam_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Saved: {filename}")
        elif key == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            new_idx = input("\nEnter camera index: ").strip()
            if new_idx.isdigit():
                test_ivcam(int(new_idx))
            return
        elif key == ord('f'):
            show_face_detection = not show_face_detection
            print(f"Face detection: {'ON' if show_face_detection else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"Confidence threshold: {confidence_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            confidence_threshold = max(0.1, confidence_threshold - 0.05)
            print(f"Confidence threshold: {confidence_threshold:.2f}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("SESSION STATISTICS")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {frame_count / (time.time() - fps_time + 1):.1f}")
    print("\nEmotion detections:")
    total = sum(emotion_counts.values())
    if total > 0:
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {emotion:12s}: {count:4d} ({pct:5.1f}%) {bar}")
    print("=" * 60)
    print("✓ Test complete")

if __name__ == "__main__":
    import sys
    
    # Check for command line argument
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        test_ivcam(int(sys.argv[1]))
    else:
        test_ivcam()
