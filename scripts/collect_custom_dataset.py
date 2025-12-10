"""
Custom Emotion Dataset Collection Tool
Captures images from webcam with emotion labels for training.
Uses YOLOv8-face for robust face detection.
"""

import cv2
import os
import numpy as np
from datetime import datetime
import argparse
import onnxruntime as ort


class EmotionDataCollector:
    def __init__(self, output_dir='datasets/custom_emotions', resolution=(640, 480), camera_source=0, save_hires=True, target_size=224):
        self.output_dir = output_dir
        self.resolution = resolution
        self.camera_source = camera_source  # Can be int (webcam ID) or string (IP cam URL)
        self.save_hires = save_hires  # Save high-res version alongside target size
        self.target_size = target_size  # 224 for YOLOv8-cls, 48 for legacy FER2013
        
        # 5 emotion classes matching FER2013
        self.emotions = ['angry', 'happy', 'neutral', 'sad', 'surprised']
        self.current_emotion = 'neutral'
        self.current_emotion_idx = 2
        
        # Collection settings
        self.frame_skip = 15  # Capture every 15 frames (avoid duplicates)
        self.frame_counter = 0
        self.captured_counts = {emotion: 0 for emotion in self.emotions}
        
        # Setup directories
        self._setup_directories()
        
        # YOLOv8-face detection setup
        self.yolo_face_model = '../yolov8n-face.onnx'
        self.face_input_size = 320
        self.face_confidence_threshold = 0.4
        self.iou_threshold = 0.45
        
        # Load YOLOv8-face model
        if not os.path.exists(self.yolo_face_model):
            print(f"❌ Error: {self.yolo_face_model} not found!")
            print("   Please ensure yolov8n-face.onnx is in the parent directory")
            raise FileNotFoundError(f"YOLOv8-face model not found at {self.yolo_face_model}")
        
        print("Loading YOLOv8-face model...")
        self.face_session = ort.InferenceSession(self.yolo_face_model, providers=['CPUExecutionProvider'])
        self.face_input_name = self.face_session.get_inputs()[0].name
        print("✓ YOLOv8-face model loaded")
        
        self.detect_face = True
        
        print("="*60)
        print("EMOTION DATASET COLLECTION TOOL")
        print("Using YOLOv8-face for robust face detection")
        print("="*60)
        print(f"Output directory: {self.output_dir}")
        print(f"Emotions: {self.emotions}")
        print("\nControls:")
        print("  1-5: Switch emotion (1=angry, 2=happy, 3=neutral, 4=sad, 5=surprised)")
        print("  SPACE: Capture current frame")
        print("  A: Auto-capture mode (captures automatically)")
        print("  F: Toggle face detection")
        print("  Q/ESC: Quit")
        print("="*60)
    
    def _setup_directories(self):
        """Create output directories for each emotion."""
        for emotion in self.emotions:
            emotion_dir = os.path.join(self.output_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            
            # Create high-res subdirectory if enabled
            if self.save_hires:
                hires_dir = os.path.join(emotion_dir, 'original')
                os.makedirs(hires_dir, exist_ok=True)
            
            # Count existing images (both jpg and png)
            existing = len([f for f in os.listdir(emotion_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.captured_counts[emotion] = existing
    
    def _preprocess_face(self, frame, face_coords):
        """Extract and resize face region for YOLOv8-cls (224x224 RGB)."""
        x, y, w, h = face_coords
        
        # Add padding (20% on each side for context)
        padding = int(0.2 * w)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2*padding)
        h = min(frame.shape[0] - y, h + 2*padding)
        
        # Extract face (keep as RGB/BGR for YOLOv8)
        face = frame[y:y+h, x:x+w]
        
        # Resize to target size (224x224 for YOLOv8-cls)
        # Keep RGB format - DO NOT convert to grayscale!
        face_resized = cv2.resize(face, (self.target_size, self.target_size))
        
        return face, face_resized  # Return both high-res and target size
    
    def _save_image(self, image_target, emotion, image_hires=None):
        """Save image to appropriate emotion folder (224x224 RGB for YOLOv8)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"custom_{emotion}_{timestamp}_{self.captured_counts[emotion]:04d}.jpg"  # Use JPG for smaller size
        
        # Save target size version (224x224 RGB for YOLOv8-cls training)
        filepath_target = os.path.join(self.output_dir, emotion, filename)
        cv2.imwrite(filepath_target, image_target)
        
        # Save high-res original (for quality/backup)
        if self.save_hires and image_hires is not None:
            filepath_hires = os.path.join(self.output_dir, emotion, 'original', filename)
            cv2.imwrite(filepath_hires, image_hires)
        
        self.captured_counts[emotion] += 1
        
        return filename
    
    def _preprocess_for_yolo(self, frame, input_size=None):
        """Preprocess image for YOLOv8 face detection"""
        if input_size is None:
            input_size = self.face_input_size
        img = cv2.resize(frame, (input_size, input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def _nms(self, boxes, scores):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.face_confidence_threshold, self.iou_threshold
        )
        return indices.flatten() if len(indices) > 0 else []
    
    def _detect_faces_yolo(self, frame):
        """Detect faces using YOLOv8-face"""
        h, w = frame.shape[:2]
        
        # Preprocess
        input_blob = self._preprocess_for_yolo(frame, self.face_input_size)
        
        # Inference
        outputs = self.face_session.run(None, {self.face_input_name: input_blob})
        
        # Post-process
        predictions = outputs[0][0].T
        boxes_xywh = predictions[:, :4]
        scores = predictions[:, 4]
        
        # Filter by confidence
        mask = scores > self.face_confidence_threshold
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        
        if len(boxes_xywh) == 0:
            return []
        
        # Convert to corner format
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        
        # NMS
        keep_indices = self._nms(boxes_xyxy, scores)
        
        if len(keep_indices) == 0:
            return []
        
        # Scale to original size
        scale_x = w / self.face_input_size
        scale_y = h / self.face_input_size
        
        final_boxes = []
        for idx in keep_indices:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            x1 = int(max(0, min(x1 * scale_x, w)))
            y1 = int(max(0, min(y1 * scale_y, h)))
            x2 = int(max(0, min(x2 * scale_x, w)))
            y2 = int(max(0, min(y2 * scale_y, h)))
            
            # Convert to (x, y, w, h) format for compatibility
            final_boxes.append((x1, y1, x2-x1, y2-y1))
        
        return final_boxes
    
    def _draw_ui(self, frame):
        """Draw UI overlay on frame."""
        # Current emotion display
        cv2.putText(frame, f"Current: {self.current_emotion.upper()}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Captured counts
        y_offset = 70
        for emotion in self.emotions:
            count = self.captured_counts[emotion]
            color = (0, 255, 0) if emotion == self.current_emotion else (200, 200, 200)
            text = f"{emotion}: {count}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        # Instructions
        cv2.putText(frame, "SPACE: Capture | 1-5: Switch emotion | Q: Quit", 
                    (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def collect(self):
        """Main collection loop."""
        # Support both webcam ID (int) and IP camera URL (string)
        cap = cv2.VideoCapture(self.camera_source)
        
        # Set resolution (may not work for IP cameras)
        if isinstance(self.camera_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        auto_capture = False
        
        print("\nCamera opened. Start collecting!")
        print("  Tip: Make different expressions and press SPACE to capture")
        print(f"  Saving {self.target_size}x{self.target_size} RGB images for YOLOv8-cls")
        if self.save_hires:
            print("  Also saving high-res originals in 'original/' subfolder")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror for better UX
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Detect faces with YOLOv8
            faces = []
            if self.detect_face:
                faces = self._detect_faces_yolo(frame)
                
                # Draw face boxes
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw UI
            display_frame = self._draw_ui(display_frame)
            
            # Auto-capture mode
            if auto_capture:
                self.frame_counter += 1
                
                if self.frame_counter >= self.frame_skip:
                    self.frame_counter = 0
                    
                    # Capture
                    if self.detect_face and len(faces) > 0:
                        # Save first detected face (224x224 RGB for YOLO)
                        face_hires, face_target = self._preprocess_face(frame, faces[0])
                        filename = self._save_image(face_target, self.current_emotion, face_hires)
                        print(f"Auto-captured: {filename}")
                        
                        # Visual feedback
                        cv2.circle(display_frame, (30, 30), 15, (0, 255, 0), -1)
                    elif not self.detect_face:
                        # Save full frame (resized) - keep RGB for YOLO
                        resized = cv2.resize(frame, (self.target_size, self.target_size))
                        filename = self._save_image(resized, self.current_emotion, frame)
                        print(f"Auto-captured: {filename}")
                
                # Show auto-capture indicator
                cv2.putText(display_frame, "AUTO", (frame.shape[1] - 80, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display
            cv2.imshow('Emotion Data Collection', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            
            elif key == ord(' '):  # Space - manual capture
                if self.detect_face and len(faces) > 0:
                    face_hires, face_target = self._preprocess_face(frame, faces[0])
                    filename = self._save_image(face_target, self.current_emotion, face_hires)
                    print(f"Captured: {filename}")
                elif not self.detect_face:
                    # Keep RGB for YOLO
                    resized = cv2.resize(frame, (self.target_size, self.target_size))
                    filename = self._save_image(resized, self.current_emotion, frame)
                    print(f"Captured: {filename}")
                else:
                    print("No face detected!")
            
            elif key == ord('a'):  # Toggle auto-capture
                auto_capture = not auto_capture
                status = "ON" if auto_capture else "OFF"
                print(f"Auto-capture: {status}")
            
            elif key == ord('f'):  # Toggle face detection
                self.detect_face = not self.detect_face
                status = "ON" if self.detect_face else "OFF"
                print(f"Face detection: {status}")
            
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                idx = int(chr(key)) - 1
                self.current_emotion = self.emotions[idx]
                self.current_emotion_idx = idx
                print(f"Switched to: {self.current_emotion.upper()}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*60)
        print("COLLECTION SUMMARY")
        print("="*60)
        total = 0
        for emotion, count in self.captured_counts.items():
            print(f"  {emotion}: {count} images")
            total += count
        print(f"\nTotal: {total} images")
        print(f"Saved to: {self.output_dir}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Collect custom emotion dataset')
    parser.add_argument('-o', '--output', default='datasets/custom_emotions',
                       help='Output directory for collected images')
    parser.add_argument('--resolution', default='640x480',
                       help='Camera resolution (e.g., 640x480, 1280x720)')
    parser.add_argument('-c', '--camera', default='0',
                       help='Camera source: 0 (default webcam), 1 (second webcam), or IP camera URL (e.g., http://192.168.1.100:8080/video)')
    parser.add_argument('--no-hires', action='store_true',
                       help='Disable saving high-resolution originals')
    parser.add_argument('--size', type=int, default=224,
                       help='Target image size (224 for YOLOv8-cls, 48 for legacy FER2013)')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Parse camera source (int or string)
    try:
        camera_source = int(args.camera)
    except ValueError:
        camera_source = args.camera  # Treat as URL string
    
    collector = EmotionDataCollector(
        output_dir=args.output,
        resolution=(width, height),
        camera_source=camera_source,
        save_hires=not args.no_hires,
        target_size=args.size
    )
    
    collector.collect()


if __name__ == '__main__':
    main()
