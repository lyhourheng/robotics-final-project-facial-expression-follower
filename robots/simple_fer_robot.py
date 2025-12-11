#!/usr/bin/env python3
"""
Simple FER Robot Controller for Raspberry Pi 4
==============================================
Minimal, easy-to-use robot controller using facial expression recognition.

Run:
    python3 simple_fer_robot.py                    # Simulation mode
    python3 simple_fer_robot.py --hardware         # Real robot
    python3 simple_fer_robot.py --hardware --web   # With web UI

Emotion → Robot Action:
    Happy     → Forward
    Angry     → Backward
    Surprised → Turn Left
    Sad       → Turn Right
    Neutral   → Stop
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import threading
from collections import deque

# Import from local modules
from emotion_smoother import EmotionSmoother
from face_selector import FaceSelector
from robot_motor_control import EmotionRobotController

# ============================================================================
# CONFIGURATION - Edit these values as needed
# ============================================================================

# Model paths
FACE_MODEL = "../final-model/yolov8n-face.onnx"
FER_MODEL = "../final-model/fer_yolov8_cls.onnx"

# Detection settings
FACE_INPUT_SIZE = 320       # Face detector input (320-416)
FER_INPUT_SIZE = 128      # FER classifier input
FACE_CONF_THRESHOLD = 0.5   # Min face detection confidence
FER_CONF_THRESHOLD = 0.5    # Min emotion classification confidence

# Robot settings
ROBOT_PORT = "/dev/ttyUSB0"
ROBOT_BAUDRATE = 115200
BASE_SPEED = 25             # Motor speed (0-99)

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
ROTATE_180 = True           # Flip camera if mounted upside down

# Smoothing
EMA_ALPHA = 0.3             # Emotion smoothing (0-1, lower=smoother)

# Safety
MAX_NO_FACE_FRAMES = 30     # Stop after N frames without face
SCAN_WHEN_NO_FACE = True    # Rotate to find faces

# Timed Action Mode
ACTION_DURATION = 5.0       # How long to execute action (seconds)

# Classes (must match model training order)
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# Emotion to action mapping
EMOTION_ACTIONS = {
    'happy': 'forward',
    'angry': 'backward',
    'surprised': 'left',
    'sad': 'right',
    'neutral': 'stop'
}


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def preprocess_face(frame, input_size):
    """Preprocess frame for face detection."""
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)


def preprocess_fer(face_crop, input_size):
    """Preprocess face crop for emotion classification."""
    img = cv2.resize(face_crop, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)


def detect_faces(session, input_name, frame, input_size, conf_threshold):
    """Detect faces in frame."""
    h, w = frame.shape[:2]
    
    # Run inference
    blob = preprocess_face(frame, input_size)
    outputs = session.run(None, {input_name: blob})
    preds = outputs[0][0].T
    
    # Parse outputs
    boxes_xywh = preds[:, :4]
    scores = preds[:, 4]
    
    # Filter by confidence
    mask = scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    
    if len(boxes_xywh) == 0:
        return [], []
    
    # Convert to xyxy
    boxes = np.zeros_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.45)
    indices = indices.flatten() if len(indices) > 0 else []
    
    # Scale to original size
    scale_x = w / input_size
    scale_y = h / input_size
    
    final_boxes = []
    final_scores = []
    
    for i in indices:
        x1, y1, x2, y2 = boxes[i]
        x1 = int(max(0, min(x1 * scale_x, w)))
        y1 = int(max(0, min(y1 * scale_y, h)))
        x2 = int(max(0, min(x2 * scale_x, w)))
        y2 = int(max(0, min(y2 * scale_y, h)))
        final_boxes.append([x1, y1, x2, y2])
        final_scores.append(float(scores[i]))
    
    return final_boxes, final_scores


def classify_emotion(session, input_name, face_crop, input_size):
    """Classify emotion from face crop."""
    blob = preprocess_fer(face_crop, input_size)
    outputs = session.run(None, {input_name: blob})
    probs = outputs[0][0]
    
    # Apply softmax if needed
    if probs.min() < 0 or probs.sum() > 1.5:
        probs = np.exp(probs) / np.sum(np.exp(probs))
    
    return probs


# ============================================================================
# VISUALIZATION
# ============================================================================

COLORS = {
    'angry': (0, 0, 255),
    'happy': (0, 255, 0),
    'neutral': (128, 128, 128),
    'sad': (255, 0, 0),
    'surprised': (0, 255, 255)
}


def draw_frame(frame, boxes, target_idx, emotion, confidence, fps):
    """Draw detection results on frame."""
    annotated = frame.copy()
    
    # Draw all faces
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if i == target_idx:
            color = COLORS.get(emotion, (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{emotion.upper()}: {confidence:.0%}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 128, 128), 2)
    
    # Draw info
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated


# ============================================================================
# WEB UI TEMPLATE (similar to fer_robot_controller.py)
# ============================================================================

WEB_UI_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Simple FER Robot Controller</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            background: #0b0f12; 
            color: #eee; 
            font-family: 'Segoe UI', Arial, sans-serif; 
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { 
            color: #7fffd4; 
            text-align: center; 
            margin-bottom: 25px;
            font-size: 2em;
        }
        .video-box { 
            background: #0f1720; 
            padding: 15px; 
            border-radius: 12px; 
            margin-bottom: 25px; 
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .video-box img { 
            border-radius: 8px; 
            max-width: 100%; 
            height: auto;
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); 
            gap: 15px; 
            margin-bottom: 25px;
        }
        .stat-card { 
            background: #0f1720; 
            padding: 18px 15px; 
            border-radius: 10px; 
            border-left: 4px solid #7fffd4;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(127, 255, 212, 0.15);
        }
        .stat-label { 
            color: #9aa7b2; 
            font-size: 0.85em; 
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .stat-value { 
            color: #fff; 
            font-size: 1.4em; 
            font-weight: bold; 
        }
        .emotion-happy { color: #00ff00 !important; }
        .emotion-angry { color: #ff4444 !important; }
        .emotion-sad { color: #4488ff !important; }
        .emotion-surprised { color: #ffff00 !important; }
        .emotion-neutral { color: #888888 !important; }
        
        .action-forward { border-left-color: #00ff00; }
        .action-backward { border-left-color: #ff4444; }
        .action-left { border-left-color: #ffff00; }
        .action-right { border-left-color: #4488ff; }
        .action-stop { border-left-color: #888888; }
        
        .emotion-legend {
            background: #0f1720;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .emotion-legend h3 {
            color: #7fffd4;
            margin-bottom: 10px;
            font-size: 1em;
        }
        .legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .footer {
            text-align: center;
            color: #666;
            font-size: 0.85em;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Simple FER Robot Controller</h1>
        
        <div class="video-box">
            <img src="/video" alt="Camera Feed">
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">FPS</div>
                <div class="stat-value" id="fps">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Faces</div>
                <div class="stat-value" id="faces">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Emotion</div>
                <div class="stat-value" id="emotion">---</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Confidence</div>
                <div class="stat-value" id="confidence">0%</div>
            </div>
            <div class="stat-card" id="action-card">
                <div class="stat-label">Action</div>
                <div class="stat-value" id="action">STOP</div>
            </div>
            <div class="stat-card" id="timer-card">
                <div class="stat-label">Timer</div>
                <div class="stat-value" id="timer">---</div>
            </div>
            <div class="stat-card" style="grid-column: span 2;">
                <div class="stat-label">State</div>
                <div class="stat-value" id="state">---</div>
            </div>
        </div>
        
        <div class="emotion-legend">
            <h3>Emotion → Action Mapping</h3>
            <div class="legend-grid">
                <div class="legend-item">
                    <div class="legend-dot" style="background:#00ff00"></div>
                    <span>Happy → Forward</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background:#ff4444"></div>
                    <span>Angry → Backward</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background:#ffff00"></div>
                    <span>Surprised → Left</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background:#4488ff"></div>
                    <span>Sad → Right</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background:#888888"></div>
                    <span>Neutral → Stop</span>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Press 'q' in terminal to quit | Simple FER Robot Controller
        </div>
    </div>
    
    <script>
        function update() {
            fetch('/status').then(r => r.json()).then(d => {
                document.getElementById('fps').textContent = d.fps.toFixed(1);
                document.getElementById('faces').textContent = d.faces;
                
                const emotionEl = document.getElementById('emotion');
                emotionEl.textContent = d.emotion.toUpperCase();
                emotionEl.className = 'stat-value emotion-' + d.emotion;
                
                document.getElementById('confidence').textContent = 
                    (d.confidence * 100).toFixed(0) + '%';
                document.getElementById('action').textContent = 
                    d.action.toUpperCase().replace('_', ' ');
                document.getElementById('state').textContent = d.state;
                
                // Update timer
                const timerEl = document.getElementById('timer');
                const timerCard = document.getElementById('timer-card');
                if (d.time_remaining > 0) {
                    timerEl.textContent = d.time_remaining.toFixed(1) + 's';
                    timerEl.style.color = '#00ff00';
                    timerCard.style.borderLeftColor = '#00ff00';
                } else {
                    timerEl.textContent = '---';
                    timerEl.style.color = '#888';
                    timerCard.style.borderLeftColor = '#7fffd4';
                }
                
                // Update action card color
                const actionCard = document.getElementById('action-card');
                actionCard.className = 'stat-card action-' + d.action.split('_')[0];
            }).catch(e => console.log('Update error:', e));
        }
        setInterval(update, 300);
        update();
    </script>
</body>
</html>
"""


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple FER Robot Controller')
    parser.add_argument('--hardware', action='store_true', help='Run on real robot hardware')
    parser.add_argument('--web', action='store_true', help='Enable web UI')
    parser.add_argument('--display', action='store_true', help='Show OpenCV window')
    parser.add_argument('--speed', type=int, default=BASE_SPEED, help='Motor speed (0-99)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMPLE FER ROBOT CONTROLLER")
    print("=" * 60)
    print(f"Mode: {'HARDWARE' if args.hardware else 'SIMULATION'}")
    print(f"Speed: {args.speed}")
    print("=" * 60)
    
    # Check models exist
    if not os.path.exists(FACE_MODEL):
        print(f"❌ Face model not found: {FACE_MODEL}")
        sys.exit(1)
    if not os.path.exists(FER_MODEL):
        print(f"❌ FER model not found: {FER_MODEL}")
        sys.exit(1)
    
    # Load models
    print("\nLoading models...")
    face_session = ort.InferenceSession(FACE_MODEL, providers=['CPUExecutionProvider'])
    face_input = face_session.get_inputs()[0].name
    print(f"✓ Face detector: {FACE_MODEL}")
    
    fer_session = ort.InferenceSession(FER_MODEL, providers=['CPUExecutionProvider'])
    fer_input = fer_session.get_inputs()[0].name
    print(f"✓ FER classifier: {FER_MODEL}")
    
    # Initialize components using imported modules
    # Use 'reactive' mode since we handle timing ourselves
    motor = EmotionRobotController(
        simulation=not args.hardware,
        port=ROBOT_PORT,
        baudrate=ROBOT_BAUDRATE,
        base_speed=args.speed,
        control_mode='reactive'  # We handle timing in this script
    )
    smoother = EmotionSmoother(
        method='ema',
        ema_alpha=EMA_ALPHA,
        class_names=CLASS_NAMES
    )
    face_selector = FaceSelector(
        mode='center',
        frame_width=CAMERA_WIDTH,
        frame_height=CAMERA_HEIGHT
    )
    
    # Open camera
    print(f"\nOpening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    print("✓ Camera ready")
    
    # Web UI state
    web_frame = None
    frame_lock = threading.Lock()
    status = {
        "emotion": "none", 
        "confidence": 0, 
        "action": "stop", 
        "fps": 0,
        "faces": 0,
        "state": "Initializing"
    }
    
    if args.web:
        from flask import Flask, Response, jsonify, render_template_string
        
        app = Flask(__name__)
        
        @app.route("/")
        def index():
            return render_template_string(WEB_UI_TEMPLATE)
        
        @app.route("/video")
        def video():
            def gen():
                while True:
                    with frame_lock:
                        if web_frame is not None:
                            frame_to_send = web_frame.copy()
                        else:
                            frame_to_send = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
                    
                    ret, buf = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
                    time.sleep(0.033)
            return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
        @app.route("/status")
        def get_status():
            return jsonify(status)
        
        # Get IP address
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except:
            ip = "localhost"
        
        threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True), 
            daemon=True
        ).start()
        print(f"\n✓ Web UI: http://{ip}:5000")
    
    print("\n" + "=" * 60)
    print("RUNNING - Press 'q' to quit")
    print("=" * 60 + "\n")
    
    # State variables
    no_face_frames = 0
    scan_dir = 1
    fps_history = deque(maxlen=30)
    
    # Timed action state
    is_executing_action = False
    action_start_time = 0
    current_timed_action = 'stop'
    current_timed_emotion = 'neutral'
    time_remaining = 0
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            if ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            h, w = frame.shape[:2]
            
            emotion = 'neutral'
            confidence = 0.0
            action = 'stop'
            target_idx = None
            state = "Running"
            
            # Check if currently executing a timed action
            if is_executing_action:
                elapsed = time.time() - action_start_time
                time_remaining = max(0, ACTION_DURATION - elapsed)
                
                if elapsed >= ACTION_DURATION:
                    # Action complete - stop and go back to finding faces
                    motor.stop()
                    is_executing_action = False
                    current_timed_action = 'stop'
                    smoother.reset()
                    state = "Action Complete - Finding Face"
                    print(f"\n✓ Action complete! Looking for face again...")
                else:
                    # Still executing - keep the action going
                    state = f"Executing: {current_timed_emotion.upper()} ({time_remaining:.1f}s)"
                    action = current_timed_action
                    emotion = current_timed_emotion
                    confidence = 1.0  # Keep confidence high during execution
                    
                    # Still draw frame but skip face detection during execution
                    boxes, scores = detect_faces(face_session, face_input, frame, 
                                                FACE_INPUT_SIZE, FACE_CONF_THRESHOLD)
                    target_result = face_selector.select_target(boxes, scores)
                    if target_result:
                        target_idx = target_result[2]
            else:
                # Not executing - detect faces and emotions
                boxes, scores = detect_faces(face_session, face_input, frame, 
                                            FACE_INPUT_SIZE, FACE_CONF_THRESHOLD)
                
                # Select target face using FaceSelector
                target_result = face_selector.select_target(boxes, scores)
                
                if target_result is None:
                    no_face_frames += 1
                    
                    if no_face_frames > MAX_NO_FACE_FRAMES and SCAN_WHEN_NO_FACE:
                        # Scan for faces
                        state = "Scanning"
                        action = 'scan_right' if scan_dir > 0 else 'scan_left'
                        motor.stop()
                        
                        # Switch direction every 2 seconds
                        if no_face_frames % 60 == 0:
                            scan_dir *= -1
                    else:
                        motor.stop()
                        state = "No Face"
                    
                    smoother.reset()
                else:
                    no_face_frames = 0
                    target_box, target_score, target_idx = target_result
                    
                    x1, y1, x2, y2 = target_box
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        # Classify emotion
                        probs = classify_emotion(fer_session, fer_input, face_crop, FER_INPUT_SIZE)
                        
                        # Smooth emotion using imported EmotionSmoother
                        smoothed_emotion = smoother.update(emotion=CLASS_NAMES[np.argmax(probs)], 
                                                           probabilities=probs)
                        emotion = smoothed_emotion
                        confidence = smoother.get_confidence() or float(np.max(probs))
                        
                        # Debug: Print detection status every 30 frames
                        if int(time.time() * 2) % 2 == 0:
                            print(f"\r👤 Face detected | Emotion: {emotion.upper():10s} | Conf: {confidence:.0%} | State: {state}", end="", flush=True)
                        
                        # Execute action if confident enough (TIMED MODE)
                        if confidence >= FER_CONF_THRESHOLD and emotion != 'neutral':
                            action = EMOTION_ACTIONS.get(emotion, 'stop')
                            
                            # Start timed action
                            is_executing_action = True
                            action_start_time = time.time()
                            current_timed_action = action
                            current_timed_emotion = emotion
                            time_remaining = ACTION_DURATION
                            
                            # Execute the motor action
                            motor.execute_emotion(emotion)
                            
                            state = f"Starting: {emotion.upper()} for {ACTION_DURATION}s"
                            print(f"\n🎯 Detected {emotion.upper()} ({confidence:.0%}) → {action.upper()} for {ACTION_DURATION}s")
                        else:
                            # Low confidence or neutral - just wait
                            state = "Detecting..."
                            action = 'stop'
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time + 0.001)
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            
            # Draw frame (get boxes if not already fetched)
            if 'boxes' not in dir() or (not is_executing_action and target_result is None):
                boxes = []
            annotated = draw_frame(frame, boxes, target_idx, emotion, confidence, avg_fps)
            
            # Draw timer on frame if executing
            if is_executing_action:
                timer_text = f"Action: {current_timed_action.upper()} - {time_remaining:.1f}s"
                cv2.putText(annotated, timer_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Update web UI (always update frame and status)
            with frame_lock:
                web_frame = annotated.copy()
            status.update({
                "emotion": emotion, 
                "confidence": confidence, 
                "action": action, 
                "fps": avg_fps,
                "faces": len(boxes),
                "state": state,
                "time_remaining": round(time_remaining, 1) if is_executing_action else 0
            })
            
            # Display
            if args.display:
                cv2.imshow('FER Robot', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        motor.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Done!")


if __name__ == '__main__':
    main()
