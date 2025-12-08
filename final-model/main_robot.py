"""
Main Robot Pipeline: Facial Expression Follower
Integrates: YOLOv8-face detection + FER classification + Target selection + 
            Emotion smoothing + Motor control
Web UI: Real-time camera feed and status display
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import sys

# Import our modules
from face_selector import FaceSelector
from emotion_smoother import EmotionSmoother
from robot_motor_control import EmotionRobotController

print("=" * 60)
print("FACIAL EXPRESSION FOLLOWER - MAIN PIPELINE")
print("=" * 60)
print("=" * 60)

# ============== CONFIGURATION ==============
FACE_MODEL_PATH = "../yolov8n-face.onnx"
FER_MODEL_PATH = "fer_yolov8_cls.onnx"
FACE_INPUT_SIZE = 320
FER_INPUT_SIZE = 128
FACE_CONFIDENCE_THRESHOLD = 0.4
FER_CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Class names (FER model output order)
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# Emotion colors for visualization
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Red
    'happy': (0, 255, 0),      # Green
    'neutral': (128, 128, 128), # Gray
    'sad': (255, 0, 0),        # Blue
    'surprised': (0, 255, 255) # Yellow
}

# Robot configuration
ROBOT_SIMULATION = False  # Set to False for real hardware
ROBOT_PORT = "/dev/ttyUSB0"
ROBOT_BAUDRATE = 115200
ROBOT_SPEED = 25
ROBOT_CONTROL_MODE = 'timed'  # 'reactive' or 'timed'
ROBOT_ACTION_DURATION = 6.0  # Duration for each action in timed mode (seconds)

# Camera configuration (matching run.py setup)
CAM_INDEX = 0  # Robot camera
CAM_WIDTH = 640
CAM_HEIGHT = 480
ROTATE_180 = True  # Rotate camera feed 180 degrees (same as run.py)

# Face selection mode: 'center', 'largest', or 'locked'
FACE_SELECTION_MODE = 'center'

# Emotion smoothing: 'majority' or 'ema'
SMOOTHING_METHOD = 'ema'
EMA_ALPHA = 0.3
MAJORITY_WINDOW = 5

# Safety settings
MAX_NO_FACE_FRAMES = 30  # Stop robot after N frames without face
SCAN_MODE_ENABLED = True  # Slow rotate when no face detected

# Web UI settings
ENABLE_WEB_UI = True  # Enable Flask web interface
WEB_PORT = 5000  # Web server port

print(f"\nConfiguration:")
print(f"  Face Model: {FACE_MODEL_PATH}")
print(f"  FER Model: {FER_MODEL_PATH}")
print(f"  Face Selection: {FACE_SELECTION_MODE}")
print(f"  Smoothing: {SMOOTHING_METHOD}")
print(f"  Robot Mode: {'SIMULATION' if ROBOT_SIMULATION else 'HARDWARE'}")
print(f"  Web UI: {'ENABLED' if ENABLE_WEB_UI else 'DISABLED'} on port {WEB_PORT}")
print(f"  Control Mode: {ROBOT_CONTROL_MODE}")
if ROBOT_CONTROL_MODE == 'timed':
    print(f"  Action Duration: {ROBOT_ACTION_DURATION}s")
print("=" * 60)

# ============== CHECK MODELS ==============
if not os.path.exists(FACE_MODEL_PATH):
    print(f"❌ Face model not found: {FACE_MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(FER_MODEL_PATH):
    print(f"❌ FER model not found: {FER_MODEL_PATH}")
    sys.exit(1)

# ============== LOAD MODELS ==============
print("\nLoading models...")
face_session = ort.InferenceSession(FACE_MODEL_PATH, providers=['CPUExecutionProvider'])
fer_session = ort.InferenceSession(FER_MODEL_PATH, providers=['CPUExecutionProvider'])
face_input_name = face_session.get_inputs()[0].name
fer_input_name = fer_session.get_inputs()[0].name
print("✓ Face detector loaded")
print("✓ FER classifier loaded")

# ============== INITIALIZE MODULES ==============
print("\nInitializing modules...")

# Face selector
face_selector = FaceSelector(
    mode=FACE_SELECTION_MODE,
    frame_width=640,
    frame_height=480
)
print(f"✓ Face selector initialized ({FACE_SELECTION_MODE} mode)")

# Emotion smoother
if SMOOTHING_METHOD == 'ema':
    emotion_smoother = EmotionSmoother(
        method='ema',
        ema_alpha=EMA_ALPHA,
        class_names=CLASS_NAMES
    )
else:
    emotion_smoother = EmotionSmoother(
        method='majority',
        window_size=MAJORITY_WINDOW,
        class_names=CLASS_NAMES
    )
print(f"✓ Emotion smoother initialized ({SMOOTHING_METHOD})")

# Robot controller
robot_controller = EmotionRobotController(
    simulation=ROBOT_SIMULATION,
    port=ROBOT_PORT,
    baudrate=ROBOT_BAUDRATE,
    base_speed=ROBOT_SPEED,
    control_mode=ROBOT_CONTROL_MODE,
    action_duration=ROBOT_ACTION_DURATION
)
print("✓ Robot controller initialized")

# ============== WEB UI SETUP ==============
# Global variables for web streaming
latest_frame = None
frame_lock = threading.Lock()
status_info = {
    "fps": 0,
    "faces": 0,
    "target_emotion": "---",
    "smoothed_emotion": "---",
    "confidence": 0.0,
    "robot_action": "IDLE",
    "mode": FACE_SELECTION_MODE,
    "state": "INITIALIZING"
}

# Flask app
app = Flask(__name__)

def get_ip_address():
    """Get local IP address."""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

# HTML template for web UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Facial Expression Follower - Robot Control</title>
    <style>
        body {
            background: #0b0f12;
            color: #eee;
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #7fffd4;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #9aa7b2;
            margin-bottom: 30px;
        }
        .video-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        .video-box {
            background: #0f1720;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .video-box h3 {
            margin: 0 0 10px 0;
            color: #7fffd4;
            font-size: 1.1em;
        }
        img {
            border-radius: 5px;
            display: block;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #0f1720;
            padding: 15px 20px;
            border-radius: 8px;
            border-left: 4px solid #7fffd4;
        }
        .stat-label {
            color: #9aa7b2;
            font-size: 0.85em;
            margin-bottom: 5px;
        }
        .stat-value {
            color: #fff;
            font-size: 1.5em;
            font-weight: bold;
        }
        .emotion-happy { color: #00ff00 !important; }
        .emotion-angry { color: #ff0000 !important; }
        .emotion-sad { color: #0080ff !important; }
        .emotion-surprised { color: #ffff00 !important; }
        .emotion-neutral { color: #888888 !important; }
        .action-forward { color: #00ff00 !important; }
        .action-backward { color: #ff0000 !important; }
        .action-turn_left { color: #ffff00 !important; }
        .action-turn_right { color: #ff8800 !important; }
        .action-stop { color: #888888 !important; }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Facial Expression Follower</h1>
        <p class="subtitle">Real-time Face Detection + Emotion Recognition + Robot Control</p>
        
        <div class="video-container">
            <div class="video-box">
                <h3>📷 Live Camera Feed</h3>
                <img src="/video_feed" width="640" height="480" alt="Camera Feed">
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">FPS</div>
                <div class="stat-value" id="fps">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Faces Detected</div>
                <div class="stat-value" id="faces">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Target Emotion</div>
                <div class="stat-value" id="emotion">---</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Smoothed Emotion</div>
                <div class="stat-value" id="smoothed">---</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Confidence</div>
                <div class="stat-value" id="confidence">0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Robot Action</div>
                <div class="stat-value" id="action">IDLE</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Selection Mode</div>
                <div class="stat-value" id="mode">---</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">System State</div>
                <div class="stat-value" id="state">---</div>
            </div>
        </div>
        
        <p class="footer">
            Auto-updates every 300ms | Press 'q' in terminal to quit | Press 's' to switch selection mode
        </p>
    </div>
    
    <script>
        function updateStats() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps;
                    document.getElementById('faces').textContent = data.faces;
                    
                    const emotionElem = document.getElementById('emotion');
                    emotionElem.textContent = data.target_emotion.toUpperCase();
                    emotionElem.className = 'stat-value emotion-' + data.target_emotion;
                    
                    const smoothedElem = document.getElementById('smoothed');
                    smoothedElem.textContent = data.smoothed_emotion.toUpperCase();
                    smoothedElem.className = 'stat-value emotion-' + data.smoothed_emotion;
                    
                    document.getElementById('confidence').textContent = 
                        (data.confidence * 100).toFixed(1) + '%';
                    
                    const actionElem = document.getElementById('action');
                    actionElem.textContent = data.robot_action.toUpperCase();
                    actionElem.className = 'stat-value action-' + data.robot_action.toLowerCase();
                    
                    document.getElementById('mode').textContent = data.mode.toUpperCase();
                    document.getElementById('state').textContent = data.state;
                })
                .catch(err => console.error('Error fetching status:', err));
        }
        
        // Update every 300ms
        setInterval(updateStats, 300);
        updateStats();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    """Main web page."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()
                else:
                    frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def status():
    """Status JSON endpoint."""
    return jsonify(status_info)

def start_web_server():
    """Start Flask web server in background thread."""
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False, threaded=True)

if ENABLE_WEB_UI:
    print("\nStarting web server...")
    threading.Thread(target=start_web_server, daemon=True).start()
    time.sleep(1.0)
    print(f"🌐 Web UI: http://{get_ip_address()}:{WEB_PORT}")
    print(f"   Local: http://localhost:{WEB_PORT}")
print("=" * 60)

# Initialize robot controller
robot_controller = EmotionRobotController(
    control_mode=ROBOT_CONTROL_MODE,
    action_duration=ROBOT_ACTION_DURATION
)
print("✓ Robot controller initialized")

# ============== WEB UI SETUP ==============
if ENABLE_WEB_UI:
    import threading
    from flask import Flask, Response, jsonify, render_template_string
    
    app = Flask(__name__)
    
    # Global state for web UI
    latest_frame = None
    frame_lock = threading.Lock()
    status_info = {
        "fps": 0,
        "face_count": 0,
        "selected_face": None,
        "detected_emotion": "None",
        "smoothed_emotion": "None",
        "robot_action": "Stopped",
        "confidence": 0.0,
        "timestamp": ""
    }
    
    HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Emotion Robot Control</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #00bcd4;
        }
        .grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .video-panel {
            background: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
        }
        .stats-panel {
            background: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
        }
        .video-feed {
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 2px solid #00bcd4;
        }
        .stat-item {
            margin: 15px 0;
            padding: 15px;
            background: #333;
            border-radius: 8px;
            border-left: 4px solid #00bcd4;
        }
        .stat-label {
            font-size: 12px;
            color: #aaa;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00bcd4;
            margin-top: 5px;
        }
        .emotion-happy { color: #4caf50 !important; }
        .emotion-sad { color: #2196f3 !important; }
        .emotion-angry { color: #f44336 !important; }
        .emotion-surprised { color: #ff9800 !important; }
        .emotion-neutral { color: #9e9e9e !important; }
        .action {
            font-size: 18px;
            padding: 10px;
            margin-top: 5px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        .action-forward { background: #4caf50; }
        .action-backward { background: #f44336; }
        .action-left { background: #ff9800; }
        .action-right { background: #2196f3; }
        .action-stop { background: #9e9e9e; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Emotion-Based Robot Control System</h1>
        <div class="grid">
            <div class="video-panel">
                <h2>Live Camera Feed</h2>
                <img src="/video_feed" class="video-feed" alt="Camera Feed">
            </div>
            <div class="stats-panel">
                <h2>System Status</h2>
                <div class="stat-item">
                    <div class="stat-label">FPS</div>
                    <div class="stat-value" id="fps">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Faces Detected</div>
                    <div class="stat-value" id="face-count">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Detected Emotion</div>
                    <div class="stat-value" id="detected-emotion">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Smoothed Emotion</div>
                    <div class="stat-value" id="smoothed-emotion">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Confidence</div>
                    <div class="stat-value" id="confidence">--</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Robot Action</div>
                    <div class="action" id="robot-action">Stopped</div>
                </div>
            </div>
        </div>
    </div>
    <script>
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('face-count').textContent = data.face_count;
                    
                    // Update detected emotion with color
                    const detectedElem = document.getElementById('detected-emotion');
                    detectedElem.textContent = data.detected_emotion;
                    detectedElem.className = 'stat-value emotion-' + data.detected_emotion.toLowerCase();
                    
                    // Update smoothed emotion with color
                    const smoothedElem = document.getElementById('smoothed-emotion');
                    smoothedElem.textContent = data.smoothed_emotion;
                    smoothedElem.className = 'stat-value emotion-' + data.smoothed_emotion.toLowerCase();
                    
                    document.getElementById('confidence').textContent = 
                        (data.confidence * 100).toFixed(1) + '%';
                    
                    // Update robot action with style
                    const actionElem = document.getElementById('robot-action');
                    actionElem.textContent = data.robot_action;
                    actionElem.className = 'action action-' + data.robot_action.toLowerCase();
                })
                .catch(error => console.error('Error fetching status:', error));
        }
        
        // Update every 100ms for smooth UI
        setInterval(updateStatus, 100);
        updateStatus();
    </script>
</body>
</html>
    """
    
    @app.route('/')
    def index():
        return HTML_TEMPLATE
    
    @app.route('/video_feed')
    def video_feed():
        def generate():
            while True:
                with frame_lock:
                    if latest_frame is None:
                        continue
                    frame = latest_frame.copy()
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/status')
    def status():
        with frame_lock:
            return jsonify(status_info)
    
    def start_web_server():
        app.run(host='0.0.0.0', port=WEB_PORT, debug=False, use_reloader=False, threaded=True)
    
    # Start Flask in background thread
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    print(f"✓ Web UI started at http://localhost:{WEB_PORT}")

# ============== HELPER FUNCTIONS ==============

def preprocess_face_detection(frame, input_size=FACE_INPUT_SIZE):
    """Preprocess image for YOLOv8 face detection."""
    img = cv2.resize(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_fer(face_img, img_size=FER_INPUT_SIZE):
    """Preprocess face crop for FER classifier."""
    face_img = cv2.resize(face_img, (img_size, img_size))
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_rgb = face_rgb.astype(np.float32) / 255.0
    face_rgb = np.transpose(face_rgb, (2, 0, 1))
    face_rgb = np.expand_dims(face_rgb, axis=0)
    return face_rgb

def nms(boxes, scores, iou_threshold=IOU_THRESHOLD):
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(),
        FACE_CONFIDENCE_THRESHOLD, iou_threshold
    )
    return indices.flatten() if len(indices) > 0 else []

def detect_faces(frame):
    """Detect faces using YOLOv8-face."""
    h, w = frame.shape[:2]
    input_blob = preprocess_face_detection(frame, FACE_INPUT_SIZE)
    outputs = face_session.run(None, {face_input_name: input_blob})
    predictions = outputs[0][0].T
    boxes_xywh = predictions[:, :4]
    scores = predictions[:, 4]
    
    mask = scores > FACE_CONFIDENCE_THRESHOLD
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    
    if len(boxes_xywh) == 0:
        return [], []
    
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    
    keep_indices = nms(boxes_xyxy, scores)
    
    if len(keep_indices) == 0:
        return [], []
    
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
    """Classify emotion from face crop."""
    face_input = preprocess_fer(face_crop, FER_INPUT_SIZE)
    outputs = fer_session.run(None, {fer_input_name: face_input})
    probs = outputs[0][0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    emotion = CLASS_NAMES[pred_idx]
    return emotion, confidence, probs

def detect_camera():
    """Open robot camera (same as run.py)."""
    print(f"\nOpening robot camera (index {CAM_INDEX})...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        ret, frame = cap.read()
        if ret:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"✓ Robot camera opened: {int(width)}x{int(height)}")
            cap.release()
            return CAM_INDEX
    cap.release()
    print("❌ Robot camera not available")
    return None

# ============== MAIN LOOP ==============

def main():
    """Main robot control loop."""
    
    # Open camera
    camera_index = detect_camera()
    if camera_index is None:
        print("❌ No cameras detected")
        return
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    print("\n✓ Camera opened")
    print("\nPress 'q' to quit, 's' to toggle selection mode")
    print("=" * 60)
    print("\nStarting robot control loop...\n")
    
    # Statistics
    fps_list = []
    frame_count = 0
    no_face_frames = 0
    emotion_counts = {emotion: 0 for emotion in CLASS_NAMES}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Rotate 180 degrees (same as run.py)
            if ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # Ensure frame is correct size
            frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
            
            frame_count += 1
            start_time = time.time()
            
            # Stage 1: Detect all faces
            face_boxes, face_scores = detect_faces(frame)
            
            # Stage 2: Select target face
            target_result = face_selector.select_target(face_boxes, face_scores)
            
            if target_result is None:
                # No face detected
                no_face_frames += 1
                
                if no_face_frames > MAX_NO_FACE_FRAMES:
                    # Stop robot after too many frames without face
                    robot_controller.stop()
                    emotion_smoother.reset()
                    
                    # Draw "NO FACE" message
                    cv2.putText(frame, "NO FACE - ROBOT STOPPED", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Face detected - reset counter
                no_face_frames = 0
                
                target_box, target_score, target_idx = target_result
                x1, y1, x2, y2 = target_box
                
                # Stage 3: Classify emotion
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    raw_emotion, emotion_conf, probs = classify_emotion(face_crop)
                    
                    # Stage 4: Smooth emotion
                    smoothed_emotion = emotion_smoother.update(raw_emotion, probabilities=probs)
                    smoothed_conf = emotion_smoother.get_confidence()
                    
                    # Update statistics
                    if smoothed_conf and smoothed_conf >= FER_CONFIDENCE_THRESHOLD:
                        emotion_counts[smoothed_emotion] += 1
                    
                    # Stage 5: Execute robot action
                    robot_controller.execute_emotion(smoothed_emotion)
                    
                    # Visualization
                    color = EMOTION_COLORS[smoothed_emotion]
                    
                    # Draw target box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw labels
                    label1 = f"Face: {target_score:.2f}"
                    label2 = f"{smoothed_emotion.upper()}: {smoothed_conf:.2f}" if smoothed_conf else f"{smoothed_emotion.upper()}"
                    
                    (w1, h1), _ = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - h1 - 30), (x1 + w1, y1 - 20), (0, 255, 0), -1)
                    cv2.putText(frame, label1, (x1, y1 - 23),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    (w2, h2), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w2, y1), color, -1)
                    cv2.putText(frame, label2, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw other detected faces (non-target) in gray
                    for i, (box, score) in enumerate(zip(face_boxes, face_scores)):
                        if i != target_idx:
                            bx1, by1, bx2, by2 = box
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (128, 128, 128), 2)
            
            # Calculate FPS
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time
            fps_list.append(fps)
            
            # Draw info overlay
            avg_fps = np.mean(fps_list[-30:]) if fps_list else 0
            info_text = f"FPS: {avg_fps:.1f} | Faces: {len(face_boxes)} | Mode: {face_selector.mode}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Facial Expression Follower', frame)
            
            # Update web UI
            if ENABLE_WEB_UI:
                with frame_lock:
                    latest_frame = frame.copy()
                    status_info.update({
                        "fps": avg_fps,
                        "face_count": len(face_boxes),
                        "selected_face": target_idx if target_result else None,
                        "detected_emotion": raw_emotion if target_result and face_crop.size > 0 else "None",
                        "smoothed_emotion": smoothed_emotion if target_result and face_crop.size > 0 else "None",
                        "robot_action": robot_controller.get_current_action(),
                        "confidence": smoothed_conf if target_result and face_crop.size > 0 and smoothed_conf else 0.0,
                        "timestamp": time.strftime("%H:%M:%S")
                    })
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {avg_fps:.1f} FPS | {len(face_boxes)} face(s)")
            
            # Handle keypresses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Toggle selection mode
                modes = ['center', 'largest', 'locked']
                current_idx = modes.index(face_selector.mode)
                new_mode = modes[(current_idx + 1) % len(modes)]
                face_selector.set_mode(new_mode)
                print(f"\n→ Selection mode changed to: {new_mode}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        robot_controller.stop()
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        if fps_list:
            print("\n" + "=" * 60)
            print("FINAL STATISTICS")
            print("=" * 60)
            print(f"Total frames: {frame_count}")
            print(f"Average FPS: {np.mean(fps_list):.2f}")
            
            print("\nEmotion Distribution:")
            for emotion in CLASS_NAMES:
                count = emotion_counts[emotion]
                percentage = (count / frame_count * 100) if frame_count > 0 else 0
                print(f"  {emotion.capitalize():10s}: {count:4d} ({percentage:.1f}%)")
            
            print("\n✓ Session complete!")
            print("=" * 60)

if __name__ == '__main__':
    main()
