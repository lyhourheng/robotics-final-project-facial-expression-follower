# YOLOv8-face ONNX Model Verification

This folder contains scripts to verify that your YOLOv8-face ONNX model works correctly.

## Files

- **test_yolov8_face_single_image.py** - Quick test without camera (verifies model loads and runs)
- **test_yolov8_face_onnx.py** - Full webcam test with face detection visualization

## Step-by-Step Testing

### Step 1: Quick Model Verification (No Camera)

```powershell
python test_yolov8_face_single_image.py
```

**Expected Output:**
```
✓ Model loaded
✓ Model is working
MODEL VERIFICATION COMPLETE
```

This verifies:
- Model file exists and loads correctly
- ONNX Runtime works
- Model can run inference
- Input/output shapes are correct

### Step 2: Real-time Face Detection Test

```powershell
python test_yolov8_face_onnx.py
```

**Expected Behavior:**
- Detects available cameras (including iVCam)
- Opens camera feed
- Draws green boxes around detected faces
- Shows FPS and face count
- Press 'q' to quit

**Expected Output:**
```
✓ iVCam detected at index 1 (1920x1080)
✓ Camera opened
Starting face detection...

Frame 30: 25.3 FPS | 1 face(s) detected
Frame 60: 26.1 FPS | 2 face(s) detected
```

### What to Look For

✅ **Success Indicators:**
- FPS: 15-30+ on Raspberry Pi 4, 30-60+ on PC
- Green boxes appear around faces
- Confidence scores > 0.5
- Smooth real-time detection

❌ **Issues to Debug:**
- No boxes: Check lighting, face angle, distance
- Low FPS (<10): Try reducing INPUT_SIZE (e.g., 256)
- Wrong boxes: Adjust CONFIDENCE_THRESHOLD or IOU_THRESHOLD

### Model Information

- **Model:** YOLOv8n-face (nano)
- **Input:** 320×320 RGB (can try 256, 416, or 640)
- **Format:** ONNX (optimized for inference)
- **Backend:** ONNX Runtime (CPU)

### Performance Tips

For Raspberry Pi 4:
- Use `INPUT_SIZE = 256` for better FPS
- Lower `CONFIDENCE_THRESHOLD = 0.4` if missing faces
- Consider reducing camera resolution to 640×480

### Next Steps

Once verification is complete:
1. ✅ YOLOv8-face ONNX model verified
2. ⏭️ Test FER classifier ONNX model (test_webcam_improved.py)
3. ⏭️ Integrate both models into main pipeline
4. ⏭️ Add motor control
5. ⏭️ Deploy to Raspberry Pi

## Troubleshooting

**Error: "Model not found"**
- The scripts look for `../yolov8n-face.onnx`
- Make sure you're running from the `final-model/` directory
- Or adjust MODEL_PATH in the script

**Error: "No cameras detected"**
- Check camera is connected and working
- Try opening camera in another app first
- On Windows, make sure drivers are installed

**Low FPS on Raspberry Pi**
- Normal: 10-20 FPS with INPUT_SIZE=320
- Try INPUT_SIZE=256 for better performance
- Use 64-bit Raspberry Pi OS
- Close other applications

**No faces detected**
- Check lighting (avoid backlight)
- Face should be within 0.5-3 meters
- Try frontal view first
- Lower CONFIDENCE_THRESHOLD

## Model Details

The YOLOv8-face model:
- Detects faces only (no landmarks)
- Returns bounding boxes + confidence scores
- Pre-trained on WIDER FACE dataset
- Optimized for edge devices

For the full pipeline, this model will:
1. Detect all faces in frame
2. Provide bounding boxes
3. Crops will be fed to FER classifier
4. Robot selects target face for tracking
