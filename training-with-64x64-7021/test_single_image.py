"""
Test YOLOv8 FER model with a single image
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Configuration
MODEL_PATH = "fer_yolov8_cls_best.pt"  # or .onnx
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']

def test_single_image(image_path):
    """Test model on a single image"""
    print("=" * 60)
    print("SINGLE IMAGE TEST")
    print("=" * 60)
    
    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    # Load and display image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    
    print(f"Image: {image_path}")
    print(f"Size: {img.shape[1]}×{img.shape[0]}")
    
    # Predict
    print("\nRunning inference...")
    results = model(image_path, verbose=False)
    
    # Get predictions
    probs = results[0].probs.data.cpu().numpy()
    top1_idx = probs.argmax()
    top1_conf = probs[top1_idx]
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"🎯 Predicted Emotion: {CLASS_NAMES[top1_idx].upper()}")
    print(f"✓ Confidence: {top1_conf*100:.2f}%")
    
    print("\nAll Class Probabilities:")
    for i, (name, prob) in enumerate(sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)):
        bar = "█" * int(prob * 50)
        print(f"  {name:12s}: {prob*100:6.2f}% {bar}")
    
    # Display image with prediction
    display_img = img.copy()
    cv2.putText(display_img, f"{CLASS_NAMES[top1_idx].upper()} ({top1_conf*100:.1f}%)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Prediction", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("=" * 60)
    return CLASS_NAMES[top1_idx], top1_conf

if __name__ == "__main__":
    # Test with your image
    test_image = input("Enter image path (or press Enter for default): ").strip()
    
    if not test_image:
        # Try to find a test image
        test_dirs = [
            "../datasets/test",
            "../datasets-fer/test",
            "../original-dataset"
        ]
        
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                # Find first image
                for emotion_dir in Path(test_dir).iterdir():
                    if emotion_dir.is_dir():
                        images = list(emotion_dir.glob("*.jpg")) + list(emotion_dir.glob("*.png"))
                        if images:
                            test_image = str(images[0])
                            print(f"Using default test image: {test_image}")
                            break
                if test_image:
                    break
    
    if test_image and Path(test_image).exists():
        test_single_image(test_image)
    else:
        print("No test image found. Please provide an image path.")
        print("\nUsage: python test_single_image.py")
        print("Then enter the full path to an image file.")
