"""
Test script for the improved YOLOv8 FER model (128×128, 3000 samples per class)
Tests on the merged dataset and compares with baseline model
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MODEL_PATH = "fer_yolov8_cls.onnx"
IMG_SIZE = 128  # New improved model uses 128×128
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']
NUM_CLASSES = len(CLASS_NAMES)

# Dataset paths - USE THE SAME STRUCTURE AS TRAINING (AffectNet-style)
DATASET_DIR = Path("../datasets")  # AffectNet structure
TEST_DIR = DATASET_DIR / "test"

# Mapping from AffectNet folder names to our class names
FOLDER_MAPPING = {
    'Anger': 'angry',
    'Happy': 'happy',
    'Neutral': 'neutral',
    'Sad': 'sad',
    'Surprise': 'surprised',
    # Skip these classes
    'Contempt': None,
    'Disgust': None,
    'Fear': None
}

print("=" * 70)
print("IMPROVED YOLOv8 FER MODEL EVALUATION")
print("=" * 70)
print(f"Model: {MODEL_PATH}")
print(f"Input size: {IMG_SIZE}×{IMG_SIZE}")
print(f"Classes: {CLASS_NAMES}")
print("=" * 70)

# Check model exists
if not os.path.exists(MODEL_PATH):
    print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
    print("Please ensure the ONNX model is in the current directory.")
    exit(1)

# Load ONNX model
print("\n📦 Loading ONNX model...")
try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✓ Model loaded successfully")
    print(f"  Input: {input_name}, shape: {session.get_inputs()[0].shape}")
    print(f"  Output: {output_name}, shape: {session.get_outputs()[0].shape}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

def preprocess_image(img_path, img_size=IMG_SIZE):
    """Preprocess image for YOLOv8 classification"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # Resize to model input size
    img = cv2.resize(img, (img_size, img_size))
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    img = np.transpose(img, (2, 0, 1))
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict(session, input_name, output_name, img_array):
    """Run inference and return predicted class index"""
    outputs = session.run([output_name], {input_name: img_array})
    probs = outputs[0][0]  # Shape: (num_classes,)
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    return pred_idx, confidence, probs

# Collect test data
print("\n📂 Loading test dataset...")
if not TEST_DIR.exists():
    print(f"❌ Test directory not found: {TEST_DIR}")
    exit(1)

y_true = []
y_pred = []
y_probs = []
inference_times = []
test_images = []

# Iterate through actual folders in test directory
for folder_name in os.listdir(TEST_DIR):
    folder_path = TEST_DIR / folder_name
    
    if not folder_path.is_dir():
        continue
    
    # Map folder name to our class name
    class_name = FOLDER_MAPPING.get(folder_name)
    
    # Skip folders not in our mapping or mapped to None
    if class_name is None or class_name not in CLASS_NAMES:
        print(f"  ⚠ Skipping {folder_name} (not in target classes)")
        continue
    
    class_idx = CLASS_NAMES.index(class_name)
    
    img_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")) + list(folder_path.glob("*.jpeg"))
    
    print(f"\n  Processing {class_name} (from {folder_name}): {len(img_files)} images...")
    
    for img_file in img_files:
        # Preprocess
        img_array = preprocess_image(img_file, IMG_SIZE)
        if img_array is None:
            continue
        
        # Inference
        start_time = time.time()
        pred_idx, confidence, probs = predict(session, input_name, output_name, img_array)
        inference_time = time.time() - start_time
        
        # Store results
        y_true.append(class_idx)
        y_pred.append(pred_idx)
        y_probs.append(probs)
        inference_times.append(inference_time)
        test_images.append(str(img_file))

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

# Overall accuracy
overall_accuracy = accuracy_score(y_true, y_pred)
print(f"\n🎯 Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

# Top-5 accuracy (for 5 classes, should be very high)
top5_correct = 0
for i in range(len(y_true)):
    top5_preds = np.argsort(y_probs[i])[-5:]
    if y_true[i] in top5_preds:
        top5_correct += 1
top5_accuracy = top5_correct / len(y_true)
print(f"🎯 Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")

# Inference speed
avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
fps = 1.0 / np.mean(inference_times)
print(f"\n⚡ Inference Speed:")
print(f"  Average: {avg_inference_time:.2f} ms/image")
print(f"  FPS: {fps:.2f}")
print(f"  Total images: {len(y_true)}")

# Per-class metrics
print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# Confusion Matrix
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Save confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=[c.upper() for c in CLASS_NAMES],
    yticklabels=[c.upper() for c in CLASS_NAMES],
    cbar_kws={'label': 'Count'}
)
plt.title(f'Confusion Matrix - Improved YOLOv8 FER ({IMG_SIZE}×{IMG_SIZE})\n'
          f'Accuracy: {overall_accuracy*100:.2f}%',
          fontsize=14, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

cm_save_path = "confusion_matrix_improved.png"
plt.savefig(cm_save_path, dpi=150, bbox_inches='tight')
print(f"\n💾 Confusion matrix saved to: {cm_save_path}")
plt.close()

# Per-class accuracy breakdown
print("\n" + "=" * 70)
print("PER-CLASS ACCURACY BREAKDOWN")
print("=" * 70)
for i, class_name in enumerate(CLASS_NAMES):
    class_mask = (y_true == i)
    class_correct = np.sum((y_true == i) & (y_pred == i))
    class_total = np.sum(class_mask)
    class_accuracy = class_correct / class_total if class_total > 0 else 0
    
    print(f"{class_name.upper():12s}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) "
          f"[{class_correct}/{class_total} correct]")

# Find worst predictions
print("\n" + "=" * 70)
print("WORST PREDICTIONS (Low Confidence)")
print("=" * 70)

# Calculate confidence for correct and incorrect predictions
correct_mask = (y_true == y_pred)
incorrect_mask = ~correct_mask

if np.sum(incorrect_mask) > 0:
    # Get confidence of predicted class for each sample
    confidences = np.array([y_probs[i][y_pred[i]] for i in range(len(y_pred))])
    
    # Find worst incorrect predictions (low confidence)
    incorrect_indices = np.where(incorrect_mask)[0]
    incorrect_confidences = confidences[incorrect_indices]
    worst_idx = incorrect_indices[np.argsort(incorrect_confidences)[:10]]  # Top 10 worst
    
    for rank, idx in enumerate(worst_idx[:5], 1):  # Show top 5
        true_class = CLASS_NAMES[y_true[idx]]
        pred_class = CLASS_NAMES[y_pred[idx]]
        conf = confidences[idx]
        img_path = Path(test_images[idx]).name
        
        print(f"{rank}. {img_path}")
        print(f"   True: {true_class.upper()}, Predicted: {pred_class.upper()}, Confidence: {conf:.4f}")
else:
    print("✓ Perfect accuracy! No incorrect predictions.")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Total test images: {len(y_true)}")
print(f"Correct predictions: {np.sum(y_true == y_pred)} ({overall_accuracy*100:.2f}%)")
print(f"Incorrect predictions: {np.sum(y_true != y_pred)} ({(1-overall_accuracy)*100:.2f}%)")
print(f"Average inference time: {avg_inference_time:.2f} ms")
print(f"Throughput: {fps:.2f} FPS")

# Compare with baseline (if available)
baseline_accuracy_path = "../training-with-64x64-7021/baseline_accuracy.txt"
if os.path.exists(baseline_accuracy_path):
    try:
        with open(baseline_accuracy_path, 'r') as f:
            baseline_acc = float(f.read().strip())
        
        improvement = (overall_accuracy - baseline_acc) * 100
        print(f"\n" + "=" * 70)
        print("COMPARISON WITH BASELINE")
        print("=" * 70)
        print(f"Baseline model accuracy: {baseline_acc*100:.2f}%")
        print(f"Improved model accuracy: {overall_accuracy*100:.2f}%")
        print(f"Improvement: {improvement:+.2f} percentage points")
    except:
        pass

# Save results
results_file = "test_results_improved.txt"
with open(results_file, 'w') as f:
    f.write("IMPROVED YOLOV8 FER MODEL TEST RESULTS\n")
    f.write("=" * 70 + "\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Input size: {IMG_SIZE}×{IMG_SIZE}\n")
    f.write(f"Test images: {len(y_true)}\n")
    f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n")
    f.write(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)\n")
    f.write(f"Average inference time: {avg_inference_time:.2f} ms\n")
    f.write(f"FPS: {fps:.2f}\n")
    f.write("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

print(f"\n💾 Results saved to: {results_file}")

print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETE")
print("=" * 70)
