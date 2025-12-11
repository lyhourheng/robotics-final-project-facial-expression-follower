"""
Compare baseline (96×96, 1200 samples) vs improved (128×128, 3000 samples) models
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Configuration
BASELINE_MODEL = "../training-with-64x64-7021/fer_yolov8_cls.onnx"
IMPROVED_MODEL = "fer_yolov8_cls.onnx"
BASELINE_SIZE = 96  # or 64 depending on your baseline
IMPROVED_SIZE = 128
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# Dataset path - USE THE SAME DATASET STRUCTURE AS TRAINING
# Option 1: AffectNet-style (Anger, Happy, Neutral, Sad, Surprise)
TEST_DIR = Path("../datasets/test")

# Mapping from AffectNet folder names to our class names
FOLDER_MAPPING = {
    'Anger': 'angry',
    'Happy': 'happy', 
    'Neutral': 'neutral',
    'Sad': 'sad',
    'Surprise': 'surprised',
    # Ignore these classes
    'Contempt': None,
    'Disgust': None,
    'Fear': None
}

print("=" * 70)
print("MODEL COMPARISON: BASELINE vs IMPROVED")
print("=" * 70)
print(f"Baseline model: {BASELINE_MODEL} ({BASELINE_SIZE}×{BASELINE_SIZE})")
print(f"Improved model: {IMPROVED_MODEL} ({IMPROVED_SIZE}×{IMPROVED_SIZE})")
print("=" * 70)

# Check models exist
if not os.path.exists(BASELINE_MODEL):
    print(f"\n⚠ Warning: Baseline model not found at {BASELINE_MODEL}")
    print("Skipping baseline comparison...")
    COMPARE_BASELINE = False
else:
    COMPARE_BASELINE = True

if not os.path.exists(IMPROVED_MODEL):
    print(f"\n❌ ERROR: Improved model not found at {IMPROVED_MODEL}")
    exit(1)

def load_model(model_path):
    """Load ONNX model"""
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

def preprocess_image(img_path, img_size):
    """Preprocess image for YOLOv8"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img

def evaluate_model(session, input_name, output_name, test_dir, img_size, model_name):
    """Evaluate model on test set"""
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}")
    
    y_true = []
    y_pred = []
    inference_times = []
    
    # Iterate through actual folders in test directory
    for folder_name in os.listdir(test_dir):
        folder_path = test_dir / folder_name
        
        if not folder_path.is_dir():
            continue
        
        # Map folder name to our class name
        class_name = FOLDER_MAPPING.get(folder_name)
        
        # Skip folders not in our mapping or mapped to None
        if class_name is None or class_name not in CLASS_NAMES:
            print(f"  Skipping {folder_name} (not in target classes)")
            continue
        
        class_idx = CLASS_NAMES.index(class_name)
        
        img_files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        print(f"  {class_name} (from {folder_name}): {len(img_files)} images")
        
        for img_file in img_files:
            img_array = preprocess_image(img_file, img_size)
            if img_array is None:
                continue
            
            start_time = time.time()
            outputs = session.run([output_name], {input_name: img_array})
            inference_time = time.time() - start_time
            
            probs = outputs[0][0]
            pred_idx = np.argmax(probs)
            
            y_true.append(class_idx)
            y_pred.append(pred_idx)
            inference_times.append(inference_time)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    avg_inference = np.mean(inference_times) * 1000  # ms
    fps = 1.0 / np.mean(inference_times)
    
    print(f"\n📊 Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Inference: {avg_inference:.2f} ms/image")
    print(f"  FPS: {fps:.2f}")
    print(f"  Test samples: {len(y_true)}")
    
    return {
        'accuracy': accuracy,
        'inference_time': avg_inference,
        'fps': fps,
        'y_true': np.array(y_true),
        'y_pred': np.array(y_pred),
        'total_samples': len(y_true)
    }

# Load models
print("\n📦 Loading models...")
improved_session, improved_input, improved_output = load_model(IMPROVED_MODEL)
print(f"✓ Improved model loaded")

if COMPARE_BASELINE:
    baseline_session, baseline_input, baseline_output = load_model(BASELINE_MODEL)
    print(f"✓ Baseline model loaded")

# Evaluate improved model
improved_results = evaluate_model(
    improved_session, improved_input, improved_output,
    TEST_DIR, IMPROVED_SIZE, "Improved Model"
)

# Evaluate baseline model
if COMPARE_BASELINE:
    baseline_results = evaluate_model(
        baseline_session, baseline_input, baseline_output,
        TEST_DIR, BASELINE_SIZE, "Baseline Model"
    )
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    accuracy_diff = (improved_results['accuracy'] - baseline_results['accuracy']) * 100
    speed_diff = improved_results['inference_time'] - baseline_results['inference_time']
    
    print(f"\n🎯 Accuracy:")
    print(f"  Baseline: {baseline_results['accuracy']*100:.2f}%")
    print(f"  Improved: {improved_results['accuracy']*100:.2f}%")
    print(f"  Difference: {accuracy_diff:+.2f} percentage points")
    
    print(f"\n⚡ Speed:")
    print(f"  Baseline: {baseline_results['inference_time']:.2f} ms ({baseline_results['fps']:.2f} FPS)")
    print(f"  Improved: {improved_results['inference_time']:.2f} ms ({improved_results['fps']:.2f} FPS)")
    print(f"  Difference: {speed_diff:+.2f} ms")
    
    print(f"\n📈 Trade-off:")
    if accuracy_diff > 0 and speed_diff < 0:
        print(f"  ✅ Improved model is BETTER: +{accuracy_diff:.2f}% accuracy & {-speed_diff:.2f} ms faster")
    elif accuracy_diff > 0:
        print(f"  ⚖️ Accuracy improved by {accuracy_diff:.2f}%, but {speed_diff:.2f} ms slower")
    else:
        print(f"  ⚠️ Mixed results - check per-class performance")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    models = ['Baseline\n(96×96, 1.2k)', 'Improved\n(128×128, 3k)']
    accuracies = [baseline_results['accuracy']*100, improved_results['accuracy']*100]
    colors = ['#3498db', '#2ecc71']
    
    axes[0].bar(models, accuracies, color=colors, alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([min(accuracies)-5, 100])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        axes[0].text(i, acc + 1, f'{acc:.2f}%', ha='center', fontweight='bold')
    
    # Speed comparison
    inference_times = [baseline_results['inference_time'], improved_results['inference_time']]
    
    axes[1].bar(models, inference_times, color=colors, alpha=0.8)
    axes[1].set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    axes[1].set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, (model, time_val) in enumerate(zip(models, inference_times)):
        axes[1].text(i, time_val + 0.5, f'{time_val:.2f} ms', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Comparison plot saved to: model_comparison.png")
    plt.close()
    
    # Per-class comparison
    print(f"\n{'='*70}")
    print("PER-CLASS ACCURACY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Class':<12} {'Baseline':<12} {'Improved':<12} {'Difference'}")
    print("-" * 60)
    
    for i, class_name in enumerate(CLASS_NAMES):
        baseline_mask = (baseline_results['y_true'] == i)
        improved_mask = (improved_results['y_true'] == i)
        
        baseline_acc = np.sum((baseline_results['y_true'] == i) & (baseline_results['y_pred'] == i)) / np.sum(baseline_mask) if np.sum(baseline_mask) > 0 else 0
        improved_acc = np.sum((improved_results['y_true'] == i) & (improved_results['y_pred'] == i)) / np.sum(improved_mask) if np.sum(improved_mask) > 0 else 0
        
        diff = (improved_acc - baseline_acc) * 100
        
        print(f"{class_name:<12} {baseline_acc*100:>6.2f}%      {improved_acc*100:>6.2f}%      {diff:>+6.2f}%")

else:
    print("\n✓ Improved model evaluation complete")
    print(f"  Accuracy: {improved_results['accuracy']*100:.2f}%")
    print(f"  Speed: {improved_results['inference_time']:.2f} ms ({improved_results['fps']:.2f} FPS)")

print(f"\n{'='*70}")
print("✅ COMPARISON COMPLETE")
print(f"{'='*70}")
