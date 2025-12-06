"""
Test YOLOv8 FER model on entire test dataset
Generates detailed metrics and visualizations
"""
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
MODEL_PATH = "fer_yolov8_cls_best.pt"
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# Test dataset paths (update these to match your structure)
TEST_DATASET_PATHS = [
    "../datasets/test",           # AffectNet test set
    "../datasets-fer/test",       # FER test set
    "test_data"                    # Local test folder
]

def find_test_dataset():
    """Find the test dataset directory"""
    for path in TEST_DATASET_PATHS:
        if Path(path).exists():
            print(f"✓ Found test dataset: {path}")
            return Path(path)
    
    print("⚠ No test dataset found in default locations")
    custom_path = input("Enter path to test dataset: ").strip()
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
    return None

def test_on_dataset(test_dir, model):
    """Test model on entire dataset"""
    y_true = []
    y_pred = []
    y_probs = []
    
    print("\n" + "=" * 60)
    print("TESTING ON DATASET")
    print("=" * 60)
    
    # Process each class
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = test_dir / class_name
        
        # Try different naming conventions
        possible_names = [
            class_name,
            class_name.capitalize(),
            class_name.upper(),
            "Anger" if class_name == "angry" else None,
            "Surprise" if class_name == "surprised" else None
        ]
        
        class_dir_found = None
        for name in possible_names:
            if name and (test_dir / name).exists():
                class_dir_found = test_dir / name
                break
        
        if not class_dir_found:
            print(f"⚠ Skipping {class_name}: directory not found")
            continue
        
        # Get all images
        images = list(class_dir_found.glob("*.jpg")) + \
                 list(class_dir_found.glob("*.png")) + \
                 list(class_dir_found.glob("*.jpeg"))
        
        print(f"\nProcessing {class_name}: {len(images)} images...")
        
        for img_path in tqdm(images, desc=f"  {class_name}"):
            try:
                # Predict
                results = model(str(img_path), verbose=False)
                probs = results[0].probs.data.cpu().numpy()
                pred_idx = probs.argmax()
                
                y_true.append(class_idx)
                y_pred.append(pred_idx)
                y_probs.append(probs)
            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")
                continue
    
    return np.array(y_true), np.array(y_pred), np.array(y_probs)

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix_test.png"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
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
    plt.title('Confusion Matrix - Test Set Evaluation\n(YOLOv8 64×64)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix: {save_path}")
    plt.show()

def plot_per_class_accuracy(y_true, y_pred, save_path="per_class_accuracy.png"):
    """Plot per-class accuracy"""
    accuracies = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = y_true == class_idx
        if mask.sum() > 0:
            acc = (y_pred[mask] == class_idx).sum() / mask.sum()
            accuracies.append(acc * 100)
        else:
            accuracies.append(0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(CLASS_NAMES)), accuracies, color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Emotion Class', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Per-Class Accuracy on Test Set\n(YOLOv8 64×64)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(CLASS_NAMES)), [c.upper() for c in CLASS_NAMES], rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved per-class accuracy: {save_path}")
    plt.show()

def analyze_misclassifications(y_true, y_pred):
    """Analyze common misclassification patterns"""
    print("\n" + "=" * 60)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    for true_idx, true_name in enumerate(CLASS_NAMES):
        mask = y_true == true_idx
        if mask.sum() == 0:
            continue
        
        preds = y_pred[mask]
        correct = (preds == true_idx).sum()
        total = len(preds)
        acc = correct / total * 100
        
        print(f"\n{true_name.upper()} (n={total}):")
        print(f"  ✓ Correct: {correct}/{total} ({acc:.1f}%)")
        
        # Find misclassifications
        misclassified = preds[preds != true_idx]
        if len(misclassified) > 0:
            print(f"  ✗ Misclassified as:")
            for pred_idx in range(len(CLASS_NAMES)):
                if pred_idx != true_idx:
                    count = (misclassified == pred_idx).sum()
                    if count > 0:
                        pct = count / len(misclassified) * 100
                        print(f"      • {CLASS_NAMES[pred_idx].upper()}: {count} ({pct:.1f}% of errors)")

def main():
    print("=" * 60)
    print("YOLOV8 FER MODEL - TEST SET EVALUATION")
    print("=" * 60)
    
    # Find test dataset
    test_dir = find_test_dataset()
    if not test_dir:
        print("ERROR: No test dataset found")
        return
    
    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✓ Model loaded")
    
    # Test on dataset
    y_true, y_pred, y_probs = test_on_dataset(test_dir, model)
    
    if len(y_true) == 0:
        print("ERROR: No test images found")
        return
    
    # Calculate overall accuracy
    accuracy = (y_pred == y_true).sum() / len(y_true)
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Total Test Images: {len(y_true)}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    
    # Analyze misclassifications
    analyze_misclassifications(y_true, y_pred)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Plot per-class accuracy
    plot_per_class_accuracy(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
