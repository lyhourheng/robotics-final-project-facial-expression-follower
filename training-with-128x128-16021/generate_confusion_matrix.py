"""
Generate Confusion Matrix for YOLOv8 FER Model
Evaluates the trained ONNX model on test dataset and creates confusion matrix visualization
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import onnxruntime as ort
from tqdm import tqdm

# ============== CONFIGURATION ==============
MODEL_PATH = "fer_yolov8_cls.onnx"  # ONNX model path
TEST_DATA_DIR = '../datasets/test'  # Path to test dataset
IMG_SIZE = 128  # Model input size
CLASS_NAMES = ['angry', 'happy', 'neutral', 'sad', 'surprised']
OUTPUT_DIR = 'evaluation_results'

# Mapping from dataset folder names to target classes
FOLDER_MAPPING = {
    'Anger': 'angry',
    'Happy': 'happy',
    'Neutral': 'neutral',
    'Sad': 'sad',
    'Surprise': 'surprised',
    # Ignore these classes
    'Contempt': None,
    'Disgust': None,
    'Fear': None,
}

# ============== HELPER FUNCTIONS ==============

def preprocess_image(img_path, img_size=128):
    """
    Preprocess image for ONNX model inference
    
    Args:
        img_path: Path to image file
        img_size: Target size for resizing
    
    Returns:
        Preprocessed image tensor (1, 3, H, W) and original image
    """
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    
    # Resize
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] and transpose to (C, H, W)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch, img_resized


def load_test_data(test_dir, folder_mapping):
    """
    Load test dataset paths and labels
    
    Args:
        test_dir: Path to test directory
        folder_mapping: Dictionary mapping folder names to class names
    
    Returns:
        List of (image_path, class_index) tuples
    """
    test_data = []
    
    for folder_name in os.listdir(test_dir):
        folder_path = os.path.join(test_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # Get target class name
        target_class = folder_mapping.get(folder_name)
        
        if target_class is None:
            print(f"Skipping folder: {folder_name} (not in target classes)")
            continue
        
        if target_class not in CLASS_NAMES:
            print(f"Warning: {target_class} not in CLASS_NAMES")
            continue
        
        class_idx = CLASS_NAMES.index(target_class)
        
        # Get all image files
        img_files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in img_files:
            img_path = os.path.join(folder_path, img_file)
            test_data.append((img_path, class_idx))
        
        print(f"Loaded {folder_name} → {target_class}: {len(img_files)} images")
    
    return test_data


def evaluate_model(model_path, test_data, img_size=128):
    """
    Evaluate model on test data
    
    Args:
        model_path: Path to ONNX model
        test_data: List of (image_path, class_index) tuples
        img_size: Input image size
    
    Returns:
        y_true, y_pred arrays
    """
    # Load ONNX model
    print(f"\nLoading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")
    
    y_true = []
    y_pred = []
    
    print(f"\nEvaluating on {len(test_data)} images...")
    
    for img_path, true_label in tqdm(test_data, desc="Processing"):
        # Preprocess image
        img_tensor, _ = preprocess_image(img_path, img_size)
        
        if img_tensor is None:
            continue
        
        # Run inference
        outputs = session.run([output_name], {input_name: img_tensor})
        
        # Get predicted class
        probs = outputs[0][0]  # Shape: (num_classes,)
        pred_label = np.argmax(probs)
        
        y_true.append(true_label)
        y_pred.append(pred_label)
    
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """
    Generate and save confusion matrix visualization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_dir: Output directory for saving plot
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[c.upper() for c in class_names],
        yticklabels=[c.upper() for c in class_names],
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - YOLOv8 FER Classifier (ONNX)',
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix: {cm_path}")
    
    plt.show()
    
    return cm


def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Overall accuracy
    accuracy = np.mean(y_pred == y_true)
    print(f"OVERALL TEST ACCURACY: {accuracy*100:.2f}%")
    print("=" * 60)


def main():
    """Main execution function"""
    
    print("=" * 60)
    print("YOLOv8 FER - CONFUSION MATRIX GENERATOR")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please ensure the ONNX model file exists.")
        return
    
    # Check if test data exists
    if not os.path.exists(TEST_DATA_DIR):
        print(f"ERROR: Test data directory not found at {TEST_DATA_DIR}")
        print("Please check the TEST_DATA_DIR path.")
        return
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Test Data: {TEST_DATA_DIR}")
    print(f"  Image Size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Classes: {CLASS_NAMES}")
    
    # Load test data
    print("\n" + "=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)
    test_data = load_test_data(TEST_DATA_DIR, FOLDER_MAPPING)
    
    if len(test_data) == 0:
        print("ERROR: No test data loaded!")
        return
    
    print(f"\nTotal test images: {len(test_data)}")
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION")
    print("=" * 60)
    y_true, y_pred = evaluate_model(MODEL_PATH, test_data, IMG_SIZE)
    
    print(f"\nEvaluated {len(y_true)} images successfully")
    
    # Print classification report
    print_classification_report(y_true, y_pred, CLASS_NAMES)
    
    # Generate confusion matrix
    print("\n" + "=" * 60)
    print("GENERATING CONFUSION MATRIX")
    print("=" * 60)
    cm = plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, OUTPUT_DIR)
    
    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
            print(f"  {class_name.upper():12s}: {class_acc*100:.2f}% ({class_mask.sum()} images)")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
