"""
Master test script for YOLOv8 FER model
Choose which test to run
"""
import sys
from pathlib import Path

print("=" * 60)
print("YOLOV8 FER MODEL - TEST SUITE")
print("=" * 60)
print("\nAvailable Tests:")
print("  1. Single Image Test - Quick verification")
print("  2. Test Set Evaluation - Complete metrics")
print("  3. Webcam Test - Real-time detection")
print("  4. Run All Tests")
print("\n" + "=" * 60)

choice = input("Select test (1-4): ").strip()

if choice == "1":
    print("\n▶ Running Single Image Test...")
    import test_single_image
    
elif choice == "2":
    print("\n▶ Running Test Set Evaluation...")
    import test_on_dataset
    test_on_dataset.main()
    
elif choice == "3":
    print("\n▶ Running Webcam Test...")
    import test_webcam
    test_webcam.test_webcam()
    
elif choice == "4":
    print("\n▶ Running All Tests...")
    
    # Test 1: Single Image
    print("\n" + "="*60)
    print("TEST 1: SINGLE IMAGE")
    print("="*60)
    import test_single_image
    
    # Test 2: Full Dataset
    print("\n" + "="*60)
    print("TEST 2: FULL DATASET EVALUATION")
    print("="*60)
    import test_on_dataset
    test_on_dataset.main()
    
    # Test 3: Webcam
    print("\n" + "="*60)
    print("TEST 3: WEBCAM (Press 'q' to skip)")
    print("="*60)
    import test_webcam
    test_webcam.test_webcam()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
else:
    print("Invalid choice")
