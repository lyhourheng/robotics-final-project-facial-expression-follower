# FER2013 Dataset

This folder will contain the FER2013 (Facial Expression Recognition 2013) dataset.

## Dataset Information

- **Total Images**: ~35,887 grayscale 48x48 face images
- **Classes**: 7 original (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Our Mapping**: 5 classes (Happy, Sad, Angry, Surprised, Neutral)
- **Split**: ~28k training, ~7k test

## Download Instructions

### Option 1: Kaggle CLI (Recommended)

1. Setup Kaggle API credentials:
   ```bash
   # Windows
   mkdir C:\Users\<username>\.kaggle
   # Place your kaggle.json here
   
   # Linux/Mac
   mkdir ~/.kaggle
   # Place your kaggle.json here
   ```

2. Download dataset:
   ```bash
   cd datasets
   kaggle datasets download -d msambare/fer2013
   unzip fer2013.zip
   ```

### Option 2: Manual Download

1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Click "Download"
3. Extract to this folder

### Option 3: Use script

```bash
python scripts/download_fer2013.py
```

## Expected Structure

```
datasets/fer2013/
├── train/
│   ├── angry/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── test/
    ├── angry/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprised/
```

## Class Mapping

| Original FER2013 | Our 5 Classes | Robot Action |
|------------------|---------------|--------------|
| Happy (3)        | Happy (0)     | Forward      |
| Sad (4)          | Sad (1)       | Turn Right   |
| Angry (0)        | Angry (2)     | Backward     |
| Surprise (5)     | Surprised (3) | Turn Left    |
| Neutral (6)      | Neutral (4)   | Stop         |
| Fear (2)         | → Drop        |  --          |
| Disgust (1)      | → Drop        |  --          |
