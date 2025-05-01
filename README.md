##Whiskey googles
This is a deep learning system that identifies whiskey brands from bottle images using text recognition, visual features, and hybrid approaches.
## Features
- Multi-model recognition (Text, Visual, Hybrid)
- Dataset evaluation capabilities
- Single image prediction
- Adjustable confidence thresholds
- Detailed performance metrics

## Prerequisites
- Python 3.8+
- pip package manager
- Git (optional)

## Installation
1. Clone the repository:
```git clone https://github.com/yourusername/whiskey-googles.git```
```cd whiskey-googles```

2. Create and activate a virtual environment
```python -m venv env```
  # Windows:
```env\Scripts\activate```
  # Mac/Linux:
```source env/bin/activate```
3. Install dependencies
```pip install -r requirements.txt```
4. For irst time setup
```python run.py```

```python main.py evaluate --csv dataset/processed/clean_image_whisky_dataset.csv --images images/ --output results/```


```python main.py predict --image path/to/your/image.jpg```
```
whiskey-googles
├─ dataset
│  └─ raw
│     └─ 501 Bottle Dataset - Sheet1.csv
├─ main.py
├─ models
│  └─ deep_learning
│     ├─ hybrid_recognizer.py
│     ├─ text_recognizer.py
│     └─ visual_recognizer.py
├─ README.md
├─ requirements.txt
├─ run.py
├─ setup.py
├─ src
│  ├─ app
│  ├─ data
│  │  ├─ data_analysis.py
│  │  └─ download_images.py
│  ├─ features
│  │  ├─ text_features.py
│  │  └─ visual_features.py
│  ├─ training
│  │  └─ train_models.py
│  └─ utils
│     ├─ confidence.py
│     ├─ predictions.py
│     └─ __init__.py
└─ __init__.py

```
