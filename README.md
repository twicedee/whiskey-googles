## Whiskey Goggles 🥃🤓

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

This is a deep learning system that identifies whiskey brands from bottle images using text recognition, visual features, and hybrid approaches.


## Features✨
- Multi-model recognition (Text, Visual, Hybrid)
- Dataset evaluation capabilities
- Single image prediction
- Detailed performance metrics

## 🛠️ Tech Stack

### Core Frameworks
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?logo=tensorflow&logoColor=white)]()
[![Keras](https://img.shields.io/badge/Keras-3.9.2-D00000?logo=keras&logoColor=white)]()

### Vision & Text
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-5C3EE8?logo=opencv&logoColor=white)]()
[![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7.2-000000)]()

### Data Science
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-150458?logo=pandas&logoColor=white)]()
[![NumPy](https://img.shields.io/badge/NumPy-2.1.3-013243?logo=numpy&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikit-learn&logoColor=white)]()

### Utilities
[![tqdm](https://img.shields.io/badge/tqdm-4.67.1-FFC107)]()
[![Jellyfish](https://img.shields.io/badge/Jellyfish-1.2.0-8A9BCC)]()

---
**Integration Flow**  
![Pipeline](https://img.shields.io/badge/PyTorch→OpenCV→TensorFlow-EE4C2C/5C3EE8/FF6F00?style=flat)  
![Text Flow](https://img.shields.io/badge/EasyOCR→Pandas→Jellyfish-000000/150458/8A9BCC?style=flat)


## ⚙️ Prerequisites
- Python 3.8+
- pip package manager
- Git (optional)

## 🛠️Installation
1. Clone the repository:
```git clone https://github.com/twicedee/whiskey-googles.git```

2. Navigate into the repository
```cd whiskey-googles```

3. Create and activate a virtual environment
```python -m venv env```
   - On Windows:
```env\Scripts\activate.bat```
   - On Mac/Linux:
```source env/bin/activate```

4. Install dependencies
```pip install -r requirements.txt```
5. For first time setup, run this command
```python run.py```


## 🚀 Usage 
To run an evaluation on the whole dataset;

```python main.py evaluate --csv dataset/processed/clean_image_whisky_dataset.csv --images images/ --output results/```

To predict a single image

```python main.py predict --image sample_images\test_whiskey.jpg```


## 📂 Project Structure
```
whiskey-googles
├─ dataset
│  └─ raw
│     └─ 501 Bottle Dataset - Sheet1.csv 
├─ main.py                                   # Main evaluation/prediction script
├─ models
│  └─ deep_learning
│     ├─ hybrid_recognizer.py
│     ├─ text_recognizer.py
│     └─ visual_recognizer.py
├─ README.md
├─ requirements.txt                           # Python dependencies
├─ run.py                                     # Full pipeline runner
├─ setup.py
├─ src
│  ├─ data                                    # Data processing
│  │  ├─ data_analysis.py
│  │  └─ download_images.py
│  ├─ features                                # Feature extraction
│  │  ├─ text_features.py
│  │  └─ visual_features.py
│  ├─ training                                # Model training
│    └─ train_models.py
└─ __init__.py

```

## 🧪 Results Interpretation 
- ```results/model_prediction.csv``` gives detailed predictions for each sample with confidence scores
- ```results/evaluation_metrics.json``` calculates the accuracy metrics
- The following metrics were achieved on a test set of 497 whiskey bottle images:

### 📊 Model Performance Metrics 

| Model        | Accuracy | Total Samples Tested |
|--------------|----------|----------------------|
| Text         | 88.13%   | 497                  |
| Visual       | 97.99%   | 497                  |
| Hybrid       | 97.38%   | 497                  |

These results demonstrate our hybrid approach combining text and visual recognition achieves near-human level accuracy in whiskey identification.

## 🐛 Troubleshooting
- **Missing dependencies:** Run pip install -r requirements.txt
- **Image not found:** Verify file paths exist
- **Model errors:** Check models/trained_models/ exists
- **Float division by zero:** Occurs when text detection fails. Ensure the bottle label is clear and unobstructed.

## 🏆 Best Practices 
✅ **Lighting:** Use well-lit, front-facing images.  
✅ **Angles:** Capture the label clearly (avoid glare).  
✅ **Supported Brands:** Only compatible with the bottles in the dataset 

  
## 🎥 Quick Demo  
<video src=".github/assets/demo.mp4" width="800" controls muted poster=".github/assets/poster.jpg">
  Your browser doesn't support video. [Download instead](.github/assets/demo.mp4).
</video>

[Watch demo](.github/assets/demo.mp4)
