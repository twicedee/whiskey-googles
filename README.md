

python run.py

python main.py evaluate --csv dataset/processed/clean_image_whisky_dataset.csv --images images/ --output results/
python main.py predict --image path/to/your/image.jpg
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