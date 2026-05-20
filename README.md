# PlantGuard — Flask Web Application

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Place these files in the same folder as app.py:
- `plant_disease_model.tflite`
- `class_indices.json`

3. Run the app:
```
python app.py
```

4. Open your browser at:
```
http://localhost:5000
```

## File Structure
```
flask_app/
├── app.py
├── requirements.txt
├── plant_disease_model.tflite   ← copy from Kaggle output
├── class_indices.json           ← copy from Kaggle output
└── templates/
    └── index.html
```

## Notes
- The app uses the TFLite model for inference — no GPU required
- Images are processed at 380×380 pixels to match EfficientNetB4 input
- Maximum upload size is 10MB
- Supported formats: JPG, JPEG, PNG, BMP
