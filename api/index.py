import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)


model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'blur_detection_mlp.pkl')
pipeline = joblib.load(model_path)


feature_names = ['sobel_x_var', 'sobel_mag_mean', 'tenengrad']

def extract_features_from_image(img_array):

    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    sobel_x_var = sobel_x.var()
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag_mean = sobel_magnitude.mean()
    tenengrad_mean = (sobel_x**2 + sobel_y**2).mean()
    
    return [sobel_x_var, sobel_mag_mean, tenengrad_mean]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']

    img_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    features = extract_features_from_image(img_array)
    X = np.array(features).reshape(1, -1)
    
    pred = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0][1]
    
    result = {
        'prediction': 'blurred' if pred == 1 else 'sharp',
        'confidence': float(proba)
    }
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
@app.route('/')
def home():
    return {"message": "Welcome! Server is running."}
