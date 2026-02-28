import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Model loading
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'blur_detection_mlp.pkl')
pipeline = joblib.load(model_path)

# Shared CSS/HTML Header for the "Iconic" Look
UI_HEADER = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionAI | Image Classifier</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; background: #0f172a; color: white; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }
        .glass { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 2rem; }
        .gradient-text { background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .upload-zone { border: 2px dashed rgba(255, 255, 255, 0.2); transition: all 0.3s ease; }
        .upload-zone:hover { border-color: #38bdf8; background: rgba(56, 189, 248, 0.05); }
    </style>
</head>
<body>
    <div class="max-w-xl w-full glass p-8 md:p-12 shadow-2xl">
'''

UI_FOOTER = '</div></body></html>'

def extract_features_from_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x_var = sobel_x.var()
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag_mean = sobel_magnitude.mean()
    tenengrad_mean = (sobel_x**2 + sobel_y**2).mean()
    return [sobel_x_var, sobel_mag_mean, tenengrad_mean]

@app.route('/')
def home():
    content = '''
        <header class="text-center mb-10">
            <div class="inline-block p-4 bg-blue-500/10 rounded-2xl mb-4 text-blue-400">
                <svg xmlns="http://www.w3.org" class="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                </svg>
            </div>
            <h1 class="text-4xl font-extrabold mb-2 tracking-tight">Vision<span class="gradient-text">AI</span></h1>
            <p class="text-slate-400">Blur & Sharpness Classifier</p>
        </header>

        <form action="/predict" method="post" enctype="multipart/form-data">
            <label class="upload-zone rounded-3xl p-12 flex flex-col items-center justify-center cursor-pointer group">
                <svg xmlns="http://www.w3.org" class="h-14 w-14 text-slate-500 group-hover:text-blue-400 mb-4 transition-all" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <span class="text-slate-300 font-medium">Choose an image to analyze</span>
                <input type="file" name="image" class="hidden" accept="image/*" onchange="this.form.submit()">
            </label>
        </form>
    '''
    return UI_HEADER + content + UI_FOOTER

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    
    file = request.files['image']
    img_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if img_array is None:
        return "Invalid image", 400
    
    features = extract_features_from_image(img_array)
    X = np.array(features).reshape(1, -1)
    
    pred = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0][1]
    
    label = 'BLURRED' if pred == 1 else 'SHARP'
    color = 'text-rose-400' if label == 'BLURRED' else 'text-emerald-400'
    bg_color = 'bg-rose-500/10' if label == 'BLURRED' else 'bg-emerald-500/10'

    result_html = f'''
        <div class="text-center">
            <div class="inline-block px-6 py-2 {bg_color} {color} rounded-full font-bold tracking-widest text-sm mb-6">
                ANALYSIS COMPLETE
            </div>
            <h2 class="text-6xl font-black mb-4 {color}">{label}</h2>
            <p class="text-slate-400 text-lg mb-8 tracking-wide">Confidence Score: <span class="text-white font-semibold">{proba:.2%}</span></p>
            
            <a href="/" class="inline-block w-full py-4 bg-white/10 hover:bg-white/20 rounded-2xl transition-all font-semibold tracking-wide">
                ← ANALYZE ANOTHER IMAGE
            </a>
        </div>
    '''
    return UI_HEADER + result_html + UI_FOOTER

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
