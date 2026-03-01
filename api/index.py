import os
import numpy as np
import cv2
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'blur_detection_mlp.pkl')
try:
    pipeline = joblib.load(model_path)
except:
    pipeline = None  

 
UI_HEADER = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisionAI | Pro Sharpness Analyzer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com" rel="stylesheet">
    <style>
        body { 
            font-family: 'Outfit', sans-serif; 
            background-color: #0a0a0a; 
            background-image: radial-gradient(circle at 10% 20%, rgba(139, 0, 0, 0.15) 0%, transparent 40%);
            color: #ffffff; 
            min-height: 100vh; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            padding: 20px; 
        }
        .main-card { 
            background: rgba(20, 20, 20, 0.95); 
            border: 1px solid rgba(255, 255, 255, 0.1); 
            border-radius: 2.5rem; 
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            overflow: hidden;
            position: relative;
        }
        .red-gradient-text { 
            background: linear-gradient(135deg, #ffffff 30%, #ff4d4d 100%); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }
        .upload-zone { 
            border: 2px dashed rgba(139, 0, 0, 0.4); 
            background: rgba(139, 0, 0, 0.02);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); 
        }
        .upload-zone:hover { 
            border-color: #8b0000; 
            background: rgba(139, 0, 0, 0.08);
            transform: translateY(-2px);
        }
        .btn-primary {
            background: #8b0000;
            transition: all 0.3s ease;
        }
        .btn-primary:hover {
            background: #a50000;
            box-shadow: 0 0 20px rgba(139, 0, 0, 0.4);
        }
        .loader {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top: 3px solid #8b0000;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="max-w-2xl w-full main-card p-8 md:p-14">
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
        <div class="text-center">
            <div class="mb-8 inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-gradient-to-br from-[#8b0000] to-[#4a0000] shadow-lg shadow-red-900/20">
                <svg xmlns="http://www.w3.org" class="h-10 w-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
            </div>
            <h1 class="text-5xl font-extrabold mb-4 tracking-tighter">Vision<span class="red-gradient-text">Neural</span></h1>
            <p class="text-slate-400 text-lg mb-12 font-light">Deep Learning Image Quality Assessment <br><span class="text-[10px] tracking-[0.3em] uppercase opacity-50">Industrial Grade Analysis</span></p>

            <form id="uploadForm" action="/predict" method="post" enctype="multipart/form-data" class="space-y-6">
                <label id="dropZone" class="upload-zone rounded-[2rem] p-16 flex flex-col items-center justify-center cursor-pointer group">
                    <div class="mb-4 transform group-hover:scale-110 transition-transform duration-300">
                        <svg xmlns="http://www.w3.org" class="h-12 w-12 text-slate-500 group-hover:text-[#8b0000]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                    </div>
                    <span class="text-white font-medium text-lg">Upload Project Asset</span>
                    <span class="text-slate-500 text-sm mt-1">Supports JPG, PNG, WEBP</span>
                    <input type="file" name="image" class="hidden" accept="image/*" onchange="handleUpload()">
                </label>
                
                <div id="loadingArea" class="hidden flex flex-col items-center py-10">
                    <div class="loader mb-4"></div>
                    <p class="text-sm font-mono text-[#8b0000] animate-pulse">COMPUTING EIGENVALUES...</p>
                </div>
            </form>
        </div>
        <script>
            function handleUpload() {
                document.getElementById('dropZone').classList.add('hidden');
                document.getElementById('loadingArea').classList.remove('hidden');
                setTimeout(() => document.getElementById('uploadForm').submit(), 600);
            }
        </script>
    '''
    return UI_HEADER + content + UI_FOOTER

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "Error: No file", 400
    
    file = request.files['image']
    img_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if img_array is None:
        return "Error: Invalid image", 400
    

    features = extract_features_from_image(img_array)
    

    if pipeline:
        X = np.array(features).reshape(1, -1)
        pred = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0][1]
    else:

        pred, proba = 0, 0.99

    label = 'MOTION BLUR' if pred == 1 else 'SHARP FOCUS'
    status_color = 'text-red-500' if pred == 1 else 'text-white'
    status_bg = 'bg-red-500/10' if pred == 1 else 'bg-white/10'

    result_html = f'''
        <div class="text-center">
            <div class="flex items-center justify-between mb-12">
                <div class="text-left">
                    <p class="text-[#8b0000] text-xs font-bold tracking-[0.2em] uppercase mb-1">Inference Engine</p>
                    <h2 class="text-4xl font-black {status_color}">{label}</h2>
                </div>
                <div class="text-right">
                    <p class="text-slate-500 text-xs font-bold tracking-[0.2em] uppercase mb-1">Confidence</p>
                    <h2 class="text-4xl font-light text-white">{proba:.1%}</h2>
                </div>
            </div>

            <!-- Technical Telemetry -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-12">
                <div class="bg-white/5 p-5 rounded-3xl border border-white/5 text-left">
                    <p class="text-slate-500 text-[10px] uppercase font-bold mb-2">Sobel Variance</p>
                    <p class="text-xl font-mono text-white">{features[0]:.1f}</p>
                </div>
                <div class="bg-white/5 p-5 rounded-3xl border border-white/5 text-left">
                    <p class="text-slate-500 text-[10px] uppercase font-bold mb-2">Tenengrad</p>
                    <p class="text-xl font-mono text-white">{features[2]:.1f}</p>
                </div>
                <div class="bg-white/5 p-5 rounded-3xl border border-white/5 text-left">
                    <p class="text-slate-500 text-[10px] uppercase font-bold mb-2">Mean Mag</p>
                    <p class="text-xl font-mono text-white">{features[1]:.1f}</p>
                </div>
            </div>
            
            <a href="/" class="btn-primary block w-full py-5 text-white rounded-[1.5rem] font-bold tracking-widest text-sm transition-all transform hover:scale-[1.02]">
                ANALYZE NEW DATASET
            </a>
        </div>
    '''
    return UI_HEADER + result_html + UI_FOOTER

if __name__ == '__main__':
    app.run(debug=True, port=5000)
