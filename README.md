# VisionAI: Image Blur & Sharpness Classifier

VisionAI is a modern, high-performance web application designed to detect image quality. Using advanced Machine Learning and Computer Vision techniques, it accurately classifies images as either **Sharp** or **Blurred**.

---

## Key Features
- **Modern Dark UI:** Iconic glassmorphism design with a seamless user experience.
- **MLP Classifier:** Powered by a Multi-Layer Perceptron model trained on extracted image features.
- **Real-time Analysis:** Instant feedback on image clarity with confidence scoring.
- **Responsive Design:** Optimized for both desktop and mobile viewing.

---

## Technology Stack
- **Frontend:** HTML5, Tailwind CSS (Modern Glass UI)
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn (MLP Classifier), Joblib
- **Computer Vision:** OpenCV (Sobel, Tenengrad Feature Extraction)
- **Deployment:** Vercel (Serverless Functions)

---

## How it Works
The application uses **Feature Engineering** to analyze the input image:
1. **Sobel Variance:** Measures the intensity of edges.
2. **Tenengrad Mean:** Evaluates the focus level of the image.
3. **ML Prediction:** These features are fed into a pre-trained MLP model to determine the final class.

---

## Getting Started

### Prerequisites
- Python 3.9+
- A [Vercel](https://vercel.com) account for deployment.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com
2. Install dependencies:
```pip install -r requirements.txt```
3. Run locally :
```python api/index.py```

## Live Demo: [View VisionAI App](https://image-blur-sharp-classification-r85.vercel.app/)
![فيديو VisionNeural](VisionNeural%20Record%20.mp4)
---


If you find this project useful or like the UI, feel free to **drop a ⭐** on this repository. It means a lot! 

---

