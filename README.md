# SkinGuard AI

A Flask web app that uses a PyTorch model to classify skin lesions for potential cancer detection. Upload an image, get a prediction.

## Tech Stack

- Flask for the web app
- PyTorch + torchvision for the ML model
- Pre-trained AlexNet with custom classifier

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   python app.py
   ```

4. **Open browser:** http://localhost:5000

## How it works

- Upload a skin lesion image
- The app runs it through a pre-trained model
- Get back a probability score for cancer detection
- Model uses AlexNet features + custom 3-layer classifier

## Files

- `app.py` - Main Flask application
- `model/final_model_bs32_lr0065_e22.pth` - Trained model weights
- `static/` - CSS and JavaScript
- `templates/` - HTML templates

**Note:** This is for educational purposes only, not medical advice.