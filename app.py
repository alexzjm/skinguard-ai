from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from PIL import Image
import os
import uuid

import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models

# PYTORCH INITIALIZATIONS 
class ann_classifier(nn.Module):
    def __init__(self):
        super(ann_classifier, self).__init__()
        self.name = "classifier"
        self.fc1 = nn.Linear(256 * 6 * 6, 1024) #256*6*6 is the size of AlexNet feature output
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.reshape(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

alexnet = torchvision.models.alexnet(weights='DEFAULT')

processor = alexnet.features
classifier = torch.load("model/final_model_bs32_lr0065_e22.pth", map_location=torch.device('cpu'), weights_only=False)

processor.eval()
classifier.eval()

transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#FLASK INITIALZATIONS
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/' #link to upload folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# File validation configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_FILE_SIZE = 1024  # 1KB

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(file):
    """Comprehensive file validation"""
    # Check if file exists
    if not file or file.filename == '':
        return False, 'No file selected'
    
    # Check file extension
    if not allowed_file(file.filename):
        return False, 'Invalid file type. Please upload: PNG, JPG, JPEG, GIF, BMP, or WebP'
    
    # Check file size by reading content
    file.seek(0, os.SEEK_END)  # Move to end of file
    file_size = file.tell()     # Get file size
    file.seek(0)               # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        return False, f'File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB'
    
    if file_size < MIN_FILE_SIZE:
        return False, 'File too small. Please upload a valid image'
    
    return True, 'Valid file'

# Make the directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']): 
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.errorhandler(413)
def too_large(e):
    """Handle files that exceed Flask's MAX_CONTENT_LENGTH"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    filepath = None
    try:
        # Validate request has file
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided in request'}), 400
        
        file = request.files['file']
        selected_model = request.form.get('model', '0')
        
        # Validate file
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Validate model selection
        if selected_model != '0':
            return jsonify({'error': 'Invalid model selection. Only model 0 is available.'}), 400
        
        # Generate secure filename with UUID to prevent conflicts
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file with error handling
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Validate saved file exists and is readable
        if not os.path.exists(filepath):
            return jsonify({'error': 'File upload failed'}), 500
        
        # Process image with error handling
        try:
            with Image.open(filepath) as img:
                # Validate image can be opened and processed
                img.verify()  # Check if image is valid
                
            # Reopen image after verify (verify() closes the image)
            with Image.open(filepath) as img:
                # Convert and transform image
                try:
                    processed_img = transform(img)
                    
                    # Run AI model prediction
                    with torch.no_grad():  # Disable gradient computation for inference
                        features = processor(processed_img.unsqueeze(0))  # Add batch dimension
                        prediction = float(F.sigmoid(classifier(features)))
                    
                    # Validate prediction is a valid number
                    if not isinstance(prediction, (int, float)) or prediction < 0 or prediction > 1:
                        raise ValueError("Invalid prediction value")
                        
                except Exception as e:
                    return jsonify({'error': 'AI model processing failed. Please try a different image.'}), 500
                
        except Exception as e:
            if "cannot identify image file" in str(e).lower():
                return jsonify({'error': 'Invalid or corrupted image file'}), 400
            elif "image file is truncated" in str(e).lower():
                return jsonify({'error': 'Image file is incomplete or corrupted'}), 400
            else:
                return jsonify({'error': 'Image processing failed. Please ensure the file is a valid image.'}), 400
        
        # Clean up uploaded file after processing
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            # Log the error but don't fail the request
            print(f"Warning: Failed to delete temporary file {filepath}: {e}")
        
        return jsonify({
            'prediction': prediction, 
            'model_used': selected_model,
            'status': 'success'
        }), 200
        
    except Exception as e:
        # Clean up file on any error
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        # Handle specific Flask errors
        if hasattr(e, 'code') and e.code == 413:
            return jsonify({'error': 'File too large for server'}), 413
        
        # Generic server error
        print(f"Unexpected error in upload_image: {e}")
        return jsonify({'error': 'Internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)