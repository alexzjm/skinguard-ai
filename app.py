from flask import Flask, request, jsonify, render_template

from PIL import Image
import os

import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models

# PYTORCH INITIALIZATIONS (what if I initiate all this in the edit image stuff)
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

alexnet = torchvision.models.alexnet(pretrained=True)

processor = alexnet.features
classifier = torch.load("model/final_model_bs32_lr0065_e22.pth", map_location=torch.device('cpu'))

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

# Make the directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']): 
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    file = request.files.get('file')
    selected_model = request.form.get('model')
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)  # Save the uploaded file
        file.save(filepath)
        prediction = -1
        with Image.open(filepath) as img: # DO IMAGE OPERATIONS HERE
            if selected_model == '0':
                img = transform(img)
                prediction = float(F.sigmoid(classifier(processor(img))))
        return jsonify({'prediction': prediction, 'model_used': selected_model}) #model used is passed back to prevent inconsistancies
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(debug=True)