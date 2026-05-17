from flask import Flask, request, jsonify, render_template
import numpy as np

# =========================
# PYTORCH IMPORTS
# =========================
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import ViTModel
from PIL import Image

app = Flask(__name__)

# =========================
# LOAD CLINICAL DATA
# =========================
inputs = np.load("inputs.npy")
outputs = np.load("outputs.npy")

# =========================
# HYBRID MODEL CLASS
# =========================
class HybridViTResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(HybridViTResNet, self).__init__()

        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Identity()
        self.resnet_fc = nn.Linear(2048, 512)

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_fc = nn.Linear(768, 512)

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        resnet_features = self.resnet(x)
        resnet_features = self.resnet_fc(resnet_features)

        vit_features = self.vit(x).last_hidden_state[:, 0, :]
        vit_features = self.vit_fc(vit_features)

        combined = torch.cat((resnet_features, vit_features), dim=1)
        return self.fc(combined)

# =========================
# LOAD IMAGE MODEL
# =========================
image_model = HybridViTResNet(num_classes=5)

state_dict = torch.load("image_model.pth", map_location=torch.device('cpu'))
image_model.load_state_dict(state_dict, strict=False)

image_model.eval()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# ROUTES (IMPORTANT FLOW)
# =========================

# ✅ WELCOME PAGE FIRST
@app.route('/')
def home():
    return render_template('login.html')   # welcome page

# ✅ DASHBOARD
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# ✅ CLINICAL PAGE
@app.route('/clinical')
def clinical():
    return render_template('index.html')

# ✅ IMAGE PAGE
@app.route('/upload')
def upload():
    return render_template('upload.html')

# =========================
# CLINICAL PREDICTION
# =========================
@app.route('/predict_clinical', methods=['POST'])
def predict_clinical():
    try:
        data = request.json

        values = list(data.values())
        values = np.array(values, dtype=float)

        mean = np.mean(inputs, axis=0)
        std = np.std(inputs, axis=0)
        values = (values - mean) / (std + 1e-8)

        values = values.reshape(1, 1, -1)

        distances = np.linalg.norm(inputs - values, axis=(1, 2))
        k = 5
        idx = np.argsort(distances)[:k]

        avg_pred = np.mean(outputs[idx])

        result = "Hepatitis Detected" if avg_pred > 0.5 else "No Hepatitis"

        return jsonify({"result": result})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"result": "Error"})

# =========================
# IMAGE PREDICTION
# =========================
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        file = request.files['image']

        img = Image.open(file).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)

        labels = [
    "No Fibrosis",
    "Mild Fibrosis",
    "Moderate Fibrosis",
    "Severe Fibrosis",
    "Cirrhosis"
]

        with torch.no_grad():
            output = image_model(img)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        result = labels[pred_class]

        return jsonify({"result": result})

    except Exception as e:
        print("IMAGE ERROR:", e)
        return jsonify({"result": "Error in image"})

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)