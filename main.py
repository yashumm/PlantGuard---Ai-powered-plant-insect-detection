from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os
import torch.nn.functional as F

app = FastAPI(title="Plant Insect Detection API")

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Classes (must match training order)
classes = ['aphids', 'caterpillar', 'healthy', 'whitefly']

# Class descriptions for better UX
class_info = {
    "aphids": {
        "emoji": "🐛",
        "description": "Small sap-sucking insects. Treat with neem oil or insecticidal soap.",
        "severity": "Medium"
    },
    "caterpillar": {
        "emoji": "🐛",
        "description": "Leaf-eating larvae. Use Bt spray or hand-pick for organic control.",
        "severity": "High"
    },
    "healthy": {
        "emoji": "✅",
        "description": "Plant looks healthy! No pest detected.",
        "severity": "None"
    },
    "whitefly": {
        "emoji": "🦟",
        "description": "Tiny white flying insects. Use yellow sticky traps or insecticidal soap.",
        "severity": "Medium"
    }
}

# Load model once at startup
MODEL_PATH = "plant_model.pth"

def load_model():
    # Use weights=None to avoid deprecation warning (model was trained locally)
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Run train.py first.")
    state = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m

try:
    model = load_model()
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

# ImageNet normalization transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/")
def home():
    return {
        "message": "Plant Insect Detection API Running",
        "model_loaded": model is not None,
        "classes": classes
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPG, PNG, etc.)")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        predicted_class = classes[predicted_idx.item()]
        confidence_pct = round(float(confidence.item()) * 100, 2)

        # All class probabilities
        all_probs = {
            classes[i]: round(float(probs[0][i].item()) * 100, 2)
            for i in range(len(classes))
        }

        return {
            "prediction": predicted_class,
            "confidence": confidence_pct,
            "all_probabilities": all_probs,
            "info": class_info[predicted_class]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
