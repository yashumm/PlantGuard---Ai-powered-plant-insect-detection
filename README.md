PlantGuard – Plant Insect Detection

About the Project

PlantGuard is an AI-based plant insect detection system designed to identify common plant-related conditions and insects from uploaded images. It uses deep learning and image classification to help users recognize possible insect infestations and distinguish healthy plants from affected ones.

Features

- Upload plant images for analysis
- AI-based image classification
- Detects Aphids
- Detects Caterpillar
- Detects Whitefly
- Identifies Healthy plant images
- Web-based user interface
- FastAPI-based backend
- Deep learning model using PyTorch and ResNet18

Technology Stack

- Python
- FastAPI
- PyTorch
- Torchvision
- ResNet18
- HTML
- CSS
- JavaScript
- NumPy
- Matplotlib
- Scikit-learn

Detection Classes

The model is designed to classify images into the following categories:

- Aphids
- Caterpillar
- Healthy
- Whitefly

Project Structure

Disease-detection/
│
├── main.py
├── index.html
├── requirements.txt
├── training.py
├── dataset/
│   ├── aphids/
│   ├── caterpillar/
│   ├── healthy/
│   └── whitefly/
└── README.md

Note: The training filename and dataset structure above should be adjusted to match the actual files in the repository.

How It Works

1. The user uploads a plant image through the web interface.
2. The image is processed and prepared for the AI model.
3. The trained deep learning model analyzes the image.
4. The model predicts the corresponding insect or healthy category.
5. The prediction is returned and displayed through the interface.

Model

The project uses a ResNet18-based image classification approach implemented with PyTorch. Images are resized to 224 × 224 pixels and normalized using ImageNet normalization values before being passed to the model.

Installation

git clone YOUR_REPOSITORY_URL
cd Disease-detection
pip install -r requirements.txt

Running the Application

Start the backend server using the appropriate command for the project:

uvicorn main:app --reload

Then open the frontend interface and connect it to the running API.

Purpose

The goal of PlantGuard is to explore how artificial intelligence and computer vision can be used to support plant monitoring and insect identification.

Limitations

Predictions depend on the dataset, image quality, and model performance. The system is an educational AI project and should not replace professional agricultural diagnosis.

Future Improvements

- Improve model accuracy with larger datasets
- Add more insect and plant disease categories
- Deploy the application online
- Add prediction confidence scores
- Improve mobile responsiveness
- Integrate real-time plant monitoring

---

PlantGuard – AI-powered plant insect detection.
