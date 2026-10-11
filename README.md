# PhytoScan – AI-Based Potato Plant Disease Detection

## About the Project
PhytoScan is a web-based application that uses a Convolutional Neural Network (CNN) to detect diseases in potato plants from leaf images. It classifies uploaded images and provides disease management recommendations.

## Objectives
- Detect potato leaf diseases using deep learning.
- Classify potato leaves into three categories.
- Provide useful disease management recommendations through a simple web interface.

## Disease Categories
1. **Healthy** – Potato leaf without the targeted diseases.
2. **Early Blight** – Potato leaf affected by early blight.
3. **Late Blight** – Potato leaf affected by late blight.

## Technologies Used
- Python
- TensorFlow / Keras
- Convolutional Neural Network (CNN)
- Flask
- HTML and CSS
- NumPy

## Dataset
The project uses the PlantVillage dataset for training the CNN model.

## Main Features
- Upload a potato leaf image.
- Predict the disease category using the trained model.
- Display the prediction and confidence score.
- Provide disease management recommendations.

## Project Structure
- `app.py` – Flask web application.
- `train.py` – Model training script.
- `predict.py` – Prediction functionality.
- `evaluate.py` – Model evaluation script.
- `phytoscan_best.keras` – Saved trained model.
- `requirements.txt` – Python dependencies.
- `templates/index.html` – Web interface.

## How to Run
1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```
5. Open the local URL displayed in the terminal in your browser.

## Disclaimer
PhytoScan provides preliminary image-based predictions and management guidance. It should not replace professional agricultural advice.
