# 🍀 Leaf Disease Detection using Deep Learning

A Machine Learning project to classify plant leaf diseases using a pretrained **DenseNet121** model. This project demonstrates end-to-end ML skills including model integration, frontend interface, and development-readiness.

---

## 🔍 Problem Statement

Plant diseases can significantly reduce crop yields and Food Security. Manual Identification is slow and error-prone. This project aims to build an automated Leaf disease Classifier using Computer Vision techniques.

---

## 🎯 Features

- 🔍 Predict plant disease from uploaded Leaf image.
- 🧠 Deep Learning Model (DenseNet121) used for classification.
- 📂 Pretrained Model integrated, No training required.
- 💻 Simple Web Interface using **Flask**
- 📦 Clean Project structure with 'requirements.txt'
- 📝 Dataset Info and retraining instructions included.

---

## 🚀 Technologies Used 

1. Python - Core Programming Language
2. Flask - Web Framework (for UI)
3. Tensorflow/Keras - Model Integration and Prediction
4. Densenet121 - Pretrained Model for CV tasks
5. HTML/CSS - For building the frontend
6. Git - Version control

---
## 📂 Project Structure

```
├── Leaf-Disease Detection
├── model.py            # Flask app for serving predictions
├── fix_model.py        # Script to fix model loading issues
├── model.h5            # Trained Keras model file(downloadable from drive link)
├── old_model.h5        # Model used to solve issues and reload in new code
├── requirements.txt    # Python dependencies
├── train.ipynb         # Jupyter notebook for training the model
├── static/             # Static files (CSS, JS, images)
├── templates/          # HTML templates for Flask app
└── uploads/            # Uploaded images for prediction
```

---

## 🎨 UI Preview

![UI Screenshot]("static/images/image1.png")

---

## 🧠 How it Works

1. User uploads a leaf image via browser.
2. Image is preprocessed using keras utilities.
3. DenseNet121 model makes a predicion.
4. Result is shown on the webpage

---

## 📑 Dataset Information

This project used publicly available PlantVillage dataset : 🔗 🔗 [Kaggle - PlantVillage Dataset] (https://www.kaggle.com/datasets/emmarex/plantdisease)

📌 The model is already trained. You don't need dataset unless you wish to retrain.

---

## 🏃‍♂️ How to run locally

```
# clone the repo
git clone https://github.com/srikanthsaladi1079/leaf-disease-detection.git

# Navigate to folder
cd leaf-disease-detection

# Create a virtual environment (optional)
python -m venv venv 
source venv/bin/activate # or venv\Scripts\acivate on Windows

# Model File Setup (Manual)
To keep repository size small, the trained model (model.h5) is not included in the Github repo. You can download it through the following link:
(Link here)

# How to Use
1. Click the link abive to download the file.
2. Place the downloaded model.h5 file in the root directory of this project (same level as your .ipynb or model.py)
3. Run the notebook or script as usual.

⚙️ Optional : Auto download with Python
You can automatically download the file using the script below:

1. Install gdown using command :
 pip install gdown

2. Run this script
```
import gdown
file_id = "FILE_ID" # replace this with actual file id from given link
output = "model.h5"
gdown.download(f"Link here id={file_id}", output, quiet=False)

⚠️ There shall be model.h5 and old_model.h5 , old_model.h5 is trained early and model.h5 is improvised version. Download model.h5 for use. If the file isn't placed correcly, you may see errors like "FileNotFoundError: model.h5 not found

# Install dependencies
pip install -r requirements.txt

# Run the app
python model.py

Then open your browser and go to http://localhost:5000

---

## 🧑‍💻 Author and Credits

This project was created by Srikanth Saladi(me), a python enthusiast passionate about solving real-world problems using Machine Learning and Backend Development.

You can connect with me on 

Github : https://github.com/srikanthsaladi1079

---

📌 This Project is for learning and showcasing ML integration with Flask. The core ML model is adapted from open-source sources and customised with python 3.9+ with updated libraries and deployment support.

---

## 🤝 Contributions

Feel free to fork and improve the project. Pull requests and issues are welcome!

---

## 📜 License

This project is licensed under the MIT License.

---

Thank you for checking out my project! 🌱
