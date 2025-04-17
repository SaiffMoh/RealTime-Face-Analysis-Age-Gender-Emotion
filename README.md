# Real-Time Emotion, Age, and Gender Detection

A deep learning-based computer vision project that detects **emotion**, **gender**, and **age** from faces in real time using webcam input.

## 🚀 Features
- Detects and classifies 7 emotions: `Anger`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`
- Predicts **gender** with confidence
- Estimates **age** as a regression output
- Displays live predictions via webcam feed with bounding boxes and labels

## 📂 Project Structure
- `main.py`: Real-time inference script using OpenCV and TensorFlow/Keras
- `emotiongenderagedet.ipynb`: Notebook used for training and saving the models
- `models/`: Directory where trained models should be placed

## 🧠 Models
- **Emotion Model**: CNN trained on FER2013
- **Age & Gender Models**: MobileNetV2-based models trained on UTKFace dataset
- Models are saved in `.h5` format and loaded during inference

👉 **Download the trained models from this Google Drive link**:  
📁 [Download models](https://drive.google.com/drive/folders/19KzbvM_0BJsuZu3LpLEWWkJQ0jflMnzh?usp=sharing)  
Then place the downloaded files inside a `models/` folder in the project root.

## 📦 Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
