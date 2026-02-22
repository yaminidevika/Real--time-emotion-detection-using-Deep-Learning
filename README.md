# Face and Emotion Detection

A Final Year Project for **face detection, face recognition, and emotion detection** using **Python**, **OpenCV**, **Keras**, and **Gradio**.

## ✨ Features

- Face detection from uploaded images
- Face recognition comparison between two faces
- Emotion classification (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
- Interactive web app using Gradio
- Jupyter notebooks for experimentation and model training workflow

## 📁 Project Structure

```text
face-and-emotion-detection/
├── app_gradio.py
├── debug_app.py
├── debug_log.txt
├── README.md
├── requirements.txt
├── emotion_detector_models/
│   └── model_v6_23.hdf5
├── src/
│   ├── EmotionDetector_v2.ipynb
│   └── facial_detection_recog_emotion.ipynb
└── test_images/
    ├── 040wrmpyTF5l.jpg
    ├── 20180901150822_new.jpg
    ├── 39.jpg
    ├── index1.jpg
    ├── index2.jpeg
    └── rajeev.jpg
```

## 🧰 Tech Stack

- Python 3.9+ (recommended: 3.10)
- OpenCV
- NumPy
- Keras / TensorFlow
- face_recognition
- Gradio
- Jupyter Notebook

## 🚀 How to Download and Use

### 1) Download the project

**Option A: Download ZIP (No git needed)**
1. Open the GitHub repository page.
2. Click **Code** → **Download ZIP**.
3. Extract ZIP.
4. Open extracted folder in VS Code.

**Option B: Clone with git**
```bash
git clone https://github.com/<your-username>/face-and-emotion-detection.git
cd face-and-emotion-detection
```

### 2) Create virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Run the Gradio app

```bash
python app_gradio.py
```

Then open the local URL shown in terminal (usually `http://127.0.0.1:7860`).

### 5) Use the app

1. Upload an image in the Gradio interface.
2. The app detects face(s).
3. It predicts emotion for each detected face.
4. It returns the processed image with bounding boxes + labels and text output with face count/status.

## 📓 Notebooks

- `src/facial_detection_recog_emotion.ipynb`  
  End-to-end demo for face detection, recognition, and emotion prediction.

- `src/EmotionDetector_v2.ipynb`  
  Model training and evaluation workflow for emotion detector.

To open notebooks:
```bash
jupyter notebook
```

## 🧠 Model Details

- Model weights file: `emotion_detector_models/model_v6_23.hdf5`
- Input preprocessing: grayscale, resize to `48x48`, normalize pixel values to `[0,1]`
- Output: 7 emotion classes

## ⚠️ Important Notes

- Keep the model path exactly as: `emotion_detector_models/model_v6_23.hdf5`
- If no faces are detected, app returns `"No faces detected."`
- For best results, use clear frontal-face images.

## 🛠️ Troubleshooting

**PowerShell blocks venv activation**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**face_recognition install issues on Windows**
If installation fails, install Visual C++ Build Tools and CMake (if required by your setup).

**Model load error**
Check that the file exists at `emotion_detector_models/model_v6_23.hdf5` and dependency versions in `requirements.txt` are installed.

## 👩‍💻 Author

`<Your Name>`  
Final Year Project

## 📄 License

This project is licensed under the MIT License.