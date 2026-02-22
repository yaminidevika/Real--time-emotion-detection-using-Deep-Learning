import gradio as gr
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from PIL import Image

# Define emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def create_model():
    """
    Recreate the model architecture from the training notebook
    to avoid configuration loading errors with newer Keras versions.
    """
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(7, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(7, kernel_size=(4, 4), activation='relu'))

    model.add(Flatten())
    model.add(Activation("softmax"))
    
    return model

# Load model weights
model = None
try:
    model = create_model()
    model.load_weights('emotion_detector_models/model_v6_23.hdf5')
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    model = None

# Load Face Cascade
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Face cascade loaded.")
except Exception as e:
    print(f"Error loading face cascade: {e}")
    face_cascade = None

def preprocess_face(face_image):
    """
    Preprocess face image for the model.
    Grayscale, Resize to 48x48, Normalize
    """
    # Resize to 48x48
    face_resized = cv2.resize(face_image, (48, 48))
    
    # Normalize 
    face_normalized = face_resized.astype('float32') / 255.0
    
    # Reshape for model input: (1, 48, 48, 1)
    face_input = np.expand_dims(face_normalized, axis=0) # add batch dim
    face_input = np.expand_dims(face_input, axis=-1) # add channel dim
    
    return face_input

def predict_emotion(image):
    if model is None or face_cascade is None:
        return image, "Error: Model or Face Cascade not loaded."

    # Convert to grayscale for face detection
    if isinstance(image, np.ndarray):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        # If PIL image
        image = np.array(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return image, "No faces detected."

    # Process each face
    for (x, y, w, h) in faces:
        # Extract face ROI (Region of Interest)
        face_roi = gray_image[y:y+h, x:x+w]
        
        try:
            # Preprocess
            face_input = preprocess_face(face_roi)
            
            # Predict
            predictions = model.predict(face_input)
            emotion_index = np.argmax(predictions)
            emotion_label = EMOTION_LABELS[emotion_index]
            confidence = predictions[0][emotion_index]
            
            label_text = f"{emotion_label} ({confidence:.2f})"
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Draw label
            cv2.putText(image, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
        except Exception as e:
            print(f"Error processing face: {e}")

    return image, f"Found {len(faces)} face(s)."

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[gr.Image(type="numpy", label="Processed Image"), "text"],
    title="Face and Emotion Detection",
    description="Upload an image to detect faces and emotions using OpenCV and Keras."
)

if __name__ == "__main__":
    iface.launch()
