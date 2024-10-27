import cv2
from mtcnn import MTCNN
from PIL import Image, ExifTags
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
import pickle


st.title('Prediction of Logistic regression')
# Load the model from the file
with open(r'trained_algo_on_umar1000.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Initialize the MTCNN face detector
detector = MTCNN()

def correct_image_orientation(image):
    try:
        # Get the image's EXIF data
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass

    return image

def detect_faces_from_image(image_):
    # img = Image.open(image_path)
    img = correct_image_orientation(img_)
    img.thumbnail((1200, 1200))  # Resize for faster detection
    img_rgb = np.array(img)
    faces = detector.detect_faces(img_rgb)
    return img_rgb, faces

def preprocess_image(image, target_size=(64, 64)):
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image_class(image):
    img_array = preprocess_image(image)
    prediction = loaded_model.predict(img_array)
    return (prediction > 0.5).astype("int32")[0][0]

# Upload images
# imag1 = st.file_uploader('Upload image 1', type=['jpg', 'jpeg', 'png'])
# imag1 = st.camera_input('capture image ')
imag2 = st.file_uploader('Upload image 2', type=['jpg', 'jpeg', 'png'])

# Process each image if uploaded
if imag2 is not None:
    img_rgb, faces = detect_faces_from_image(imag2)
else:
    # processed_image1 = None
    st.write("Please upload image.")

# Draw rectangles around detected faces and classify them
for face in faces:
    x, y, width, height = face['box']
    face_img = img_rgb[y:y + height, x:x + width]
    predicted_class = predict_image_class(face_img)
    # Draw rectangle and label if it's class A (0)
    if predicted_class == 0:  # Class A
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Green rectangle
        cv2.putText(img_rgb, 'Umar', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), (255, 0, 0), 2)  # Green rectangle
        cv2.putText(img_rgb, 'Not Umar', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.imshow(img_rgb)
plt.axis('off')
plt.show()

