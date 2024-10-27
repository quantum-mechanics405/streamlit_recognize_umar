import cv2
from mtcnn import MTCNN
from PIL import Image, ExifTags
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle


st.title('Recognizing Umar from Image')
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
    img = correct_image_orientation(image_)
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

# Upload image
imag2 = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])
# imag2 = st.camera_input('capture image ')
# Process image if uploaded
if imag2 is not None:
    # Convert uploaded file to a PIL image
    image = Image.open(imag2)
    img_rgb, faces = detect_faces_from_image(image)
    
    # Draw rectangles around detected faces and classify them
    for face in faces:
        x, y, width, height = face['box']
        face_img = img_rgb[y:y + height, x:x + width]
        predicted_class = predict_image_class(face_img)
        
        # Draw rectangle and label
        color, label = ((0, 255, 0), 'Umar') if predicted_class == 0 else ((255, 0, 0), 'Not Umar')
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), color, 2)
        cv2.putText(img_rgb, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the result
    st.image(img_rgb, caption="Detected Faces", use_column_width=True)
else:
    st.write("Please upload an image.")
