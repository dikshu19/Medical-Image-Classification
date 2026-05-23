import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

model = load_model("models/models/brain_tumor_model.h5")

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("Brain Tumor MRI Classification")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded MRI Image")

    img = img.resize((224, 224))

    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    pred = np.argmax(prediction)

    st.success(f"Predicted Class: {classes[pred]}")