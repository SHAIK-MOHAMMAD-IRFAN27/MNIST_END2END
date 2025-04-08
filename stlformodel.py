import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the exported SavedModel
model = tf.keras.models.load_model("mnist_cnn_model.h5", compile=False)

st.title("🧠 Handwritten Digit Classifier (CNN + TensorFlow)")

uploaded_file = st.file_uploader("Upload a 28x28 grayscale image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(image, caption="Uploaded Image (resized to 28x28)", width=150)

    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

    if st.button("Predict"):
        prediction = model(img_array, training=False).numpy()
        predicted_label = int(np.argmax(prediction))
        st.success(f"Predicted Digit: **{predicted_label}**")
