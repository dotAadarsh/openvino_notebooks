import os
import cv2
import streamlit as st
import numpy as np
import openvino as ov
from pathlib import Path
import tempfile

# Initialize OpenVINO and Streamlit
core = ov.Core()
st.title("Image Colorization using OpenVINO")

# Device selection
device = st.selectbox("Select Device:", core.available_devices + ["AUTO"])
st.write(f"Selected Device: {device}")

PRECISION = "FP16"
MODEL_DIR = "models"
MODEL_NAME = "colorization-v2"
# MODEL_NAME="colorization-siggraph"
MODEL_PATH = f"{MODEL_DIR}/public/{MODEL_NAME}/{PRECISION}/{MODEL_NAME}.xml"
DATA_DIR = "data"

# Create a temporary directory to store the uploaded image
temp_dir = tempfile.mkdtemp()


# Add a file uploader to your Streamlit app
uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png"])

if uploaded_file is not None: 

    col1, col2 = st.columns(2)
    with col1:    

        # If the user has uploaded an image, save it to the temporary directory
        if uploaded_file is not None:
            with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())

            # Read the uploaded image into memory
            image = cv2.imread(os.path.join(temp_dir, uploaded_file.name))

            # Do something with the image here, such as displaying it or performing image processing
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        model = core.read_model(model=MODEL_PATH)
        compiled_model = core.compile_model(model=model, device_name=device)
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        N, C, H, W = list(input_layer.shape)

        def colorize_image(input_image):
            h_in, w_in, _ = input_image.shape
            img_rgb = input_image.astype(np.float32) / 255
            img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
            img_l_rs = cv2.resize(img_lab.copy(), (W, H))[:, :, 0]
            inputs = np.expand_dims(img_l_rs, axis=[0, 1])
            res = compiled_model([inputs])[output_layer]
            update_res = np.squeeze(res)
            out = update_res.transpose((1, 2, 0))
            out = cv2.resize(out, (w_in, h_in))
            img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
            img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2RGB), 0, 1)
            colorized_image = (cv2.resize(img_bgr_out, (w_in, h_in)) * 255).astype(np.uint8)
            return colorized_image

        colorized_image = colorize_image(image)
        st.image(colorized_image, caption="Colorized Image", use_column_width=True)
