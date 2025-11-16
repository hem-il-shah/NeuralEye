import streamlit as st
import cv2
import numpy as np
import torch

def run_object_detection():
    st.title("1. Object Detection ✨🔎")
    st.info("Upload an image to detect objects. 👀")

    uploaded_file = st.file_uploader("Choose an image... 😶‍🌫️", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the uploaded image file as a byte stream and decode it using cv2
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Load YOLOv5 model (only runs once thanks to Streamlit's caching)
        @st.cache_resource
        def load_model():
            return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        model = load_model()
        
        # Perform inference
        results = model(image)

        # Display the processed image with bounding boxes
        st.image(results.render()[0], caption='Processed Image 💯', use_container_width=True)
        
        # --- FIX APPLIED HERE ---
        # st.write(results.pandas().xyxy[0], caption='Results: Objects Detected 👾') <--- ERROR LINE
        
        # Display the detection results (DataFrame) using st.dataframe 
        # and st.subheader for a title/caption.
        st.subheader('Results: Objects Detected 👾')
        st.dataframe(results.pandas().xyxy[0]) # Use st.dataframe for Pandas DataFrames
        # ------------------------
