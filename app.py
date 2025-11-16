import streamlit as st
from multipage import MultiPage

# Feature Pages
from object_detection import run_object_detection
from image_segmentation import run_image_segmentation
from neural_style_transfer import run_neural_style_transfer
from cartoonizer import run_cartoonizer
from canny_edge_detection import run_canny_edge_detection
from pose_estimation import run_pose_estimation
from quick_detection import run_quick_detection
from plant_disease_detection import run_plant_disease_detection  # ✅ NEW PAGE IMPORT

# Home Page
from home import run_home

# Set the page config with the app logo as favicon
st.set_page_config(page_title="NeuralEye", page_icon="assets/logo.jpg", layout="wide")

# Initialize MultiPage
app = MultiPage()

# Add Home Page
app.add_page("🏠 Home", run_home)

# Add all feature pages
app.add_page("1.) Object Detection 🫂", run_object_detection)
app.add_page("2.) Image Segmentation 🪇", run_image_segmentation)
app.add_page("3.) Neural Style Transfer (NST) 💅🏼", run_neural_style_transfer)
app.add_page("4.) Quick Detection 🚨", run_quick_detection)
app.add_page("5.) Cartoonizer 🧿", run_cartoonizer)
app.add_page("6.) Canny Edge Detection 📄", run_canny_edge_detection)
app.add_page("7.) Human Pose Estimation 🏃‍♂️", run_pose_estimation)
app.add_page("9.) Plant Disease Detection 🌾", run_plant_disease_detection)

# Run the app
app.run()

st.markdown("---")
