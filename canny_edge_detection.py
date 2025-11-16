import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

@st.cache_data
def edge_detection(image, low_thres, high_thresh):
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny Edge Detection
    edged = cv2.Canny(image, low_thres, high_thresh)
    
    return edged

# Function to save as PDF
def save_as_pdf(image):
    img_pil = Image.fromarray(image)
    pdf_bytes = io.BytesIO()
    img_pil.save(pdf_bytes, format="PDF")
    return pdf_bytes.getvalue()

# Function to save as JPG
def save_as_jpg(image):
    img_pil = Image.fromarray(image)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="JPEG")
    return img_bytes.getvalue()

def run_canny_edge_detection():
    st.title("6. Canny Edge Detection 📄")
    st.info("Detect your perfect EDGE...! ")

    img_file_buffer = st.file_uploader("📤 Upload an image (clear image recommended)", type=["jpg", "jpeg", "png"])

    if img_file_buffer is None:
        st.warning("⚠ Please upload a picture!")
        return  # Stop execution if no image is uploaded

    # Read and process image
    image = np.array(Image.open(img_file_buffer))

    st.subheader("🖼️ Original Image")
    st.image(image, caption="Original Image 💘", use_container_width=True)

    # Threshold Sliders
    low_thres = st.slider("🔽 Lower threshold for edge detection", min_value=0, max_value=240, value=80)
    high_thresh = st.slider("🔼 High threshold for edge detection", min_value=10, max_value=240, value=100)

    # Adjust high_thresh if lower threshold is greater
    if low_thres > high_thresh:
        high_thresh = low_thres + 5

    # Process Image
    edges = edge_detection(image, low_thres, high_thresh)

    st.subheader("📏 Edged Image")
    st.image(edges, caption="Processed Edge Image 🔅", use_container_width =True)

    # Generate Download Options
    pdf_data = save_as_pdf(edges)
    jpg_data = save_as_jpg(edges)

    st.subheader("📥 Download Options")
    st.download_button("📄 Download as PDF", data=pdf_data, file_name="edged_image.pdf", mime="application/pdf")
    st.download_button("🖼️ Download as JPG", data=jpg_data, file_name="edged_image.jpg", mime="image/jpeg")
