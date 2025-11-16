import matplotlib.pylab as plt
from API import transfer_style
import streamlit as st
from PIL import Image
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import io

def run_neural_style_transfer():
    st.title("3. Neural Style Transfer ✨😎")
    st.info("Upload Content and Style Images to do some magic. 🗣")


    # Path of the pre-trained model
    model_path = "model_nst"

    # Upload images
    content_file = st.file_uploader("Upload Content Image: 🌐", type=["jpg", "png", "jpeg"])
    style_file = st.file_uploader("Upload Style Image: 🌀", type=["jpg", "png", "jpeg"])

    if content_file and style_file:
        # Open images
        content_image = Image.open(content_file).convert("RGB")
        style_image = Image.open(style_file).convert("RGB")

        # Perform style transfer
        img = transfer_style(content_image, style_image, model_path)

        # Convert output to Image format for Streamlit
        output_image = Image.fromarray((img * 255).astype('uint8'))  # Normalize & convert

        # Display images
        st.subheader("Content Image: 🌐")
        st.image(content_image, caption="Uploaded Content Image 👻", use_container_width=True)

        st.subheader("Style Image: 🌀")
        st.image(style_image, caption="Uploaded Style Image 🦖", use_container_width=True)

        st.subheader("Stylized Output: ☯️")
        st.image(output_image, caption="Generated Stylized Image 🌈", use_container_width=True)

        # Provide download option
        img_io = io.BytesIO()
        output_image.save(img_io, format="JPEG")
        img_io.seek(0)
        st.download_button("🔎 Download Stylized Image 🔍", img_io, file_name="stylized_image.jpg", mime="image/jpeg")


