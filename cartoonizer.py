# import streamlit as st
# from PIL import Image
# import numpy as np
# import cv2

# # Function to sketch the image
# @st.cache_data
# def sketch_img(img):
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_gray = cv2.medianBlur(img_gray, 5)
#     edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
#     _, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
#     return thresholded

# # Function to cartoonize the image
# @st.cache_data
# def cartoonize_image(img, gray_mode=False):
#     thresholded = sketch_img(img)
#     filtered = cv2.bilateralFilter(img, 10, 250, 250)
#     cartoonized = cv2.bitwise_and(filtered, filtered, mask=thresholded)

#     if gray_mode:
#         return cv2.cvtColor(cartoonized, cv2.COLOR_BGR2GRAY)
    
#     return cartoonized

# # Function to apply OpenCV's pencil sketch and stylization effects
# @st.cache_data
# def apply_effects(img):
#     sketch_gray, sketch_color = cv2.pencilSketch(img, sigma_s=30, sigma_r=0.1, shade_factor=0.1)
#     stylized_image = cv2.stylization(img, sigma_s=60, sigma_r=0.07)
#     return sketch_gray, sketch_color, stylized_image

# # Main function for the Cartoonizer Page
# def run_cartoonizer():
#     st.title("4. Cartoonizer 🎨")
#     st.info("Convert Your Image into a Cartoon! 🤥")

#     img_file_buffer = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

#     if img_file_buffer is not None:
#         image = np.array(Image.open(img_file_buffer))
        
#         # Display original image
#         st.image(image, caption="🖼️ Original Image", use_container_width =True)

#         # Generate effects
#         sketch_img_result = sketch_img(image)
#         cartoonized_img = cartoonize_image(image)
#         cartoonized_img_gray = cartoonize_image(image, gray_mode=True)
#         sketch_gray, sketch_color, stylized_image = apply_effects(image)

#         # Display results
#         st.subheader("✏️ Sketch Image")
#         st.image(sketch_img_result, caption="Sketch Effect 🦋", use_container_width=True)

#         st.subheader("🎭 Cartoonized Image")
#         st.image(cartoonized_img, caption="Cartoonized Effect 👼🏻", use_container_width=True)

#         st.subheader("🖤 Cartoonized Image (Grayscale)")
#         st.image(cartoonized_img_gray, caption="Cartoon Gray Effect 👼🏼", use_container_width=True)

#         st.subheader("🖌️ Pencil Sketch (Color)")
#         st.image(sketch_color, caption="Pencil Sketch Color ✍🏻", use_container_width=True)

#         st.subheader("🖤 Pencil Sketch (Grayscale)")
#         st.image(sketch_gray, caption="Pencil Sketch Gray ✍️", use_container_width=True)

#         st.subheader("🎨 Stylized Image")
#         st.image(stylized_image, caption="Stylized Effect 🤖", use_container_width=True)
    
#     else:
#         st.warning("⚠ Please upload a picture!")

# -------------------------------------------------------------------------------------------------------------------------------------------------------------

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import zipfile

# Function to sketch the image
@st.cache_data
def sketch_img(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    _, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
    return thresholded

# Function to cartoonize the image
@st.cache_data
def cartoonize_image(img, gray_mode=False):
    thresholded = sketch_img(img)
    filtered = cv2.bilateralFilter(img, 10, 250, 250)
    cartoonized = cv2.bitwise_and(filtered, filtered, mask=thresholded)

    if gray_mode:
        return cv2.cvtColor(cartoonized, cv2.COLOR_BGR2GRAY)
    
    return cartoonized

# Function to apply OpenCV's pencil sketch and stylization effects
@st.cache_data
def apply_effects(img):
    sketch_gray, sketch_color = cv2.pencilSketch(img, sigma_s=30, sigma_r=0.1, shade_factor=0.1)
    stylized_image = cv2.stylization(img, sigma_s=60, sigma_r=0.07)
    return sketch_gray, sketch_color, stylized_image

# Function to save images as PDF
def save_images_as_pdf(images):
    pdf_bytes = io.BytesIO()
    images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=images[1:])
    return pdf_bytes.getvalue()

# Function to save images as ZIP
def save_images_as_zip(image_dict):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for name, img in image_dict.items():
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            zipf.writestr(f"{name}.jpg", img_byte_arr.getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Main function for the Cartoonizer Page
def run_cartoonizer():
    st.title("5. Cartoonizer 🎨")
    st.info("Convert Your Image into a Cartoon! 🤥")

    img_file_buffer = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

        # Convert images to PIL format for saving
        original_pil = Image.fromarray(image)

        # Generate effects
        sketch_img_result = sketch_img(image)
        cartoonized_img = cartoonize_image(image)
        cartoonized_img_gray = cartoonize_image(image, gray_mode=True)
        sketch_gray, sketch_color, stylized_image = apply_effects(image)

        # Convert processed images to PIL format
        sketch_pil = Image.fromarray(sketch_img_result)
        cartoon_pil = Image.fromarray(cartoonized_img)
        cartoon_gray_pil = Image.fromarray(cartoonized_img_gray)
        sketch_gray_pil = Image.fromarray(sketch_gray)
        sketch_color_pil = Image.fromarray(sketch_color)
        stylized_pil = Image.fromarray(stylized_image)

        # Display images in columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✏️ Sketch Image")
            st.image(sketch_img_result, caption="Sketch Effect 🦋", use_container_width=True)

            st.subheader("🖤 Cartoonized Image (Grayscale)")
            st.image(cartoonized_img_gray, caption="Cartoon Gray Effect 👼🏼", use_container_width=True)

            st.subheader("🖌️ Pencil Sketch (Color)")
            st.image(sketch_color, caption="Pencil Sketch Color ✍🏻", use_container_width=True)

        with col2:
            st.subheader("🎭 Cartoonized Image")
            st.image(cartoonized_img, caption="Cartoonized Effect 👼🏻", use_container_width=True)

            st.subheader("🖤 Pencil Sketch (Grayscale)")
            st.image(sketch_gray, caption="Pencil Sketch Gray ✍️", use_container_width=True)

            st.subheader("🎨 Stylized Image")
            st.image(stylized_image, caption="Stylized Effect 🤖", use_container_width=True)

        # Create dictionary of images for ZIP file
        images_dict = {
            "Original": original_pil,
            "Sketch": sketch_pil,
            "Cartoon": cartoon_pil,
            "Cartoon_Gray": cartoon_gray_pil,
            "Sketch_Gray": sketch_gray_pil,
            "Sketch_Color": sketch_color_pil,
            "Stylized": stylized_pil
        }

        # Generate PDF and ZIP files
        pdf_data = save_images_as_pdf(list(images_dict.values()))
        zip_data = save_images_as_zip(images_dict)

        # Download buttons
        st.subheader("📥 Download Processed Images")
        st.download_button(label="📄 Download as PDF", data=pdf_data, file_name="cartoonized_images.pdf", mime="application/pdf")
        st.download_button(label="📦 Download as ZIP (JPGs)", data=zip_data, file_name="cartoonized_images.zip", mime="application/zip")

    else:
        st.warning("⚠ Please upload a picture!")
