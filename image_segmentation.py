# import streamlit as st
# import numpy as np
# from PIL import Image
# import torch
# from segmentation_models_pytorch import Unet
# from torchvision import transforms

# # Load the U-Net model with pre-trained weights
# def load_unet_model():
#     model = Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')  # Binary segmentation
#     model.eval()  # Set the model to evaluation mode
#     return model

# # Preprocess the image to match model input
# def preprocess_image(image):
#     # Resize the image to match the model input size
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),  # Convert to tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
#     ])
#     return transform(image).unsqueeze(0)  # Add batch dimension

# # Run the image segmentation functionality
# def run_image_segmentation():
#     st.title("Image Segmentation")
#     st.write("Upload an image for segmentation.")

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert("RGB")  # Ensure the image is in RGB format
#         model = load_unet_model()  # Load the model
#         image_tensor = preprocess_image(image)  # Preprocess the image
        
#         # Make predictions
#         with torch.no_grad():  # Disable gradient calculation
#             prediction = model(image_tensor)
        
#         # Thresholding to create a binary mask
#         segmented_image = (prediction[0][0] > 0.5).cpu().numpy().astype(np.uint8) * 255  # Convert to binary mask
        
#         # Convert the segmented image back to PIL format for display
#         segmented_image_pil = Image.fromarray(segmented_image)

#         # Display the original and segmented images
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(image, caption='Original Image', use_container_width=True)
#         with col2:
#             st.image(segmented_image_pil, caption='Segmented Image', use_container_width=True)

# # Call the function to run the app
# if __name__ == "__main__":
#     run_image_segmentation()  # Run the image segmentation app

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------

import os
import cv2
import clip
import time
import torch
import warnings
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from segment_anything import build_sam, SamAutomaticMaskGenerator

warnings.filterwarnings("ignore")

MODEL_CHECKPOINT = "models_is/sam_vit_h_4b8939.pth"

@st.cache_resource()
def mask_generate(MODEL_CHECKPOINT):
    model_start_time = time.time()
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=MODEL_CHECKPOINT))
    model_end_time = time.time()
    print(f"Model loaded in {model_end_time - model_start_time} seconds.")
    return mask_generator

def generate_image_masks(image, mask_generator):
    image = np.array(image)
    mask_generator.predictor.set_image(image)
    return mask_generator.generate(image)

def convert_box_xywh_to_xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

def load_CLIP():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def retriev(elements: list[Image.Image], search_text: str) -> int:
    model, preprocess, device = load_CLIP()
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    image_features = model.encode_image(torch.stack(preprocessed_images))
    text_features = model.encode_text(tokenized_text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return (100.0 * image_features @ text_features.T)[:, 0].softmax(dim=0)

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]

def run_image_segmentation():
    st.title("2. Image Segmentation 🔮🫵")
    st.info("Perform open-vocabulary image segmentation. 🦹🏻")
    
    mask_generator = mask_generate(MODEL_CHECKPOINT)
    col_a, col_b = st.columns(2)
    
    prompt = st.text_input("Enter your text 🧐:", "")
    image_file = st.file_uploader("Upload Image 🚀", type=["png", "jpg", "bmp", "jpeg"])
    
    if image_file is not None and prompt.strip():
        with st.spinner("Processing... 💫"):
            image = Image.open(image_file).convert("RGB")
            masks = generate_image_masks(image, mask_generator)
            
            cropped_boxes = [
                segment_image(image, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"]))
                for mask in masks
            ]
            
            scores = retriev(cropped_boxes, str(prompt))
            indices = get_indices_of_values_above_threshold(scores, 0.05)
            
            segmentation_masks = [
                Image.fromarray(masks[i]["segmentation"].astype("uint8") * 255)
                for i in indices
            ]
            
            overlay_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_image)
            
            for segmentation_mask_image in segmentation_masks:
                draw.bitmap((0, 0), segmentation_mask_image, fill=(255, 0, 0, 200))
            
            result_image = Image.alpha_composite(image.convert("RGBA"), overlay_image)
            
            col1, col2 = st.columns(2)
            # with col1:
            #     st.image(image, width=500, caption="Original Image")
            # with col2:
            #     st.image(result_image, width=500, caption="Segmented Output")
            st.image(image, width=500, caption="Original Image 👽")
            st.image(result_image, width=500, caption="Segmented Output 🤡")

    else:
        st.warning("⚠ Please upload an image and provide a search prompt! 😡🤬")
