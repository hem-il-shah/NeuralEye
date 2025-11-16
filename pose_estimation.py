import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained pose detection model
POSE_MODEL = "graph_opt.pb"

# Define body parts and pose pairs
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Load OpenPose model
net = cv2.dnn.readNetFromTensorflow(POSE_MODEL)

def poseDetector(image, threshold):
    """ Detects human pose in an image using OpenPose """
    frameWidth, frameHeight = image.shape[1], image.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(image, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Extract only relevant parts

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, confidence, _, point = cv2.minMaxLoc(heatMap)
        x = int((frameWidth * point[0]) / out.shape[3])
        y = int((frameHeight * point[1]) / out.shape[2])
        points.append((x, y) if confidence > threshold else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(image, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.circle(image, points[idFrom], 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, points[idTo], 5, (0, 0, 255), cv2.FILLED)

    return image

def run_pose_estimation():
    """ Streamlit UI for Pose Estimation """
    st.title("7. Human Pose Estimation 🏃‍♂️")

    uploaded_file = st.file_uploader("📤 Upload an image for pose detection", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.subheader("🖼 Original Image")
        st.image(image, use_column_width=True)

        threshold = st.slider("🎯 Set confidence threshold", min_value=0, max_value=100, value=20, step=5) / 100
        
        output = poseDetector(image, threshold)
        st.subheader("📌 Pose Estimated Image")
        st.image(output, use_container_width=True)
    else:
        st.warning("⚠ Please upload an image to proceed!")

