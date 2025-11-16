# import streamlit as st
# import cv2
# import torch
# import numpy as np
# import time
# import tempfile
# from pathlib import Path

# # Import detection utilities
# from detection_utils import load_model, detect_objects, draw_boxes, ObjectTracker

# def initialize_video_capture(input_source, video_file=None, url=None):
#     """Initialize video capture and writer"""
#     cap = None
#     out = None
#     output_path = None
    
#     if input_source == "Video File 🎥" and video_file is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
#         tfile.write(video_file.read())
#         tfile.flush()
#         video_path = tfile.name
#         cap = cv2.VideoCapture(video_path)
        
#         if cap.isOpened():
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
#             temp_dir = tempfile.gettempdir()
#             output_path = str(Path(temp_dir) / 'detected_output.mp4')
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
#     elif input_source == "Live Stream URL 📽" and url:
#         cap = cv2.VideoCapture(url)
    
#     return cap, out, output_path

# def get_model_info():
#     """Return YOLO model details"""
#     return {
#         'yolov8n.pt': {'name': 'YOLOv8 Nano', 'speed': '⚡⚡⚡⚡⚡', 'accuracy': '⭐⭐'},
#         'yolov8s.pt': {'name': 'YOLOv8 Small', 'speed': '⚡⚡⚡⚡', 'accuracy': '⭐⭐⭐'},
#         'yolov8m.pt': {'name': 'YOLOv8 Medium', 'speed': '⚡⚡⚡', 'accuracy': '⭐⭐⭐⭐'},
#         'yolov8l.pt': {'name': 'YOLOv8 Large', 'speed': '⚡⚡', 'accuracy': '⭐⭐⭐⭐⭐'},
#         'yolov8x.pt': {'name': 'YOLOv8 XLarge', 'speed': '⚡', 'accuracy': '⭐⭐⭐⭐⭐⭐'}
#     }

# def run_quick_detection():
#     """Runs the accident detection page"""
#     st.title("4. Quick Detection 🚨")
#     st.info("Analyze your Videos here...! 🤠")
    
#     model_info = get_model_info()
#     selected_model = st.selectbox("Choose Model", list(model_info.keys()), format_func=lambda x: model_info[x]['name'])
    
#     with st.expander("Model Details: 📋"):
#         st.write(model_info[selected_model]['name'])
#         st.write(f"Speed: {model_info[selected_model]['speed']}")
#         st.write(f"Accuracy: {model_info[selected_model]['accuracy']}")
    
#     if st.button("Load Model 🎓"):
#         st.session_state.model = load_model(selected_model)
#         st.success("Model Loaded! 😁")
    
#     confidence = st.slider("Confidence: 😬", 0.0, 1.0, 0.5)
#     input_source = st.radio("Select Source: 🤐", ["Video File 🎥", "Live Stream URL 📽"])
    
#     video_file, url = None, None
#     if input_source == "Video File 🎥":
#         video_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
#     elif input_source == "Live Stream URL 📽":
#         url = st.text_input("Enter Stream URL 🎞") 
    
#     if st.button("Start Detection! 💊"):
#         st.session_state.cap, st.session_state.out, st.session_state.output_path = initialize_video_capture(input_source, video_file, url)
        
#         if st.session_state.cap and st.session_state.cap.isOpened():
#             video_placeholder = st.empty()
            
#             while True:
#                 ret, frame = st.session_state.cap.read()
#                 if not ret:
#                     break
#                 detections = detect_objects(st.session_state.model, frame, confidence)
#                 annotated_frame = draw_boxes(frame, detections, ObjectTracker())
#                 video_placeholder.image(annotated_frame, channels="BGR")
#                 time.sleep(0.001)
            
#             st.session_state.cap.release()
#             if st.session_state.out:
#                 st.session_state.out.release()
            
#             st.success("Detection Complete! 🎉")
    
#             if "output_path" in st.session_state and st.session_state.output_path:
#                 video_path = st.session_state.output_path
#                 if Path(video_path).exists():
#                     with open(video_path, 'rb') as f:
#                         video_data = f.read()
#                     st.download_button(
#                         "📥 Download Processed Video",
#                         data=video_data,
#                         file_name="detected_output.mp4",
#                         mime="video/mp4"
#             )
#         else:
#             st.warning("⚠️ Processed video file not found. Please run detection first.")

import streamlit as st
import cv2
import torch
import numpy as np
import time
import tempfile
from pathlib import Path

# Import detection utilities
from detection_utils import load_model, detect_objects, draw_boxes, ObjectTracker

def initialize_video_capture(input_source, video_file=None, url=None):
    """Initialize video capture and writer"""
    cap = None
    out = None
    output_path = None
    
    if input_source == "Video File 🎥" and video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.flush()
        video_path = tfile.name
        cap = cv2.VideoCapture(video_path)
        
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            temp_dir = tempfile.gettempdir()
            output_path = str(Path(temp_dir) / 'detected_output.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
    elif input_source == "Live Stream URL 📽" and url:
        cap = cv2.VideoCapture(url)
    
    return cap, out, output_path

def get_model_info():
    """Return YOLO model details"""
    return {
        'yolov8n.pt': {'name': 'YOLOv8 Nano', 'speed': '⚡⚡⚡⚡⚡', 'accuracy': '⭐⭐'},
        'yolov8s.pt': {'name': 'YOLOv8 Small', 'speed': '⚡⚡⚡⚡', 'accuracy': '⭐⭐⭐'},
        'yolov8m.pt': {'name': 'YOLOv8 Medium', 'speed': '⚡⚡⚡', 'accuracy': '⭐⭐⭐⭐'},
        'yolov8l.pt': {'name': 'YOLOv8 Large', 'speed': '⚡⚡', 'accuracy': '⭐⭐⭐⭐⭐'},
        'yolov8x.pt': {'name': 'YOLOv8 XLarge', 'speed': '⚡', 'accuracy': '⭐⭐⭐⭐⭐⭐'}
    }

def run_quick_detection():
    """Runs the accident detection page"""
    st.title("4. Quick Detection 🚨")
    st.info("Analyze your Videos here...! 🤠")
    
    model_info = get_model_info()
    selected_model = st.selectbox("Choose Model", list(model_info.keys()), format_func=lambda x: model_info[x]['name'])
    
    if not selected_model:
        st.warning("⚠️ Please Select a Model!")
    
    with st.expander("Model Details: 📋"):
        st.write(model_info[selected_model]['name'])
        st.write(f"Speed: {model_info[selected_model]['speed']}")
        st.write(f"Accuracy: {model_info[selected_model]['accuracy']}")
    
    if st.button("Load Model 🎓"):
        st.session_state.model = load_model(selected_model)
        st.success("Model Loaded! 😁")
    
    confidence = st.slider("Confidence: 😬", 0.0, 1.0, 0.5)
    input_source = st.radio("Select Source: 🤐", ["Video File 🎥", "Live Stream URL 📽"])
    
    video_file, url = None, None
    if input_source == "Video File 🎥":
        video_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        if not video_file:
            st.warning("⚠️ Please Upload a Video File!")
    elif input_source == "Live Stream URL 📽":
        url = st.text_input("Enter Stream URL 🎞")
        if not url:
            st.warning("⚠️ Please Enter a Stream URL!")
    
    if st.button("Start Detection! 💊"):
        if not selected_model:
            st.warning("⚠️ Please Select a Model before starting detection!")
            return
        if not video_file and not url:
            st.warning("⚠️ Please Upload a Video or Enter a Stream URL before starting detection!")
            return
        
        st.session_state.cap, st.session_state.out, st.session_state.output_path = initialize_video_capture(input_source, video_file, url)
        
        if st.session_state.cap and st.session_state.cap.isOpened():
            video_placeholder = st.empty()
            
            while True:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    break
                detections = detect_objects(st.session_state.model, frame, confidence)
                annotated_frame = draw_boxes(frame, detections, ObjectTracker())
                video_placeholder.image(annotated_frame, channels="BGR")
                if st.session_state.out:
                    st.session_state.out.write(annotated_frame)
                time.sleep(0.001)
            
            st.session_state.cap.release()
            if st.session_state.out:
                st.session_state.out.release()
                st.session_state.out = None
            
            st.success("Detection Complete! 🎉")
    
    if "output_path" in st.session_state and st.session_state.output_path:
        video_path = st.session_state.output_path
        if Path(video_path).exists():
            with open(video_path, 'rb') as f:
                video_data = f.read()
            st.download_button(
                "📥 Download Processed Video",
                data=video_data,
                file_name="detected_output.mp4",
                mime="video/mp4"
            )
        else:
            st.warning("⚠️ Processed video file not found. Please run detection first.")
