import streamlit as st

def run_home():
    # Title and subtitle
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>👁️‍🗨️ Welcome to NeuralEye!</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #4B8BFF;'>Unlock the Power of AI-driven Computer Vision 🚀</h3>", unsafe_allow_html=True)
    
    # Banner
    # st.image("assets/banner.jpg", use_column_width=True)

    # Introduction
    st.write(
        """
        NeuralEye is your one-stop solution for cutting-edge **Computer Vision** applications.  
        From detecting objects to transforming images into stunning artistic renditions, we've got you covered!  
        Explore the various functionalities below. ⬇️  
        """
    )

    # Features Section
    st.markdown("---")  # Horizontal line for separation
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Object Detection 🔍")
        st.write("Detect multiple objects in images with high accuracy using Faster R-CNN. 📸")
        
        st.subheader("2. Image Segmentation 🎨")
        st.write("Segment images into meaningful regions using DeepLabV3. Perfect for scene understanding! 🏞️")
        
        st.subheader("3. Neural Style Transfer (NST) 🖌️")
        st.write("Transform your images into breathtaking artworks using VGG19-based NST. 🎭")

        st.subheader("4. Quick Detection 🚨")
        st.write("Fast object detection with real-time insights. Ideal for quick visual analysis! ⚡")

    with col2:
        st.subheader("5. Cartoonizer 🧿")
        st.write("Convert your photos into fun, animated cartoon-style images. 🖼️")
        
        st.subheader("6. Canny Edge Detection 📄")
        st.write("Detect sharp edges and outlines in images using the Canny edge detection algorithm. ✂️")
        
        st.subheader("7. Human Pose Estimation 🏃‍♂️")
        st.write("Analyze human movements and poses with advanced deep-learning models. 🏋️")

    # Closing Section
    st.markdown("---")
    st.markdown(
        "<h3 style='text-align: center;'>Start Exploring Now! 🌟</h3>", 
        unsafe_allow_html=True
    )
    st.write("Use the sidebar to navigate through different features and experience the power of AI in vision.")

    # Footer
    st.markdown("<small>Made with ❤️ using Streamlit</small>", unsafe_allow_html=True)
