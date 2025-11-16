
## 👁️ NeuralEye: Comprehensive Computer Vision Streamlit App

**NeuralEye** is a versatile web application built with **Streamlit** and various deep learning models, offering a comprehensive suite of computer vision functionalities in one easy-to-use interface.

---

### ✨ Features Overview

This application combines several advanced computer vision tasks:

* **Object Detection:** Identify and locate multiple objects within an image using **YOLO variants**.
* **Image Segmentation (SAM):** Highly accurate, promptable segmentation of objects in images using the **Segment Anything Model (SAM)**.
* **Plant Disease Detection:** Specialized feature for identifying common agricultural plant diseases.
* **Pose Estimation:** Detect and track human poses and key points.
* **Style Transfer:** Apply the artistic style of one image to the content of another.
* **Cartoonizer:** Transform images into cartoon-style artwork.
* **Canny Edge Detection:** Classic computer vision technique for robust edge detection.

---

### 🚀 Setup and Installation

#### Prerequisites

You must have **Python 3.8 or higher** installed to run the application.

#### 1. Clone the Repository

```bash
git clone [https://github.com/hem-il-shah/NeuralEye.git](https://github.com/hem-il-shah/NeuralEye.git)
cd NeuralEye
````

#### 2\. Install Dependencies

Install all required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### 3\. Download Model Weights (Critical Step)

> Due to file size limitations, some model weights are hosted externally. The **Image Segmentation Model (SAM)** must be downloaded manually to function.

| Feature | Model File | Destination Folder |
| :--- | :--- | :--- |
| Image Segmentation (SAM) | `sam_vit_h_1945395.pth` (or similar) | `model_is/` |

**Mandatory Step for Image Segmentation:**

1.  Download the necessary model weight file from the following Google Drive link:
      * **File Link:** `https://drive.google.com/file/d/1DCx11dB2oW7HsBHhN74PgK_rwHnXJyrq/view?usp=sharing`
2.  Place the downloaded file directly inside the **`model_is`** folder in your project directory.

-----

### ▶️ Running the Application

Start the Streamlit application from your terminal:

```bash
streamlit run app.py
```

Your web browser will automatically open the application. If it does not, navigate to `http://localhost:8501`.

-----

### 📂 Project Structure Overview

| Folder/File | Description |
| :--- | :--- |
| `app.py` | The main Streamlit file that runs the application and handles navigation. |
| `weights/` | Contains various YOLO and general object detection weights. |
| `model_is/` | **Folder for the Image Segmentation (SAM) model weights.** |
| `model_nst/` | Folder for the Neural Style Transfer model files. |
| `images/` | Sample images used for testing the functionalities. |
| `requirements.txt` | Lists all necessary Python dependencies for the project. |

```
