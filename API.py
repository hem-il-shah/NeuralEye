# import matplotlib.pylab as plt
# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# import io

# def transfer_style(content_image, style_image, model_path):
#     """
#     :param content_image: PIL Image object of the content image
#     :param style_image: PIL Image object of the style image
#     :param model_path: Path to the downloaded pre-trained model.

#     :return: A stylized image as a 3D NumPy array.
#     """

#     print("Loading images...")

#     # Convert PIL Images to NumPy arrays
#     content_image = np.array(content_image).astype(np.float32) / 255.0
#     style_image = np.array(style_image).astype(np.float32) / 255.0

#     # Add batch dimension
#     content_image = np.expand_dims(content_image, axis=0)
#     style_image = np.expand_dims(style_image, axis=0)

#     print("Resizing and Normalizing images...")
#     # Resize style image to recommended size
#     style_image = tf.image.resize(style_image, (256, 256))

#     print("Loading pre-trained model...")
#     hub_module = hub.load(model_path)

#     print("Generating stylized image now...wait a minute")
#     # Stylize the image
#     outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
#     stylized_image = outputs[0].numpy()

#     # Remove batch dimension and reshape
#     stylized_image = np.squeeze(stylized_image, axis=0)

#     print("Stylizing completed...")
#     return stylized_image

# ---------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import io

def transfer_style(content_image, style_image, model_path):
    """
    :param content_image: PIL Image object of the content image
    :param style_image: PIL Image object of the style image
    :param model_path: Path to the downloaded pre-trained model.

    :return: A stylized image as a 3D NumPy array.
    """

    print("i.) Loading images... 🎢")
    st.write("i.) Loading images... 🎢")

    # Convert PIL Images to NumPy arrays
    content_image = np.array(content_image).astype(np.float32) / 255.0
    style_image = np.array(style_image).astype(np.float32) / 255.0

    # Add batch dimension
    content_image = np.expand_dims(content_image, axis=0)
    style_image = np.expand_dims(style_image, axis=0)

    print("ii.) Resizing and Normalizing images... ⚙️")
    st.write("ii.) Resizing and Normalizing images... ⚙️")
    # Resize style image to recommended size
    style_image = tf.image.resize(style_image, (256, 256))

    print("iii.) Loading the model... 🧬")
    st.write("iii.) Loading the model... 🧬")
    hub_module = hub.load(model_path)

    print("iv.) Generating stylized image now... Wait a minute...! ✏️")
    st.write("iv.) Generating stylized image now... Wait a minute...! ✏️")
    # Stylize the image
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0].numpy()

    # Remove batch dimension and reshape
    stylized_image = np.squeeze(stylized_image, axis=0)

    print("Output: Stylizing completed... 💝")
    st.write("Output: Stylizing completed... 💝")
    return stylized_image
