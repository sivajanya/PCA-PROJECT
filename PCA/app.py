import streamlit as st
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

# Directory for saving uploads and compressed files
UPLOAD_FOLDER = 'uploads'
COMPRESSED_FOLDER = 'compressed'

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

def reduce_image(file_name, accuracy, output_path):
    """Compresses the image using PCA."""
    image = io.imread(file_name)
    gray_image = color.rgb2gray(image)

    # Apply PCA
    pca = PCA(n_components=accuracy)
    transformed_image = pca.fit_transform(gray_image)
    reconstructed_image = pca.inverse_transform(transformed_image)

    # Normalize and save the compressed image
    compressed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (
        reconstructed_image.max() - reconstructed_image.min()
    )
    compressed_image_uint8 = img_as_ubyte(compressed_image_normalized)
    io.imsave(output_path, compressed_image_uint8)

st.title("🎨 Image Compression using PCA")
st.markdown("Upload your image, select compression accuracy, and download the compressed image.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an Image File", type=["png", "jpg", "jpeg"])

# Dropdown for accuracy selection
accuracy = st.selectbox("Select Compression Accuracy", [0.8, 0.9, 0.95, 0.99], index=2)

if uploaded_file is not None:
    # Save the uploaded file to the UPLOAD_FOLDER
    image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    # Define compressed file path
    compressed_filename = f"compressed_{uploaded_file.name}"
    compressed_path = os.path.join(COMPRESSED_FOLDER, compressed_filename)

    # Compress the image
    reduce_image(image_path, accuracy, compressed_path)
    st.success("✅ Image compressed successfully!")

    # Display the compressed image
    st.image(compressed_path, caption="Compressed Image", use_column_width=True)

    # Provide a download link for the compressed image
    with open(compressed_path, "rb") as file:
        btn = st.download_button(
            label="📥 Download Compressed Image",
            data=file,
            file_name=compressed_filename,
            mime="image/jpeg",
        )
