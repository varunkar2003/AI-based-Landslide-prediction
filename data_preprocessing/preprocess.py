import os
import numpy as np
import cv2
import h5py
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to preprocess standard images (PNG, JPG)
def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess PNG or JPG image by resizing and normalizing it.
    
    :param image_path: str, path to the PNG/JPG image file.
    :param target_size: tuple, desired image size (width, height).
    :return: numpy array, preprocessed image ready for prediction.
    """
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)  # Resize the image to the target size
        img_array = np.array(img) / 255.0  # Normalize pixel values to range [0, 1]
        
        # If the image is grayscale, expand its dimensions
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
        
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

# Function to preprocess HDF5 images (including mask and img components)
def preprocess_hdf5_image(h5_file, target_size=(128, 128)):
    """
    Preprocess HDF5 image that contains both 'img' and 'mask' components.
    
    :param h5_file: str, path to the HDF5 file.
    :param target_size: tuple, desired image size (width, height).
    :return: numpy array, preprocessed image ready for prediction.
    """
    try:
        with h5py.File(h5_file, 'r') as f:
            img = f['img'][()]  # Replace with actual dataset key if different
            mask = f['mask'][()]  # Replace with actual dataset key if different

            # Resize and normalize the img
            img = cv2.resize(img, target_size)
            img = img / 255.0  # Normalize pixel values to range [0, 1]

            # Resize the mask if needed and combine with img (optional)
            mask = cv2.resize(mask, target_size)
            combined = np.concatenate((img, mask), axis=-1) if mask is not None else img

            return np.expand_dims(combined, axis=0)  # Add batch dimension
    except Exception as e:
        logging.error(f"Error processing HDF5 file {h5_file}: {e}")
        return None

# Main preprocessing function to handle any image input type
def preprocess_image_input(image_input, target_size=(128, 128)):
    """
    Determine the type of image input (PNG/JPG or HDF5) and preprocess accordingly.
    
    :param image_input: str, path to the image file (PNG, JPG, HDF5).
    :param target_size: tuple, desired image size (width, height).
    :return: numpy array, preprocessed image ready for prediction.
    """
    if image_input.endswith(('.png', '.jpg', '.jpeg')):  # Handle standard image formats
        return preprocess_image(image_input, target_size)
    elif image_input.endswith('.h5'):  # Handle HDF5 image format
        return preprocess_hdf5_image(image_input, target_size)
    else:
        logging.error("Unsupported file format.")
        return None
