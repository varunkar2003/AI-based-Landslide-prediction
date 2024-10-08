import os
import numpy as np
import pandas as pd
import cv2  # For image processing (JPG, PNG)
import h5py  # For HDF5 files
from PIL import Image, ImageEnhance  # For handling and augmenting images
from sklearn.preprocessing import StandardScaler
import concurrent.futures  # For multithreading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory paths
IMAGE_DIR = '../data/images'
H5_DIR = '../data/h5_files'
CSV_DIR = '../data/csv_files'
OUTPUT_DIR = '../output/'

# Function to apply augmentation to images
def augment_image(img):
    # Apply random augmentations (brightness, contrast, etc.)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(np.random.uniform(0.8, 1.2))  # Random brightness adjustment
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(np.random.uniform(0.8, 1.2))  # Random contrast adjustment
    img = img.rotate(np.random.uniform(-15, 15))  # Random rotation
    return img

# Function to preprocess a single image
def preprocess_single_image(img_file, image_dir, resize_dim=(128, 128), augment=False):
    img_path = os.path.join(image_dir, img_file)
    try:
        img = Image.open(img_path)
        if augment:
            img = augment_image(img)
        img = img.resize(resize_dim)  # Resize image to provided dimensions
        img_array = np.array(img) / 255.0  # Normalize the image
        return img_array
    except Exception as e:
        logging.error(f"Error processing image {img_file}: {e}")
        return None

# Function to preprocess images with multithreading and optional augmentation
def preprocess_images(image_dir, resize_dim=(128, 128), augment=False):
    logging.info("Preprocessing images...")
    processed_images = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(preprocess_single_image, img_file, image_dir, resize_dim, augment)
                   for img_file in os.listdir(image_dir) if img_file.endswith(('.png', '.jpg'))]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                processed_images.append(result)
    return np.array(processed_images)

# Function to preprocess HDF5 files with multithreading
def preprocess_single_hdf5(h5_file, h5_dir):
    h5_path = os.path.join(h5_dir, h5_file)
    try:
        with h5py.File(h5_path, 'r') as f:
            data = f['dataset'][()]  # Replace 'dataset' with the correct dataset name
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            return data_scaled
    except Exception as e:
        logging.error(f"Error processing HDF5 file {h5_file}: {e}")
        return None

def preprocess_hdf5(h5_dir):
    logging.info("Preprocessing HDF5 files...")
    processed_data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(preprocess_single_hdf5, h5_file, h5_dir)
                   for h5_file in os.listdir(h5_dir) if h5_file.endswith('.h5')]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                processed_data.append(result)
    return processed_data

# Function to preprocess a single CSV file
def preprocess_single_csv(csv_file, csv_dir):
    csv_path = os.path.join(csv_dir, csv_file)
    try:
        df = pd.read_csv(csv_path)
        df.fillna(method='ffill', inplace=True)  # Fill missing values
        df_scaled = StandardScaler().fit_transform(df.select_dtypes(include=[np.number]))  # Scale numerical data
        return df_scaled
    except Exception as e:
        logging.error(f"Error processing CSV file {csv_file}: {e}")
        return None

# Function to preprocess CSV files with multithreading
def preprocess_csv(csv_dir):
    logging.info("Preprocessing CSV files...")
    processed_csvs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(preprocess_single_csv, csv_file, csv_dir)
                   for csv_file in os.listdir(csv_dir) if csv_file.endswith('.csv')]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                processed_csvs.append(result)
    return processed_csvs

# Main function to handle preprocessing based on file format
def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Preprocess images
    if os.path.exists(IMAGE_DIR):
        processed_images = preprocess_images(IMAGE_DIR, augment=True)  # Enable augmentation
        if len(processed_images) > 0:
            np.save(os.path.join(OUTPUT_DIR, 'processed_images.npy'), processed_images)
            logging.info(f"Processed {len(processed_images)} images saved to processed_images.npy")

    # Preprocess HDF5 files
    if os.path.exists(H5_DIR):
        processed_h5 = preprocess_hdf5(H5_DIR)
        if len(processed_h5) > 0:
            np.save(os.path.join(OUTPUT_DIR, 'processed_h5.npy'), processed_h5)
            logging.info(f"Processed HDF5 data saved to processed_h5.npy")

    # Preprocess CSV files
    if os.path.exists(CSV_DIR):
        processed_csv = preprocess_csv(CSV_DIR)
        if len(processed_csv) > 0:
            np.save(os.path.join(OUTPUT_DIR, 'processed_csv.npy'), processed_csv)
            logging.info(f"Processed CSV data saved to processed_csv.npy")

if __name__ == '__main__':
    main()
