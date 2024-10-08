import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Directory paths
IMAGE_DIR = r''  # Update this path to where your PNG images are stored
OUTPUT_DIR = r''

# Function to apply data augmentation (optional)
def augment_image(image):
    # Randomly apply transformations (e.g., rotation, flipping)
    image = image.rotate(np.random.choice([0, 90, 180, 270]))  # Random rotation
    if np.random.rand() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Horizontal flip
    if np.random.rand() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Vertical flip
    return image

# Function to preprocess a single image
def preprocess_image(img_path, resize_shape=(128, 128), augment=False):
    try:
        # Open image
        img = Image.open(img_path)
        
        # Optionally augment the image
        if augment:
            img = augment_image(img)
        
        # Resize the image
        img = img.resize(resize_shape)
        
        # Convert image to numpy array and normalize pixel values
        img_array = np.array(img) / 255.0  # Normalizing pixel values to the range [0, 1]
        
        # Ensure that the image is in the correct format (e.g., RGB or grayscale)
        if len(img_array.shape) == 2:  # If grayscale, add an additional dimension
            img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")
        return None

# Function to preprocess all PNG images using multithreading
def preprocess_png_images(image_dir, output_file, resize_shape=(128, 128), augment=False, batch_size=100):
    print("Preprocessing PNG images...")
    processed_images = []
    count = 0

    def process_image_batch(batch):
        nonlocal processed_images
        processed_batch = [preprocess_image(img, resize_shape, augment) for img in batch]
        processed_images.extend([img for img in processed_batch if img is not None])
        logging.info(f"Processed {len(processed_images)} images so far.")
    
    with ThreadPoolExecutor() as executor:
        # List all PNG files
        img_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')]
        
        # Split into batches and submit them for processing
        for i in range(0, len(img_files), batch_size):
            batch = img_files[i:i + batch_size]
            executor.submit(process_image_batch, batch)

    # Save the processed images as a NumPy array file (.npy)
    if len(processed_images) > 0:
        np.save(output_file, np.array(processed_images))
        logging.info(f"Processed {len(processed_images)} PNG images saved to {output_file}")
    else:
        logging.warning("No PNG images were processed.")

# Main function to handle preprocessing
def main():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Preprocess PNG images with data augmentation and multithreading
    output_file = os.path.join(OUTPUT_DIR, 'processed_png_images.npy')
    preprocess_png_images(IMAGE_DIR, output_file, resize_shape=(128, 128), augment=True, batch_size=50)

if __name__ == '__main__':
    main()
