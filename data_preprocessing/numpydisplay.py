import numpy as np
import matplotlib.pyplot as plt

# Path to your .npy file
npy_file_path = r'C:\Users\boopa\Downloads\Bijie-landslide-dataset\landslide\output\processed_png_images.npy'

# Load the .npy file
preprocessed_images = np.load(npy_file_path)

# Check the shape of the array (number of images, image dimensions, channels)
print("Shape of preprocessed images:", preprocessed_images.shape)

# Visualize a few preprocessed images
for i in range(20):  # Adjust the range to view more images
    plt.imshow(preprocessed_images[i])
    plt.title(f"Preprocessed Image {i+1}")
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Path to your .npy file
npy_file_path = r''

# Load the .npy file
preprocessed_images = np.load(npy_file_path)

# Check the shape of the array (number of images, image dimensions, channels)
print("Shape of preprocessed images:", preprocessed_images.shape)

# Visualize a few preprocessed images
for i in range(20):  # Adjust the range to view more images
    plt.imshow(preprocessed_images[i])
    plt.title(f"Preprocessed Image {i+1}")
    plt.show()
