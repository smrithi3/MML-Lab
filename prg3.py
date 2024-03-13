import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skimage import io, color

def compress_image(image_path, n_components):
    # Load the image
    image = io.imread(image_path)
    
    # Convert the image to grayscale
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    # Flatten the image into a 1D array
    flat_image = image.flatten()
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    compressed_image = pca.fit_transform(flat_image.reshape(1, -1))
    
    # Inverse transform to get the compressed image back
    reconstructed_image = pca.inverse_transform(compressed_image)
    
    # Reshape the image to its original shape
    reconstructed_image = reconstructed_image.reshape(image.shape)
    
    return reconstructed_image

def plot_images(original_image, compressed_image):
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(compressed_image, cmap='gray')
    plt.title('Compressed Image')
    
    plt.show()

# Example usage
image_path =  'C:\\Users\\SMRITHI\\Desktop\\MML-Lab\\image.jpg'

n_components = 0  # Adjust the number of components according to your needs

original_image = io.imread(image_path)
compressed_image = compress_image(image_path, n_components)
plot_images(original_image, compressed_image)