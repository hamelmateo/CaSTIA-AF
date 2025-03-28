import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image(
    image: np.ndarray,
    title: str = "Image",
    size: tuple = (24, 24)
) -> None:
    """
    Show a 16-bit grayscale image at a specified resolution and with a specific color map.
    
    Parameters:
        image (np.ndarray): The image to display (16-bit grayscale).
        title (str): The title of the image (e.g., file name). Default is "Image".
        size (tuple): The size of the figure in inches (width, height). Default is (12, 12).
    
    Returns:
        None: This function does not return any value.
    """
    # Error handling
    if not isinstance(image, np.ndarray):
        raise TypeError("The 'image' parameter must be a numpy ndarray.")
    if image.dtype != np.uint16:
        raise ValueError("The 'image' must be a 16-bit grayscale image.")
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("The 'size' parameter must be a tuple of two integers (width, height).")

    # Display the image
    plt.figure(figsize=size)
    plt.imshow(image, vmin=0, vmax=65535)
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()