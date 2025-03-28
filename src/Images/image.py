import numpy as np
import cv2
import matplotlib.pyplot as plt

class Image:
    def __init__(self, array: np.ndarray = None, title: str = None, path: str = None, channel: int = None, mask: np.ndarray = None, objects: list = None):
        """
        Initialize the Image class.

        :param array: np.ndarray, the image data.
        :param title: str, the title of the image.
        :param path: str, the path to the image file.
        :param channel: str, the channel of the image (e.g., 'DAPI', 'FITC', 'BF').
        :param mask: np.ndarray, optional, a mask for the image.
        :param objects: list, optional, a list of objects detected in the image.
        """
        self.array = array if array is not None else np.array([])
        self.title = title
        self.path = path
        self.channel = channel
        self.mask = mask if mask is not None else (np.zeros_like(array) if array is not None else None)
        self.objects = objects if objects is not None else []

    @classmethod
    def load_image(cls, path: str, channel: int = 1):
        """
        Load an image from a file.

        :param path: str, the path to the image file.
        :param channel: int, the channel to load (1 for grayscale, 3 for RGB).
        :return: Image instance.
        """
        array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        title = path.split('/')[-1]
        return cls(array=array, title=title, path=path, channel=channel)


    def visualize(self):
        """
        Visualize the image using matplotlib.
        """
        plt.figure(figsize=(8, 8))
        if self.channel == 1:
            plt.imshow(self.array, cmap='gray')
        elif self.channel == 3:
            plt.imshow(self.array)
        else:
            raise ValueError("Unsupported channel for visualization.")
        plt.title(self.title)
        plt.axis('off')
        plt.show()

    def crop(self, x_start: int, y_start: int, width: int, height: int):
        """
        Crop the image within a specified region of interest (ROI).

        :param x_start: int, the starting x-coordinate of the ROI.
        :param y_start: int, the starting y-coordinate of the ROI.
        :param width: int, the width of the ROI.
        :param height: int, the height of the ROI.
        :return: np.ndarray, the cropped image.
        """
        x_end = x_start + width
        y_end = y_start + height
        cropped_array = self.array[y_start:y_end, x_start:x_end]
        return cropped_array

    def apply_mask(self):
        """
        Apply the mask to the image.
        """
        if self.mask is None:
            raise ValueError("No mask available to apply.")
        self.array = cv2.bitwise_and(self.array, self.array, mask=self.mask)

    def add_object(self, obj: str):
        """
        Add an object to the list of objects in the image.

        :param obj: str, the name of the object to add.
        """
        self.objects.append(obj)

    def __repr__(self):
        """
        String representation of the Image class.
        """
        return f"Image(title={self.title}, path={self.path}, channel={self.channel}, objects={self.objects})"
