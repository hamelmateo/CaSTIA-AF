from src.Images.loader import load_dapi_image
from src.Images.visualize import show_image
from pathlib import Path
from src.config import DAPI_DIR


if __name__ == "__main__":
    
    # Load and show DAPI image
    dapi_image = load_dapi_image(DAPI_DIR)
    show_image(dapi_image, "DAPI image", (24, 24))
