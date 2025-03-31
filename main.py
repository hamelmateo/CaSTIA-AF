from src.images.loader import load_images
from src.config import DATA_DIR, OUTPUT_DIR
from src.objects.segmentation import segmented



if __name__ == "__main__":

    # Load Hoechst images
    nucleis_img = load_images(DATA_DIR / "HOECHST")

    overlay_path = OUTPUT_DIR / "overlay.png"

    nuclei_mask=segmented(
        images=nucleis_img,  # Pass the loaded image for segmentation
        output_path=overlay_path,  # Save overlay image path
        save_overlay=True  # Set to True to save the overlay image
    )

    #TODO convert from labeled mask to list of Objects using np.unique


