from src.images.loader import load_and_crop_images
from src.config import DATA_DIR, OUTPUT_DIR, ROI_SCALE
from src.objects.segmentation import segmented
from src.objects.cell import Cell
import numpy as np
import tifffile
import time
import pickle


if __name__ == "__main__":

    start = time.time()
    print(f"[INFO] Pipeline started at {time.ctime(start)}")
    # Check if nuclei_mask.TIF exists and a boolean flag is set to True
    print("[DEBUG 1] Starting pipeline...")

    load_existing_cells=True  # Set this flag to True if you want to load existing cells from a file
    cells_file_path = OUTPUT_DIR / "cells.pkl"
    
    if load_existing_cells and cells_file_path.exists():
        print("[DEBUG] Loading existing cells from file...")
        with open(cells_file_path, "rb") as f:
            cells = pickle.load(f)
        print(f"[INFO] Loaded {len(cells)} cells from {cells_file_path}")
    
    else:
        load_existing_mask = True  # Set this flag as needed
        nuclei_mask_path = OUTPUT_DIR / "nuclei_mask.TIF"

        if load_existing_mask and nuclei_mask_path.exists():
            print("[DEBUG 2] Loading existing nuclei mask from file.")
            nuclei_mask = tifffile.imread(nuclei_mask_path)
        else:
            print("[DEBUG 3] Loading Hoechst images...")
            nucleis_imgs = load_and_crop_images(DATA_DIR / "HOECHST", ROI_SCALE)

            overlay_path = OUTPUT_DIR / "overlay.png"

            print("[DEBUG 4] Performing segmentation...")
            nuclei_mask = segmented(
                images=nucleis_imgs,  # Pass the loaded image for segmentation
                output_path=overlay_path,  # Save overlay image path
                save_overlay=True  # Set to True to save the overlay image
            )

            print("[DEBUG 5] Saving nuclei mask...")
            tifffile.imwrite(nuclei_mask_path, nuclei_mask)
            print(f"[INFO] Nuclei mask saved to {nuclei_mask_path}")


        # Convert from labeled mask to list of Objects using np.unique
        print("[DEBUG 6] Converting labeled mask to Cell objects...")
        cells = []
        label = 0
        while (np.any(nuclei_mask == label)):
            # Get the coordinates of all pixels with the current label
            pixel_coords = np.argwhere(nuclei_mask == label)
            
            if pixel_coords.size > 0:
                # Calculate centroid as the mean of the pixel coordinates
                centroid = np.array(np.mean(pixel_coords, axis=0), dtype=int)
                # Create a new Cell object with the current label and pixel coordinates
                cell = Cell(label=label, centroid=centroid, pixel_coords=pixel_coords)
                
                # Check if the centroid is less than 20 pixels from the border
                if (centroid[0] < 20 or centroid[1] < 20 or 
                    centroid[0] > nuclei_mask.shape[0] - 20 or 
                    centroid[1] > nuclei_mask.shape[1] - 20):
                    cell.is_valid = False  # Mark the cell as inactive
                
                cells.append(cell)
            
            label += 1

        # Save cells to an external file for debugging purposes

        print("[DEBUG] Saving cells to external file for future debugging...")
        with open(cells_file_path, "wb") as f:
            pickle.dump(cells, f)


    print(f"[DEBUG 7] Number of cells detected: {len(cells)}")
    if len(cells) > 25:
        print(f"[INFO] Example cell centroid: {cells[25].centroid}")
    else:
        print("[INFO] Less than 26 cells detected.")
    

    # Determine active vs inactive cells
    print("[DEBUG 8] Determining active vs inactive cells...")
    active_cells = [cell for cell in cells if cell.is_valid]
    inactive_cells = [cell for cell in cells if not cell.is_valid]

    print(f"[INFO] Number of active cells: {len(active_cells)}")
    print(f"[INFO] Number of inactive cells: {len(inactive_cells)}")


    # Load FITC images (if any) and add timepoints to each cell
    print("[DEBUG 9] Loading FITC images...")
    calcium_imgs = load_and_crop_images(DATA_DIR / "FITC_temp", ROI_SCALE)
    if calcium_imgs.size > 0:
        print("[DEBUG 10] Adding timepoints to cells...")
        for idx, img in enumerate(calcium_imgs):
            print(f"[DEBUG] Processing image {idx + 1}...")
            cells_to_process = len(cells)
            for cell in cells:
                cell.add_mean_intensity(img)  # Add the current image as a timepoint for each cell
    else:
        print("[INFO] No FITC images found.")

    # Plot the intensity traces of the 5 random active cells
    print("[DEBUG 11] Plotting intensity traces of 5 first active cells...")
    for i, cell in enumerate(active_cells[:5]):
        print(f"[INFO] Plotting intensity profile for Cell {cell.label}...")
        cell.plot_intensity_profile()

    print("[DEBUG 12] Pipeline completed successfully.")
    end = time.time()
    print(f"[INFO] Total time taken: {end - start:.2f} seconds")