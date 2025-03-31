from src.images.loader import load_and_crop_images
from src.config import DATA_DIR, OUTPUT_DIR, ROI_SCALE
from src.objects.segmentation import segmented
from src.objects.cell import Cell  
import numpy as np


if __name__ == "__main__":

    # Load Hoechst images
    nucleis_imgs = load_and_crop_images(DATA_DIR / "HOECHST", ROI_SCALE)

    overlay_path = OUTPUT_DIR / "overlay.png"

    nuclei_mask=segmented(
        images=nucleis_imgs,  # Pass the loaded image for segmentation
        output_path=overlay_path,  # Save overlay image path
        save_overlay=True  # Set to True to save the overlay image
    )

    # Convert from labeled mask to list of Objects using np.unique
    cells = []
    label = 0
    while (np.any(nuclei_mask == label)):
        # Get the coordinates of all pixels with the current label
        pixel_coords = np.argwhere(nuclei_mask == label)
        
        if pixel_coords.size > 0:
            # Calculate centroid as the mean of the pixel coordinates
            centroid = np.mean(pixel_coords, axis=0)
            # Create a new Cell object with the current label and pixel coordinates
            cell = Cell(label=label, centroid=centroid, pixel_coords=pixel_coords)
            
            # Check if the centroid is less than 20 pixels from the border
            if (centroid[0] < 20 or centroid[1] < 20 or 
                centroid[0] > nuclei_mask.shape[0] - 20 or 
                centroid[1] > nuclei_mask.shape[1] - 20):
                cell.set_inactive()  # Mark the cell as inactive
            
            cells.append(cell)
        
        label += 1
    
    print(f"[INFO] Number of cells detected: {len(cells)}")
    print(f"[INFO] Example cell centroid: {cells[25].centroid if cells else 'No cells detected'}")
    

    # Determine active vs inactive cells
    active_cells = [cell for cell in cells if cell.is_active()]
    inactive_cells = [cell for cell in cells if not cell.is_active()]

    print(f"[INFO] Number of active cells: {len(active_cells)}")
    print(f"[INFO] Number of inactive cells: {len(inactive_cells)}")


    # Load FITC images (if any) and add timepoints to each cell
    calcium_imgs = load_and_crop_images(DATA_DIR / "FITC", ROI_SCALE)
    if calcium_imgs.size > 0:
        for img in calcium_imgs:
            for cell in cells:
                cell.add_timepoint(img)  # Add the current image as a timepoint for each cell

