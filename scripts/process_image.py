"""
Script to process a single calcium imaging .TIF file using ImageProcessor.

Example:
    $ python process_image.py /path/to/frame0005.TIF
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from calcium_activity_characterization.config.presets import ImageProcessingConfig, ImageProcessingPipeline, HotPixelMethod, HotPixelParameters
from calcium_activity_characterization.preprocessing.image_processing import ImageProcessor
from calcium_activity_characterization.io.export import save_tif_image



def main(input_dir: Path, output_dir: Path) -> None:
    """
    Process a single TIF image and save it to another directory.

    Args:
        image_path (Path): Path to the .TIF image file.
        output_dir (Path): Directory where the processed image will be saved.
    """
    # Define default config (you can tune as needed)

    config = ImageProcessingConfig(
        pipeline=ImageProcessingPipeline(
            padding=False,
            cropping=True,
            hot_pixel_cleaning=True
        ),
        padding_digits=5,
        roi_scale=0.75,
        roi_centered=False,
        hot_pixel_cleaning=HotPixelParameters(
            method=HotPixelMethod.CLIP,           # "replace" or "clip"
            static_threshold=50000,                   # used if auto threshold disabled
            use_auto_threshold=False,
            percentile=99.9,
            mad_scale=10.0,
            window_size=3
        )
    )

    processor = ImageProcessor(config=config)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all tif images (case-insensitive)
    #image_paths = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.TIF"))
    image_paths = [
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01723.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01725.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01727.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01735.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01737.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01739.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01741.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01743.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01745.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01747.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01749.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01751.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01753.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01755.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01757.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01759.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01761.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01763.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01765.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01767.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01769.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01771.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01773.TIF"),
        Path("D:/Mateo/20250409/Data/IS10/FITC/20250409__w3FITC_t01775.TIF"),
    ]

    if not image_paths:
        print(f"No .tif images found in {input_dir}")
        return

    for image_path in image_paths:
        print(f"Processing {image_path.name}...")
        processed_img = processor.process_single_image(image_path)

        # Define output path
        output_path = output_dir / f"processed_{image_path.name}"

        # Save as 16-bit TIFF
        save_tif_image(processed_img, output_path)

        print(f"Processed image saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_image.py <path_to_image.TIF> <output_directory>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    if not img_path.exists():
        print(f"Error: {img_path} does not exist.")
        sys.exit(1)

    main(img_path, out_dir)