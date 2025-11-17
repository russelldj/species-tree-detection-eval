from pathlib import Path
from tree_detection_framework.preprocessing.preprocessing import create_dataloader
from tree_detection_framework.detection.detector import GeometricTreeTopDetector
from tree_detection_framework.postprocessing.postprocessing import (
    remove_edge_detections,
)

INPUT_FOLDER = "/ofo-share/species-prediction-project/intermediate/CHMs"
OUTPUT_FOLDER = "/ofo-share/scratch-david/species-tree-detection-eval/predictions"
PARAMETER_SETS = [
    {"c": 1.3, "b": 0.0222, "a": 0.0},
    {"c": 0.1, "b": -0.1, "a": 0.0},
    {"c": 1.9, "b": 0.046, "a": 0.0},
    {"c": 0.9, "b": 0.046, "a": 0.0},
    {"c": 1.1, "b": 0.046, "a": 0.0},
    {"c": 0.1, "b": 0.096, "a": 0.0},
    {"c": 0.5, "b": 0.046, "a": 0.0},
    {"c": 0.1, "b": 0.046, "a": 0.0},
    {"c": 0.7, "b": 0.046, "a": 0.0},
]
CHIP_SIZE = 2000
CHIP_STRIDE = 1900
RESOLUTION = 0.2

if __name__ == "__main__":
    input_files = Path(INPUT_FOLDER).glob("*.tif")

    for params in PARAMETER_SETS:
        output_subfolder = (
            Path(OUTPUT_FOLDER)
            / f"predictions_c{params['c']}_b{params['b']}_a{params['a']}"
        )

        for file in input_files:
            output_file = (output_subfolder / file.stem).with_suffix(".gpkg")
            dataloader = create_dataloader(
                raster_folder_path=file,
                chip_size=CHIP_SIZE,
                chip_stride=CHIP_STRIDE,
                resolution=RESOLUTION,
            )
            treetop_detector = GeometricTreeTopDetector(
                confidence_feature="distance", **params
            )

            # Generate tree top predictions
            treetop_detections = treetop_detector.predict(dataloader)

            suppression_distance = (
                0
                if len(dataloader) == 1
                else (CHIP_SIZE - CHIP_STRIDE) * RESOLUTION / 2
            )
            # Remove suppressed detections
            treetop_detections = remove_edge_detections(
                treetop_detections,
                suppression_distance=suppression_distance,
            )

            treetop_detections.save(output_file)
