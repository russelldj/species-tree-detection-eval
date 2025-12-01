from pathlib import Path
from tree_registration_and_matching.register_MEE import align_plot
import geopandas as gpd

import numpy as np

FIELD_TREES_FILE = Path(
    "/ofo-share/scratch/david/species-tree-detection-eval/data/inputs/ofo_ground-reference_trees.gpkg"
)
FIELD_PLOT_BOUNDS_FILE = Path(
    "/ofo-share/scratch/david/species-tree-detection-eval/data/inputs/ofo_ground-reference_plots.gpkg"
)
DETECTED_TREES_FOLDER = Path(
    "/ofo-share/scratch/david/species-tree-detection-eval/data/outputs/a0.0_b0.0325_c0.25"
)

def get_shifted_trees(detected_tree_file):
    detected_trees = gpd.read_file(detected_tree_file)

    plot_id = detected_tree_file.stem.split("_")[0]

    ground_reference_trees_plot = ground_reference_trees[
        ground_reference_trees["plot_id"] == plot_id
    ]
    obs_bounds_plot = obs_bounds[obs_bounds["plot_id"] == plot_id]

    print(detected_tree_file)
    print(obs_bounds_plot)
    print(ground_reference_trees_plot)

    shifted_field_trees, final_shift = align_plot(
        field_trees=ground_reference_trees_plot,
        drone_trees=detected_trees,
        obs_bounds=obs_bounds_plot,
        vis=True,
    )
    return final_shift

SHIFTED = [
    "0010_000273_000271",
    "0020_000278_000279",
    "0030_000452_000448",
    "0044_000152_000151",
    "0051_000174_000171",
    "0060_000349_000348",
    "0073_000230_000223",
    "0080_000232_000225",
    "0090_000190_000193",
    "0100_000153_000155",
    "0110_000136_000133",
    "0121_000015_000013",
    "0130_000015_000013",
    "0140_000208_000210",
    "0150_000221_000229",
    "0160_000211_000220",
    "0170_000452_000448",
    "0180_001238_001237",
    "0190_001218_001219",
    "0200_001011_001012",
    "0210_000112_000310",
    "0220_000311_000322",
    "0230_000315_000314",
    "0240_000317_000316",
]

detected_tree_files = [Path(DETECTED_TREES_FOLDER, f"{plot}_tree_tops.gpkg") for plot in SHIFTED]
ground_reference_trees = gpd.read_file(FIELD_TREES_FILE)
obs_bounds = gpd.read_file(FIELD_PLOT_BOUNDS_FILE)

# Clean up field trees and add height where missing

ground_reference_trees = ground_reference_trees[ground_reference_trees.live_dead != "D"]

# First replace any missing height values with pre-computed allometric values
nan_height = ground_reference_trees.height.isna()
ground_reference_trees[nan_height].height = ground_reference_trees[
    nan_height
].height_allometric

# For any remaining missing height values that have DBH, use an allometric equation to compute
# the height
nan_height = ground_reference_trees.height.isna()
# These parameters were fit on paired height, DBH data from this dataset.
allometric_height_func = lambda x: 1.3 + np.exp(
    -0.3136489123372108 + 0.84623571 * np.log(x)
)
# Compute the allometric height and assign it
allometric_height = allometric_height_func(
    ground_reference_trees[nan_height].dbh.to_numpy()
)
ground_reference_trees.loc[nan_height, "height"] = allometric_height

# Filter out any trees that still don't have height (just 1 in current experiments)
ground_reference_trees = ground_reference_trees[~ground_reference_trees.height.isna()]

shifts = [
    get_shifted_trees(detected_tree_file) for detected_tree_file in detected_tree_files
]
print(shifts)
breakpoint()