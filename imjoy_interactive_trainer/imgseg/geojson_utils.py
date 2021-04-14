import os
import cv2
from skimage import io
import shutil
from imjoy_interactive_trainer.imgseg import segmentationUtils
from imjoy_interactive_trainer.imgseg import annotationUtils
from geojson import FeatureCollection, dump
from skimage import measure


def masks_to_annotation(datasets_dir, save_path):
    masks_dir = os.path.join(datasets_dir, "labels_all")
    nucleis_dir = os.path.join(datasets_dir, "images_all")

    # %% Process one folder and save as one json file allowing multiple annotation types
    simplify_tol = (
        0  # Tolerance for polygon simplification with shapely (0 to not simplify)
    )

    # outputs_dir = os.path.abspath(os.path.join('..', 'data', 'postProcessing', 'mask2json'))
    if os.path.exists(masks_dir):
        print(f"Analyzing folder:{masks_dir}")
        for file in [f for f in os.listdir(masks_dir)]:
            file_id = os.path.splitext(file)[0]

            # Read png with mask
            print(f"Analyzing file:{file}")

            file_full = os.path.join(masks_dir, file)
            mask_img = io.imread(file_full)
            print("mask_img.shape:", mask_img.shape)
            mask = measure.label(mask_img)
            label = "nuclei"
            print("label:", label)
            sample_path = os.path.join(save_path, file_id)
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            io.imsave(os.path.join(sample_path, "mask_labels.png"), mask_img)
            shutil.copyfile(
                os.path.join(nucleis_dir, file.replace(".tif", ".png")),
                os.path.join(sample_path, "nuclei.png"),
            )
            segmentationUtils.masks_to_polygon(
                mask,
                label=label,
                simplify_tol=simplify_tol,
                save_name=os.path.join(sample_path, "annotation.json"),
            )


def geojson_to_masks(
    file_proc, mask_types=["filled", "edge", "labels"], img_size=None,
):

    # annot_types = list(masks_to_create.keys())

    annotationsImporter = annotationUtils.GeojsonImporter()

    # Instance to save masks
    masks = annotationUtils.MaskGenerator()

    weightedEdgeMasks = annotationUtils.WeightedEdgeMaskGenerator(sigma=8, w0=10)
    distMapMasks = annotationUtils.DistanceMapGenerator(truncate_distance=None)

    # Decompose file name
    drive, path_and_file = os.path.splitdrive(file_proc)
    path, file = os.path.split(path_and_file)
    # file_base, ext = os.path.splitext(file)

    # Read annotation:  Correct class has been selected based on annot_type
    annot_dict_all, roi_size_all, image_size = annotationsImporter.load(file_proc)
    if img_size is not None:
        image_size = img_size

    annot_types = set(
        annot_dict_all[k]["properties"]["label"] for k in annot_dict_all.keys()
    )
    masks = {}
    for annot_type in annot_types:
        # print("annot_type: ", annot_type)
        # Filter the annotations by label
        annot_dict = {
            k: annot_dict_all[k]
            for k in annot_dict_all.keys()
            if annot_dict_all[k]["properties"]["label"] == annot_type
        }
        # Create masks
        # Binary - is always necessary to creat other masks
        binaryMasks = annotationUtils.BinaryMaskGenerator(
            image_size=image_size, erose_size=5, obj_size_rem=500, save_indiv=True
        )
        mask_dict = binaryMasks.generate(annot_dict)

        # Distance map
        if "distance" in mask_types:
            mask_dict = distMapMasks.generate(annot_dict, mask_dict)

        # Weighted edge mask
        if "weigthed" in mask_types:
            mask_dict = weightedEdgeMasks.generate(annot_dict, mask_dict)

        # border_mask
        if "border_mask" in mask_types:
            border_detection_threshold = max(
                round(1.33 * image_size[0] / 512 + 0.66), 1
            )
            borderMasks = annotationUtils.BorderMaskGenerator(
                border_detection_threshold=border_detection_threshold
            )
            mask_dict = borderMasks.generate(annot_dict, mask_dict)

    return mask_dict
