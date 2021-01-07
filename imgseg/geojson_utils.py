import os
import cv2
from skimage import io
import shutil
from imgseg import segmentationUtils
from imgseg import annotationUtils
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


def gen_mask_from_geojson(
    files_proc,
    masks_to_create_value=["filled", "edge", "distance", "weigthed", "border_mask"],
    img_size=None,
    infer=False,
):
    masks_to_create = {}

    # annot_types = list(masks_to_create.keys())

    annotationsImporter = annotationUtils.GeojsonImporter()

    # Instance to save masks
    masks = annotationUtils.MaskGenerator()

    weightedEdgeMasks = annotationUtils.WeightedEdgeMaskGenerator(sigma=8, w0=10)
    distMapMasks = annotationUtils.DistanceMapGenerator(truncate_distance=None)

    # %% Loop over all files
    for i, file_proc in enumerate(files_proc):
        print("PROCESSING FILE:")
        print(file_proc)

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
        print("annot_types: ", annot_types)

        for annot_type in annot_types:
            if infer:
                file_name_save = os.path.join(
                    drive, path, annot_type + "_filled_output.png"
                )
            else:
                file_name_save = os.path.join(drive, path, annot_type + "_filled.png")
            if os.path.exists(file_name_save):
                print("skip to generate mask:", file_name_save)
                continue
            # print("annot_type: ", annot_type)
            masks_to_create[annot_type] = masks_to_create_value

            # Filter the annotations by label
            annot_dict = {
                k: annot_dict_all[k]
                for k in annot_dict_all.keys()
                if annot_dict_all[k]["properties"]["label"] == annot_type
            }
            # print("len(annot_dict):", len(annot_dict))
            # print("annot_dict.keys():", annot_dict.keys())

            # Create masks

            # Binary - is always necessary to creat other masks
            print(" .... creating binary masks .....")
            binaryMasks = annotationUtils.BinaryMaskGenerator(
                image_size=image_size, erose_size=5, obj_size_rem=500, save_indiv=True
            )
            mask_dict = binaryMasks.generate(annot_dict)

            # Save binary masks FILLED if specified
            if "filled" in masks_to_create[annot_type]:
                if infer:
                    file_name_save = os.path.join(
                        drive, path, annot_type + "_filled_output.png"
                    )
                else:
                    file_name_save = os.path.join(
                        drive, path, annot_type + "_filled.png"
                    )
                masks.save(mask_dict, "fill", file_name_save)

            # Edge mask
            if "edge" in masks_to_create[annot_type]:
                if infer:
                    file_name_save = os.path.join(
                        drive, path, annot_type + "_edge_output.png"
                    )
                else:
                    file_name_save = os.path.join(drive, path, annot_type + "_edge.png")
                masks.save(mask_dict, "edge", file_name_save)

            # Distance map
            if "distance" in masks_to_create[annot_type]:
                print(" .... creating distance maps .....")
                mask_dict = distMapMasks.generate(annot_dict, mask_dict)

                # Save
                if infer:
                    file_name_save = os.path.join(
                        drive, path, annot_type + "_distmap_output.png"
                    )
                else:
                    file_name_save = os.path.join(
                        drive, path, annot_type + "_distmap.png"
                    )
                masks.save(mask_dict, "distance_map", file_name_save)

            # Weighted edge mask
            if "weigthed" in masks_to_create[annot_type]:
                print(" .... creating weighted edge masks .....")
                mask_dict = weightedEdgeMasks.generate(annot_dict, mask_dict)

                # Save
                if infer:
                    file_name_save = os.path.join(
                        drive, path, annot_type + "_edgeweight_output.png"
                    )
                else:
                    file_name_save = os.path.join(
                        drive, path, annot_type + "_edgeweight.png"
                    )
                masks.save(mask_dict, "edge_weighted", file_name_save)

            # border_mask
            if "border_mask" in masks_to_create[annot_type]:
                print(" .... creating border masks .....")
                border_detection_threshold = max(
                    round(1.33 * image_size[0] / 512 + 0.66), 1
                )
                borderMasks = annotationUtils.BorderMaskGenerator(
                    border_detection_threshold=border_detection_threshold
                )
                mask_dict = borderMasks.generate(annot_dict, mask_dict)

                # Save
                if infer:
                    file_name_save = os.path.join(
                        drive, path, annot_type + "_border_mask_output.png"
                    )
                else:
                    file_name_save = os.path.join(
                        drive, path, annot_type + "_border_mask.png"
                    )
                cv2.imwrite(
                    file_name_save,
                    mask_dict["border_mask"],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9],
                )
    print(" .... binary masks created .....")


if __name__ == "__main__":
    # # generate annotation from mask
    # datasets_dir = "/home/alex/Downloads/test/data"
    # save_path = "/home/alex/Downloads/test/data/kaggle_data/train"
    # masks_to_annotation(datasets_dir, save_path)
    #
    # # move the mask.png to the mask_labels.png
    # # for id in os.listdir(save_path):
    # #     shutil.move(os.path.join(save_path, id, "mask.png"),
    # #                 os.path.join(save_path, id, "mask_labels.png"))

    # generate mask from annotation.josn
    datasets_dir = "/home/alex/Downloads/test/data/kaggle_data"
    err_list = []
    for file_id in os.listdir(os.path.join(datasets_dir, "train")):
        file_path = os.path.join(datasets_dir, "train", file_id, "annotation.json")
        # gen_mask_from_geojson([file_path], masks_to_create_value=["border_mask"])
        try:
            gen_mask_from_geojson([file_path], masks_to_create_value=["border_mask"])
        except:
            print("generate mask error:", os.path.join(datasets_dir, "train", file_id))
            err_list.append(file_id)
    print("err_list:", err_list)

    # # change the mask file name
    # for file_id in os.listdir(os.path.join(datasets_dir, "train")):
    #     file_path = os.path.join(datasets_dir, "train", file_id)
    #     for id in os.listdir(file_path):
    #         try:
    #             shutil.move(os.path.join(file_path, "nuclei_weighted_boarder.png"),
    #                         os.path.join(file_path, "nuclei_border_mask.png"))
    #         except:
    #             if os.path.exists(os.path.join(file_path, "nuclei_border_mask.png")):
    #                 print("file exist:", os.path.join(file_path, "nuclei_border_mask.png"))
    #             elif not os.path.exists(os.path.join(file_path, "nuclei_weighted_boarder.png")):
    #                 print("file not exist:", os.path.join(file_path, "nuclei_weighted_boarder.png"))
    #             else:
    #                 print("move error:", os.path.join(file_path, "nuclei_weighted_boarder.png"))
