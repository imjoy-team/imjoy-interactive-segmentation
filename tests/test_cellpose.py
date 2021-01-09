import os
import sys

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from imgseg.geojson_utils import geojson_to_masks

import cellpose
from models.interactive_cellpose import CellPoseInteractiveModel
from cellpose.utils2 import (
    read_multi_channel_image,
    geojson_to_label,
    read_image,
    load_train_test_data,
)
from download_example_dataset import download_with_url


dataset_path = "./data/hpa_dataset_v2.zip"
if not os.path.exists(dataset_path):
    url = "https://kth.box.com/shared/static/hcnspau5lndyhkkzgv2ygsyq1978qo90.zip"
    print("downloading dataset from " + url)
    download_with_url(url, dataset_path, unzip=True)
    print("dataset saved to " + dataset_path)

channels = ["er.png", "nuclei.png"]
mask_filter = "cell_masks.png"
folder = "./data/hpa_dataset_v2/test/407_1852_D7_32"
annotation_file = os.path.join(folder, "annotation.json")
mask_filter = "cells_masks.png"
image_filter = "er.png,nuclei.png"

# create labels file
geojson_to_label(annotation_file, save_as="_masks.png")

# read the input image
X = (read_multi_channel_image(folder, channels, rescale=1.0)).transpose(1, 2, 0)

model = CellPoseInteractiveModel(
    "./data/hpa_dataset_v2/__models__", style_on=0, default_diameter=100
)


# read the labels

labels = cellpose.io.imread(folder + "/cell_masks.png")
y = model.transform_labels(np.expand_dims(labels, axis=2))


def test_train_on_batch():
    global X, y
    assert y.shape[:2] == (512, 512)
    X = np.expand_dims(X, axis=0)
    y = np.expand_dims(y, axis=0)
    model.train_on_batch(X, y)


def test_transform_labels():
    mask_dict = geojson_to_masks(annotation_file, mask_types=["labels"])
    labels = mask_dict["labels"]
    flows = model.transform_labels(np.expand_dims(labels, axis=2))


def test_train():
    output = load_train_test_data(
        "./data/hpa_dataset_v2/train/",
        "./data/hpa_dataset_v2/test/",
        image_filter.split(","),
        mask_filter,
        unet=False,
        rescale=1.0,
    )
    images, labels, image_names, test_images, test_labels, image_names_test = output
    model.train((images, labels), (test_images, test_labels), iterations=len(images))


def test_train_steps():
    output = load_train_test_data(
        "./data/hpa_dataset_v2/train/",
        None,
        image_filter.split(","),
        mask_filter,
        unet=False,
        rescale=1.0,
    )
    images, labels, image_names, test_images, test_labels, image_names_test = output
    # diams = np.array([cellpose.utils.diameters(labels[i])[0] for i in range(len(images))])
    # mean_diam = diams.mean()

    for i in range(10):
        y_predict = model.predict(X)
        assert y_predict.shape == (512, 512)
        file_path = os.path.join(
            "./data/cellpose_predicted_mask_epoch_{}.png".format(i * 100)
        )
        cellpose.io.imsave(file_path, y_predict)

        print("Training for round {}".format(i))
        model.train(
            (images, labels), (test_images, test_labels), iterations=len(images) * 100
        )


def test_predict():
    y_predict = model.predict(np.expand_dims(X, axis=0))
    assert y_predict.shape == (1, 512, 512, 1), "shape of prediction {}".format(
        y_predict.shape
    )
    file_path = os.path.join("./data/cellpose_predicted_mask.png")
    cellpose.io.imsave(file_path, y_predict[0, :, :, 0])
    assert os.path.exists(file_path)
    # os.remove(file_path)

    # save ground truth
    file_path = os.path.join("./data/cellpose_ground_truth_mask.png")
    if y.shape[0] == 4:
        yt = y[1:, :, :]
    else:
        yt = y[0, :, :]
    cellpose.io.imsave(file_path, yt)


def test_save():
    model_path = "./data/cellpose_model.pth"
    model.save(model_path)
    assert os.path.exists(model_path)
    # os.remove(model_path)


if __name__ == "__main__":
    # test_train()
    # test_train_steps()
    test_transform_labels()
    test_predict()
    test_save()
    test_train_on_batch()
