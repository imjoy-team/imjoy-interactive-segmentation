import os
import cellpose
from models.cellpose import CellPoseInteractiveModel
from cellpose.utils2 import read_multi_channel_image, geojson_to_label, read_image

channels = ["er.png", "nuclei.png"]
mask_filter = "cell_masks.png"
folder = "./data/hpa_dataset_v2/train/3213_1239_D9_1"
annotation_file = os.path.join(folder, 'annotation.json')

# create labels file
geojson_to_label(annotation_file, save_as='_masks.png')

# read the input image
X = read_multi_channel_image(folder, channels, rescale=1.0)

# read the labels
y = read_image(folder + "/cell_masks.png", rescale=1.0)

model = CellPoseInteractiveModel()

def test_train_once():
    model.train_once(X, y)

def test_predict():
    y_predict = model.predict(X)
    assert y_predict.shape == (512, 512)
    cellpose.io.imsave(os.path.join(folder, "cellpose_predicted_mask.png"), y_predict)

if __name__ == "__main__":
    test_train_once()
    test_predict()
