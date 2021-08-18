import os
import sys
import time
import shutil
import random
import numpy as np
from imageio import imread, imwrite

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from imjoy_interactive_trainer.models.interactive_unet_1 import UnetInteractiveModel
from imjoy_interactive_trainer.interactive_trainer import InteractiveTrainer, load_sample_pool, load_image
from imjoy_interactive_trainer.data_utils import (
    plot_history,
    fig2img,
    plot_mask_overlay,
)
from imjoy_interactive_trainer.imgseg.geojson_utils import geojson_to_masks

model_config = dict(
    type="unet",
    model_dir="./data/hpa_dataset_v2/__models__",
    use_gpu=False,
    channels=[2, 3],
    pretrained_model=None,
    resume=False
)
model = UnetInteractiveModel(**model_config)

data_dir =  "./data/hpa_dataset_v2"
folder = "train"
test_load_img = True
if test_load_img == True:
    sample_name = "10580_1758_B1_1"
    scale_factor=1
    input_channels =  ["microtubules.png", "er.png", "nuclei.png"]
    sample_path = os.path.join(data_dir, folder, sample_name)
    img = load_image(sample_path, input_channels, scale_factor)

    annotation_file = os.path.join(
        data_dir, folder, sample_name, "annotation.json"
    )
    mask_dict = geojson_to_masks(annotation_file, mask_types=["labels"])
    labels = mask_dict["labels"]
    mask = model.transform_labels(np.expand_dims(labels, axis=2))
    print(np.expand_dims(img, axis=0).shape, mask.shape)

sample_pool = load_sample_pool(data_dir, folder, ["microtubules.png", "er.png", "nuclei.png"], 1, model.transform_labels)
x, y, _ = random.choice(sample_pool)
x = np.expand_dims(x, axis=0)
y = np.expand_dims(y, axis=0)
#model.train()
print(x.shape, y.shape)
#x_, y_ = model.augment(x, y)
sample = model.augmentator(image=img, mask=mask)
x_, y_ = sample["image"], sample["mask"]
print("sample x y shapes:", x_.shape, y_.shape)
#model.predict(np.expand_dims(x_, axis=0))

batchX = []
batchY = []
config = model.get_config()
for i in range(config.get("batch_size", 1)):
    x, y, info = random.choice(sample_pool)
    batchX += [x]
    batchY += [y]
    #augmented = model.augmentator(image=x, mask=y)
    #batchX += [augmented["image"]]
    #batchY += [augmented["mask"]]
    print(np.asarray(batchX, dtype=batchX[0].dtype).dtype)
    loss_metrics = model.train_on_batch(
        np.asarray(batchX, dtype=batchX[0].dtype), 
        np.asarray(batchY, dtype=batchX[0].dtype)
    )
    print(loss_metrics)

trainer = InteractiveTrainer(
    model_config,
    "./data/hpa_dataset_v2",
    ["microtubules.png", "er.png", "nuclei.png"],
    scale_factor=1.0,
)
print(f"try testing shape {np.expand_dims(x_, axis=0).shape}, max {x_.max()}")
trainer.predict(x_)
#trainer.get_plot_augmentations()

