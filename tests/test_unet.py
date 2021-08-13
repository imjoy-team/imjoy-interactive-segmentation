import os
import sys
import time
import shutil
import numpy as np
from imageio import imread, imwrite

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from imjoy_interactive_trainer.interactive_trainer import InteractiveTrainer
from imjoy_interactive_trainer.data_utils import (
    plot_history,
    fig2img,
    plot_mask_overlay,
)

model_config = dict(
    type="unet",
    model_dir="./data/hpa_dataset_v2/__models__",
    use_gpu=False,
    channels=[2, 3],
    pretrained_model=None,
    resume=False
)
print(model_config["type"])

from imjoy_interactive_trainer.models.interactive_unet_1 import UnetInteractiveModel
model = UnetInteractiveModel(**model_config)


trainer = InteractiveTrainer.get_instance(
    model_config,
    "./data/hpa_dataset_v2",
    ["microtubules.png", "er.png", "nuclei.png"],
    object_name="cell",
    scale_factor=1.0,
)
