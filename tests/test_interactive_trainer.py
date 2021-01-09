import os
import sys
import time
import asyncio
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from interactive_trainer import InteractiveTrainer
from models.interactive_cellpose import CellPoseInteractiveModel

model = CellPoseInteractiveModel(
    "./data/hpa_dataset_v2/__models__",
    use_gpu=False,
    channels=[2, 3],
    style_on=0,
    default_diameter=100,
)
trainer = InteractiveTrainer.get_instance(
    model,
    "./data/hpa_dataset_v2",
    ["microtubules.png", "er.png", "nuclei.png"],
    object_name="cell",
    scale_factor=1.0,
    batch_size=1,
    resume=True,
)


def test_train_once():
    for i in range(10):
        trainer.train_once()


def test_predict():
    sample = trainer.get_test_sample()
    geojson, mask = trainer.predict(sample[0])
    mask = np.clip(mask > 0 * 255, 0, 255).astype("uint8")


if __name__ == "__main__":
    test_train_once()
    test_predict()
    os._exit(0)
