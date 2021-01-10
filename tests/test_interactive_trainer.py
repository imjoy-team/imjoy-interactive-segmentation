import os
import sys
import time
import shutil
import asyncio
import threading
import numpy as np
from imageio import imwrite

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from interactive_trainer import InteractiveTrainer
from models.interactive_cellpose import CellPoseInteractiveModel
from data_utils import plot_history

model = CellPoseInteractiveModel(
    "./data/hpa_dataset_v2/__models__",
    # use_gpu=False,
    channels=[2, 3],
    style_on=0,
    default_diameter=100,
    pretrained_model=False,
)

trainer = InteractiveTrainer.get_instance(
    model,
    "./data/hpa_dataset_v2",
    ["microtubules.png", "er.png", "nuclei.png"],
    object_name="cell",
    scale_factor=1.0,
    batch_size=2,
    resume=False,
)


def test_train_once():
    for i in range(10):
        trainer.train_once()


def test_predict():
    sample = trainer.get_test_sample()
    geojson, mask = trainer.predict(sample[0])
    mask = np.clip(mask > 0 * 255, 0, 255).astype("uint8")


def test_workflow():
    data_dir = "./data/hpa_dataset_v2"
    val_name = "30609_1240_G3_4"
    val_sample = trainer.get_sample("valid", val_name)
    history = []
    data_size = []
    iter_size = 5000
    for i in range(iter_size):
        loss = trainer.train_once()
        print(loss, len(trainer.sample_pool))
        history += [loss]
        data_size += [len(trainer.sample_pool)]
        if i % 1000 == 0:
            geojson, mask = trainer.predict(val_sample[0])
            mask = np.clip(mask > 0 * 255, 0, 255).astype("uint8")
            imwrite(f"./data/hpa_dataset_v2/{val_name}_epoch_{i}.png", mask)
            _, _, info = trainer.get_test_sample()
            # trainer.push_sample(info.name, geojson_annotation, target_folder="train")
            sample_dir = info["path"]
            sample_name = info["name"]
            new_sample_dir = sample_dir.replace("test", "train")
            shutil.move(sample_dir, new_sample_dir)

            img = trainer.load_input_image("train", sample_name)
            mask = trainer.load_target_image("train", sample_name)
            trainer.sample_pool.append(
                (img, mask, {"name": sample_name, "path": new_sample_dir})
            )
            if len(trainer.sample_pool) > trainer.max_pool_length:
                trainer.sample_pool.pop(0)

    plot_history(
        history, data_size, iter_size, "./data/hpa_dataset_v2/test_history.png"
    )


if __name__ == "__main__":
    # test_train_once()
    # test_predict()
    test_workflow()
    os._exit(0)
