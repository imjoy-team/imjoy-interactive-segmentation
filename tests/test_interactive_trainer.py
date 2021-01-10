import os
import sys
import time
import shutil
import asyncio
import threading
import numpy as np
from imageio import imread, imwrite

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from interactive_trainer import InteractiveTrainer
from data_utils import plot_history, fig2img, plot_mask_overlay
import cellpose

model_config = dict(
    type="cellpose",
    model_dir="./data/hpa_dataset_v2/__models__",
    use_gpu=True,
    channels=[2, 3],
    style_on=0,
    default_diameter=100,
    pretrained_model=False,
    resume=False,
)

trainer = InteractiveTrainer.get_instance(
    model_config,
    "./data/hpa_dataset_v2",
    ["microtubules.png", "er.png", "nuclei.png"],
    object_name="cell",
    scale_factor=1.0,
)


def test_train_once():
    for i in range(3):
        trainer.train_once()


def test_predict():
    sample = trainer.get_test_sample()
    geojson, mask = trainer.predict(sample[0])
    mask = np.clip(mask > 0 * 255, 0, 255).astype("uint8")


def test_workflow():
    time_list = [i for i in range(10, 120, 10)]
    """time_list = [
        0,
        2 * 60,
        4 * 60 + 30,
        5 * 60 + 50,
        8 * 60 + 50,
        9 * 60 + 30,
        10 * 60 + 30,
        11 * 60 + 10,
        12 * 60 + 10,
        14 * 60 + 10,
        16 * 60 + 30,
        17 * 60 + 30,
        20 * 60,
        21 * 60,
        21 * 60 + 30,
        25 * 60 + 30,
        26 * 60 + 30,
        28 * 60 + 30,
        29 * 60 + 30,
        30 * 60,
        30 * 60 + 30,
        31 * 60,
        31 * 60 + 40,
        32 * 60 + 10,
        33 * 60 + 30,
        35 * 60,
        35 * 60 + 40,
        36 * 60 + 20,
        36 * 60 + 50,
        37 * 60 + 30,
        38 * 60 + 50,
        40 * 60 + 50,
        41 * 60 + 30,
        42 * 60 + 20,
        43 * 60,
        43 * 60 + 40,
        44 * 60 + 50,
        45 * 60 + 40,
        46 * 60 + 30,
        47 * 60,
        47 * 60 + 40,
        47 * 60 + 40,
        49 * 60 + 10,
        50 * 60,
        50 * 60 + 40,
        54 * 60 + 40,
        55 * 60 + 10,
        56 * 60,
        57 * 60,
        58 * 60 + 30,
        60 * 60,
        60 * 60 + 50,
        61 * 60 + 50,
        62 * 60 + 20,
        63 * 60,
        66 * 60,
        66 * 60 + 40,
        67 * 60 + 10,
    ]"""
    output = False
    t = time.time()
    data_dir = "./data/hpa_dataset_v2"
    save_dir = data_dir + '/save'
    os.makedirs(save_dir, exist_ok=True)
    val_name = "30609_1240_G3_4"
    val_sample = trainer.get_sample("valid", val_name)
    val_img = trainer.load_input_image("valid", val_name)
    imwrite(f"{save_dir}/{val_name}_image.png", val_img)
    # val_mask = trainer.load_target_image("valid", val_name)
    # cellpose.io.imsave(f"{data_dir}/{val_name}_gt_mask.png", val_mask)
    history = []
    data_size = []
    iter_size = 10000000000000000
    time_limit = 90 * 60
    time_order = 0
    for i in range(iter_size):
        loss = trainer.train_once()
        history += [loss]
        data_size += [len(trainer.sample_pool)]
        #if i % 500 == 0:
        time_point = time.time() - t
        if time_point > time_list[time_order]:
            time_order += 1
            if time_order > time_limit:
                break
            timestamp = int(time.time() - t)
            print("iteration", i, timestamp)
            geojson, mask = trainer.predict(val_sample[0])
            cellpose.io.imsave(f"{save_dir}/{val_name}_epoch_{i}_time_{timestamp}.png", mask)
            #mask = imread(f"{data_dir}/{val_name}_epoch_{i}.png")
            #plot_mask_overlay(
            #    val_img, mask, f"{data_dir}/{val_name}_epoch_{i}_imgoverlay.png"
            #)
            _, _, info = trainer.get_test_sample()
            # trainer.push_sample(info.name, geojson_annotation, target_folder="train")
            sample_dir = info["path"]
            sample_name = info["name"]
            new_sample_dir = sample_dir.replace("test", "train")
            #shutil.move(sample_dir, new_sample_dir)

            ## this part is commented out only for the first two minutes
            #img = trainer.load_input_image("train", sample_name)
            #mask = trainer.load_target_image("train", sample_name)
            #trainer.sample_pool.append(
            #    (img, mask, {"name": sample_name, "path": new_sample_dir})
            #)
            ## comment out till here
            if len(trainer.sample_pool) > trainer.max_pool_length:
                trainer.sample_pool.pop(0)
    print(f"{time.time()-t} to train {iter_size} iterations")
    plot_history(
        history, data_size, iter_size, "./data/hpa_dataset_v2/test_history.png"
    )


def test_aug_plot():
    tmp = trainer.plot_augmentations()
    tmp = np.flipud(tmp)
    imwrite("./data/hpa_dataset_v2/test_aug.png", tmp)


if __name__ == "__main__":
    #test_train_once()
    #test_predict()
    #test_aug_plot()
    test_workflow()
    #os._exit(0)
