import os
import sys
import random
import shutil
import logging
import json
import time
import traceback
import warnings
import albumentations as A

import numpy as np
from tqdm import tqdm
import imageio
from skimage import measure, morphology
from skimage.transform import rescale

from imgseg.geojson_utils import geojson_to_masks
from data_utils import mask_to_geojson
from imgseg.hpa_seg_utils import label_nuclei, label_cell2
import asyncio
import janus
import json
from models.interactive_cellpose import CellPoseInteractiveModel

from data_utils import plot_images


os.makedirs("data", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("data", "log.txt")),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("interactive-trainer." + __name__)


def byte_scale(img):
    if img.dtype == np.dtype(np.uint16):
        img = np.clip(img, 0, 65535)
        img = img / 65535 * 255
    elif img.dtype == np.dtype(np.float32) or img.dtype == np.dtype(np.float64):
        img = (img * 255).round()
    elif img.dtype != np.dtype(np.uint8):
        raise Exception("Invalid image dtype " + str(img.dtype))
    return img


def load_image(img_path, channels, scale_factor):
    imgs = []
    last_img = None
    for ch in channels:
        if ch is None:
            imgs.append(None)
            continue
        img = imageio.imread(os.path.join(img_path, ch))
        if scale_factor != 1.0:
            img = rescale(
                img, scale_factor, multichannel=(img.ndim == 3), anti_aliasing=True
            )
        if len(img.shape) == 2:
            imgs.append(img)
            last_img = img
        elif len(img.shape) == 3:
            for i in range(img.shape[2]):
                imgs.append(img[:, :, i])
            last_img = img[:, :, 0]
        else:
            raise Exception("invalid image dimension number: " + str(len(img.shape)))

    # fill empty channels with zeros
    for i in range(len(imgs)):
        if imgs[i] is None:
            imgs[i] = np.zeros_like(last_img)

    # normalized = byte_scale(np.stack(imgs, axis=2)) / 255.0

    # return normalized.astype("float32")

    return np.stack(imgs, axis=2)


def load_sample_pool(data_dir, folder, input_channels, scale_factor, transform_labels):
    sample_list = [
        name
        for name in os.listdir(os.path.join(data_dir, folder))
        if not name.startswith(".")
    ]
    sample_pool = []
    logger.info(
        "loading samples, input channels: %s",
        input_channels,
    )
    for sample_name in tqdm(sample_list):
        sample_path = os.path.join(data_dir, folder, sample_name)
        if not all(
            [
                os.path.exists(os.path.join(data_dir, folder, sample_name, ch))
                for ch in input_channels
            ]
        ):
            continue
        img = load_image(sample_path, input_channels, scale_factor)

        annotation_file = os.path.join(data_dir, folder, sample_name, "annotation.json")
        mask_dict = geojson_to_masks(annotation_file, mask_types=["labels"])
        labels = mask_dict["labels"]
        mask = transform_labels(np.expand_dims(labels, axis=2))
        info = {"name": sample_name, "path": sample_path}
        sample_pool.append((img, mask, info))
    logger.info(
        "loaded %d samples from %s", len(sample_list), os.path.join(data_dir, folder)
    )
    return sample_pool


class InteractiveTrainer:
    __instance__ = None

    @staticmethod
    def get_instance(*args, **kwargs):
        """Static method to fetch the current instance."""
        instance = InteractiveTrainer.__instance__
        if len(args) == 0 and instance is None:
            raise Exception("No instance available")
        elif instance is None:
            return InteractiveTrainer(*args, **kwargs)
        elif len(args) == 0:
            return instance

        if instance._training_loop_running and instance.training_enabled:
            print(
                "Reconnecting to an existing training session, if you want to start a new one, please stop the training and run it again."
            )
            return instance
        else:
            instance.model = None

            InteractiveTrainer.__instance__ = None
            return InteractiveTrainer(*args, **kwargs)

    def __init__(
        self,
        model,
        data_dir,
        input_channels,
        folder="train",
        object_name="cell",
        batch_size=2,
        max_pool_length=30,
        min_object_size=100,
        scale_factor=1.0,
        resume=True,
    ):
        if InteractiveTrainer.__instance__ is None:
            InteractiveTrainer.__instance__ = self
        else:
            raise Exception(
                "You cannot create another InteractiveTrainer class, use InteractiveTrainer.get_instance() to retrieve the current instance."
            )
        self._training_error = None
        self._initialized = False
        self._training_loop_running = False
        self.training_enabled = False
        self.data_dir = data_dir
        assert model is not None
        self.model = model

        self.reports = []

        if resume:
            resume_weights_path = os.path.join(self.model.model_dir, "snapshot")
            if os.path.exists(resume_weights_path):
                print("Resuming model from " + resume_weights_path)
                self.model.load(resume_weights_path)

        self.object_name = object_name
        self.input_channels = input_channels
        self.scale_factor = scale_factor

        self.sample_pool = load_sample_pool(
            data_dir,
            folder,
            input_channels,
            self.scale_factor,
            self.model.transform_labels,
        )
        # _img, _mask, _info = self.sample_pool[0]

        self.batch_size = batch_size
        self.latest_samples = []
        self.max_pool_length = max_pool_length
        self.min_object_size = min_object_size

        self.training_config = {"save_freq": 200}
        self._initialized = True
        try:
            if sys.version_info < (3, 7):
                self.loop = asyncio.get_event_loop()
                asyncio.ensure_future(self.start_training_loop())
            else:
                asyncio.get_running_loop()
                asyncio.create_task(self.start_training_loop())
        except RuntimeError:
            asyncio.run(self.start_training_loop())

    def get_error(self):
        return self._training_error

    async def start_training_loop(self):
        self.queue = janus.Queue()
        if sys.version_info < (3, 7):
            self.loop = asyncio.get_event_loop()
        else:
            self.loop = asyncio.get_running_loop()
        self.loop.run_in_executor(
            None,
            self._training_loop,
            self.queue.sync_q,
            self.reports,
            self.training_config,
        )

    def train_once(self):
        batchX = []
        batchY = []
        for i in range(self.batch_size):
            x, y, info = self.get_random_training_sample()
            batchX += [x]
            batchY += [y]
            # augmented = self.augmentor(image=x, mask=y)
            # batchX += [augmented["image"]]
            # batchY += [augmented["mask"]]
        loss_metrics = self.model.train_on_batch(
            np.asarray(batchX, dtype=batchX[0].dtype),
            np.asarray(batchY, dtype=batchY[0].dtype),
        )
        return loss_metrics

    def load_input_image(self, folder, sample_name):
        img = load_image(
            os.path.join(self.data_dir, folder, sample_name),
            self.input_channels,
            self.scale_factor,
        )
        return img

    def load_target_image(self, folder, sample_name):
        annotation_file = os.path.join(
            self.data_dir, folder, sample_name, "annotation.json"
        )
        mask_dict = geojson_to_masks(annotation_file, mask_types=["labels"])
        labels = mask_dict["labels"]
        mask = self.model.transform_labels(np.expand_dims(labels, axis=2))
        return mask

    def _training_loop(self, sync_q, reports, training_config):
        self.training_enabled = False
        if len(self.reports) > 0:
            iteration = self.reports[-1]["iteration"]
        else:
            iteration = 0
        self._training_loop_running = True
        self._training_error = None
        self._exit = False
        while True:
            try:
                try:
                    if self._exit:
                        break
                    task = sync_q.get_nowait()
                    if task["type"] == "stop":
                        if self.training_enabled:
                            self.training_enabled = False
                            self.model.save(os.path.join(self.model.model_dir, "final"))
                    elif task["type"] == "start":
                        self.training_enabled = True
                    elif task["type"] == "predict":
                        self._prediction_result = self.predict(task["data"])
                    elif task["type"] == "push_sample":
                        self.push_sample(*task["args"], **task["kwargs"])
                    elif task["type"] == "plot_augmentations":
                        self._plot_augmentations_result = self.plot_augmentations()
                    else:
                        logger.warn("unsupported task type %s", task["type"])
                    sync_q.task_done()
                except janus.SyncQueueEmpty:
                    pass

                if self.training_enabled:
                    loss_metrics = self.train_once()
                    if not isinstance(loss_metrics, (list, tuple)):
                        loss_metrics = [loss_metrics]
                    iteration += 1
                    reports.append(
                        {
                            "loss": loss_metrics[0],
                            "iteration": iteration,
                        }
                    )
                    if iteration % training_config["save_freq"] == 0:
                        self.model.save(os.path.join(self.model.model_dir, "snapshot"))
                    # logger.info('trained for 1 iteration %s', reports[-1])
                else:
                    time.sleep(0.1)
            except Exception as e:
                self.training_enabled = False
                self._training_error = traceback.format_exc()
                self._training_loop_running = False
                logger.error("training loop exited with error: %s", e)
                break
        self._training_loop_running = False
        time.sleep(1)

    def start(self):
        logger.info("starting training")
        if not self._training_loop_running:
            try:
                if sys.version_info < (3, 7):
                    self.loop = asyncio.get_event_loop()
                    asyncio.ensure_future(self.start_training_loop())
                else:
                    asyncio.get_running_loop()
                    asyncio.create_task(self.start_training_loop())
            except RuntimeError:
                asyncio.run(self.start_training_loop())
        self.queue.sync_q.put({"type": "start"})

    def exit(self):
        logger.info("Exiting")
        self._exit = True

    def stop(self):
        logger.info("stopping training")
        self.queue.sync_q.put({"type": "stop"})

    def get_reports(self):
        return self.reports

    def get_random_training_sample(self):
        return random.choice(self.sample_pool)

    def get_training_sample(self, sample_name=None):
        return self.get_sample("train", sample_name)

    def get_test_sample(self, sample_name=None):
        return self.get_sample("test", sample_name)

    def get_sample(self, folder, sample_name=None):
        if sample_name is None:
            samples = [
                name
                for name in os.listdir(os.path.join(self.data_dir, folder))
                if not name.startswith(".")
            ]
            sample_name = random.choice(samples)
        img = self.load_input_image(folder, sample_name)
        info = {
            "name": sample_name,
            "path": os.path.join(self.data_dir, folder, sample_name),
        }
        return img, None, info

    def push_sample_async(self, *args, **kwargs):
        self.queue.sync_q.put({"type": "push_sample", "args": args, "kwargs": kwargs})

    def push_sample(
        self, sample_name, geojson_annotation, target_folder="train", prediction=None
    ):
        sample_dir = os.path.join(self.data_dir, "test", sample_name)
        img = imageio.imread(os.path.join(sample_dir, self.input_channels[0]))
        geojson_annotation["bbox"] = [0, 0, img.shape[0] - 1, img.shape[1] - 1]
        for item in geojson_annotation["features"]:
            item["properties"]["label"] = self.object_name
        with open(
            os.path.join(sample_dir, "annotation.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(geojson_annotation, f)

        new_sample_dir = os.path.join(self.data_dir, target_folder, sample_name)
        shutil.move(sample_dir, new_sample_dir)

        if target_folder == "train":
            # get mask_diff
            # prediction = imageio.imread(os.path.join(new_sample_dir, # "prediction.png"))
            # prediction = prediction.astype("int32")
            # mask = self.model.generate_mask(os.path.join(sample_dir, "annotation.json"))
            # mask_diff = np.abs(mask - prediction)
            # mask_diff = mask_diff[..., 1] + mask_diff[..., 2]
            # mask_diff = np.clip(mask_diff, 0, 255)
            # mask_diff = mask_diff.astype("uint8")
            # mask_diff = rescale(
            #     mask_diff,
            #     self.scale_factor,
            #     multichannel=(img.ndim == 3),
            #     anti_aliasing=True,
            # )
            # mask_diff = mask_diff.astype("float32")
            # mask_diff = mask_diff / mask_diff.max()

            img = self.load_input_image(target_folder, sample_name)
            mask = self.load_target_image(target_folder, sample_name)
            # mask[..., 0] = mask_diff

            self.sample_pool.append(
                (img, mask, {"name": sample_name, "path": new_sample_dir})
            )
            if len(self.sample_pool) > self.max_pool_length:
                self.sample_pool.pop(0)
            print("done with sample pushing")

    def predict_async(self, image):
        self._prediction_result = None
        self.queue.sync_q.put({"type": "predict", "data": image})

    def get_prediction_result(self):
        return self._prediction_result

    def predict(self, image):
        labels = self.model.predict(np.expand_dims(image, axis=0))
        labels = labels[0, :, :, 0]
        geojson_features = mask_to_geojson(
            np.flipud(labels), label=self.object_name, simplify_tol=None
        )
        return geojson_features, labels

    def plot_augmentations_async(self):
        self._plot_augmentations_result = None
        self.queue.sync_q.put({"type": "plot_augmentations"})

    def get_plot_augmentations(self):
        return self._plot_augmentations_result

    def plot_augmentations(self):
        batchX = []
        batchY = []
        x, y, _ = self.get_random_training_sample()
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        for i in range(4):
            x_, y_ = self.model.augment(x, y)
            x_ = np.swapaxes(np.squeeze(x_), 0, -1)
            if x_.shape[-1] == 2:
                x_3ch = np.dstack(
                    (np.zeros_like(x_[:, :, 0]), x_[:, :, 0], x_[:, :, 1])
                )
                x_ = x_3ch
            y_ = np.swapaxes(np.squeeze(y_), 0, -1)

            print(x_.shape, y_.shape)
            batchX += [x_]
            batchY += [y_]

        return plot_images(batchX, batchY, np.squeeze(x), np.squeeze(y))
