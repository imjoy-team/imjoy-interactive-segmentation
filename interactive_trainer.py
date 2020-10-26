import os
import random
import shutil
import logging
import json
import time
import tensorflow as tf
import segmentation_models as sm
import albumentations as A
import numpy as np
from tqdm import tqdm
import imageio
from skimage import measure, morphology
from skimage.transform import rescale

from imgseg.geojson_utils import gen_mask_from_geojson
from data_utils import mask_to_geojson
from imgseg.hpa_seg_utils import label_nuclei
import asyncio
import janus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("data", "log.txt")),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("interactive-trainer." + __name__)


def load_unet_model(model_path=None, backbone="mobilenetv2"):
    preprocess_input = sm.get_preprocessing(backbone)
    if model_path and os.path.exists(model_path):
        logger.info("model loaded from %s", model_path)
        return tf.keras.models.load_model(model_path)

    # define model
    model = sm.Unet(
        backbone,
        encoder_weights="imagenet",
        classes=3,
        activation="sigmoid",
        layers=tf.keras.layers,
        models=tf.keras.models,
        backend=tf.keras.backend,
        utils=tf.keras.utils,
    )
    model.compile(
        "Adam",
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    logger.info("model built from scratch, backbone: %s", backbone)
    return model


def get_augmentor():
    return A.Compose(
        [
            A.RandomCrop(362, 362),
            A.OneOf(
                [
                    A.VerticalFlip(),
                    A.HorizontalFlip(),
                    A.RandomBrightnessContrast(p=0.8),
                    A.RandomGamma(p=0.8),
                ],
                p=1,
            ),
            A.RandomRotate90(p=1),
            A.CenterCrop(256, 256),
        ]
    )


def byte_scale(img):
    if img.dtype == np.dtype(np.uint16):
        img = np.clip(img, 0, 65535)
        img = img / 65535 * 255
    elif img.dtype == np.dtype(np.float32) or img.dtype == np.dtype(np.float64):
        img = (img * 255).round()
    elif img.dtype != np.dtype(np.uint8):
        raise Exception("Invalid image dtype " + img.dtype)
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

    normalized = byte_scale(np.stack(imgs, axis=2)) / 255.0

    return normalized.astype("float32")


def load_sample_pool(data_dir, folder, input_channels, target_channels, scale_factor):
    sample_list = [
        name
        for name in os.listdir(os.path.join(data_dir, folder))
        if not name.startswith(".")
    ]
    sample_pool = []
    logger.info(
        "loading samples, input channels: %s, target channeles: %s",
        input_channels,
        target_channels,
    )
    for sample_name in tqdm(sample_list):
        sample_path = os.path.join(data_dir, folder, sample_name)
        img = load_image(sample_path, input_channels, scale_factor)
        mask = load_image(sample_path, target_channels, scale_factor)
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
        if not InteractiveTrainer.__instance__:
            InteractiveTrainer(*args, **kwargs)
        return InteractiveTrainer.__instance__

    def __init__(
        self,
        data_dir,
        input_channels,
        folder="train",
        object_name="cell",
        mask_type="border_mask",
        batch_size=2,
        max_pool_length=30,
        min_object_size=100,
        scale_factor=1.0,
        initial_pool=True,
    ):
        if InteractiveTrainer.__instance__ is None:
            InteractiveTrainer.__instance__ = self
        else:
            raise Exception(
                "You cannot create another InteractiveTrainer class, use InteractiveTrainer.get_instance() to retrieve the current instance."
            )
        self.data_dir = data_dir
        self.model_dir = os.path.join(self.data_dir, "__models__")
        # load latest model if exists
        self.model = load_unet_model(os.path.join(self.model_dir, "model_latest.h5"))
        self.object_name = object_name
        self.mask_type = mask_type
        self.input_channels = input_channels
        self.target_channels = [f"{self.object_name}_{self.mask_type}.png"]
        self.scale_factor = scale_factor
        self.queue = janus.Queue()
        if initial_pool:
            self.sample_pool = load_sample_pool(
                data_dir,
                folder,
                input_channels,
                self.target_channels,
                self.scale_factor,
            )
            _img, _mask, _info = self.sample_pool[0]
            assert (
                _img.shape[2] == self.model.input_shape[3]
            ), f"shape mismatch: { _img.shape[2] } != {self.model.input_shape[3]}"
            assert (
                _mask.shape[2] == self.model.output_shape[3]
            ), f"shape mismatch: { _mask.shape[2] } != {self.model.output_shape[3]}"
        else:
            self.sample_pool = []

        self.augmentor = get_augmentor()
        self.batch_size = batch_size

        self.latest_samples = []
        self.max_pool_length = max_pool_length
        self.min_object_size = min_object_size
        self.loop = asyncio.get_running_loop()
        self.reports = []
        self.training_config = {"save_freq": 200}
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
            x, y, info = self.get_training_sample()
            augmented = self.augmentor(image=x, mask=y)
            batchX += [augmented["image"]]
            batchY += [augmented["mask"]]
        loss = self.model.train_on_batch(
            np.asarray(batchX, dtype="float32"), np.asarray(batchY, dtype="float32")
        )
        return loss

    def load_input_image(self, folder, sample_name):
        img = load_image(
            os.path.join(self.data_dir, folder, sample_name),
            self.input_channels,
            self.scale_factor,
        )
        return img

    def load_target_image(self, folder, sample_name):
        img = load_image(
            os.path.join(self.data_dir, folder, sample_name),
            self.target_channels,
            self.scale_factor,
        )
        return img

    def _training_loop(self, sync_q, reports, training_config):
        training_enabled = False
        iteration = 0
        while True:
            try:
                task = sync_q.get(block=False)
                if task["type"] == "stop":
                    training_enabled = False
                elif task["type"] == "start":
                    training_enabled = True
                elif task["type"] == "predict":
                    result = self.predict(task["data"])
                    task["callback"](result)
                else:
                    logger.warn("unsupported task type %s", task["type"])
                sync_q.task_done()
            except janus.SyncQueueEmpty:
                pass

            if training_enabled:
                loss = self.train_once()
                iteration += 1
                reports.append({"loss": loss, "iteration": iteration})
                if iteration % training_config["save_freq"] == 0:
                    self.save_model()
                # logger.info('trained for 1 iteration %s', reports[-1])
            else:
                time.sleep(0.1)

    def save_model(self, label="latest"):
        os.makedirs(self.model_dir, exists_ok=True)
        self.model.save(os.path.join(self.model_dir, f"model_{label}.h5"))

    def start(self):
        logger.info("starting training")
        self.queue.sync_q.put({"type": "start"})

    def stop(self):
        logger.info("stopping training")
        self.queue.sync_q.put({"type": "stop"})

    def get_reports(self):
        return self.reports

    def get_training_sample(self):
        return random.choice(self.sample_pool)

    def get_test_sample(self):
        samples = [
            name
            for name in os.listdir(os.path.join(self.data_dir, "test"))
            if not name.startswith(".")
        ]
        sample_name = random.choice(samples)
        img = self.load_input_image("test", sample_name)
        info = {
            "name": sample_name,
            "path": os.path.join(self.data_dir, "test", sample_name),
        }
        return img, None, info

    def push_sample(self, sample_name, geojson_annotation, target_folder="train"):
        sample_dir = os.path.join(self.data_dir, "test", sample_name)
        img = imageio.imread(os.path.join(sample_dir, self.input_channels[0]))
        geojson_annotation["bbox"] = [0, 0, img.shape[0] - 1, img.shape[1] - 1]
        for item in geojson_annotation["features"]:
            item["properties"]["label"] = self.object_name
        with open(
            os.path.join(sample_dir, "annotation.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(geojson_annotation, f)

        files = [os.path.join(sample_dir, "annotation.json")]
        gen_mask_from_geojson(files, masks_to_create_value=[self.mask_type])
        new_sample_dir = os.path.join(self.data_dir, target_folder, sample_name)
        shutil.move(sample_dir, new_sample_dir)

        if target_folder == "train":
            img = self.load_input_image(target_folder, sample_name)
            mask = self.load_target_image(target_folder, sample_name)
            self.sample_pool.append(
                (img, mask, {"name": sample_name, "path": new_sample_dir})
            )
            if len(self.sample_pool) > self.max_pool_length:
                self.sample_pool.pop(0)

    def predict(self, image):
        mask = self.model.predict(np.expand_dims(image, axis=0))
        labels = label_nuclei(mask[0, :, :, :])
        geojson = mask_to_geojson(labels, label=self.object_name, simplify_tol=1.5)
        return geojson