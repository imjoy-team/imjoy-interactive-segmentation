import os
import random
import shutil
import logging
import tensorflow as tf
import segmentation_models as sm
import albumentations as A
import numpy as np
from tqdm import tqdm

from imgseg.geojson_utils import gen_mask_from_geojson
from data_utils import mask_to_geojson
import imageio


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("data", "log.txt")),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("interactive-trainer." + __name__)


def load_unet_model(backbone="mobilenetv2"):
    preprocess_input = sm.get_preprocessing(backbone)
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


def load_image(img_path, channels, resize=None):
    imgs = []
    for ch in channels:
        logger.info('reading ' + ch)
        img = imageio.imread(os.path.join(img_path, ch))
        if resize is not None:
            new_x, new_y = resize
            img = img.resize((new_x, new_y))
        if len(img.shape) == 2:
            imgs.append(img)
        elif len(img.shape) == 3:
            for i in range(img.shape[2]):
                imgs.append(img[:, :, i])
        else:
            raise Exception("invalid image dimension number: " + str(len(img.shape)))
    normalized = byte_scale(np.stack(imgs, axis=2)) / 255.0
    return normalized.astype("float32")


def load_sample_pool(
    data_dir, folder, input_channels, target_channels, target_size=None
):  # folder='test'
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
        img = load_image(sample_path, input_channels, target_size)
        mask = load_image(sample_path, target_channels, target_size)
        info = {"name": sample_name, "path": sample_path}
        sample_pool.append((img, mask, info))
    logger.info(
        "loaded %d samples from %s", len(sample_list), os.path.join(data_dir, folder)
    )
    return sample_pool


class InteractiveTrainer:
    def __init__(
        self, data_dir, input_channels, folder="train", object_name="cell", mask_type="border_mask", batch_size=2, max_pool_length=30
    ):
        self.model = load_unet_model()
        self.input_channels = input_channels
        self.object_name = object_name
        self.mask_type = mask_type
        self.target_channels = [f"{self.object_name}_{self.mask_type}.png"]
        self.sample_pool = load_sample_pool(
            data_dir, folder, input_channels, self.target_channels
        )
        self.augmentor = get_augmentor()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.latest_samples = []
        self.max_pool_length = max_pool_length

    def train_once(self):
        batchX = []
        batchY = []
        for i in range(self.batch_size):
            x, y, info = self.get_training_sample(sample_pool)
            augmented = self.augmentor(image=x, mask=y)
            batchX += [augmented["image"]]
            batchY += [augmented["mask"]]
        self.model.train_on_batch(
            np.asarray(batchX, dtype="float32"), np.asarray(batchY, dtype="float32")
        )

    def load_input_image(self, folder, sample_name):
        img = load_image(
            os.path.join(data_dir, folder, sample_name), self.input_channels
        )
        return img

    def load_target_image(self, folder, sample_name):
        img = load_image(
            os.path.join(data_dir, folder, sample_name), self.target_channels
        )
        return img

    def start(self):
        logger.info("start training")
        while True:
            self.train_once()

    def get_training_sample(self):
        return random.choice(self.sample_pool)

    def get_test_sample(self):
        sample_name = random.choice(os.listdir(os.path.join(self.data_dir, "test")))
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
        mask = self.model.predict(image)
        geojson = mask_to_geojson(mask, label="cell", simplify_tol=1.5)
        return geojson
