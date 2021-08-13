import os
import tensorflow as tf
import albumentations as A

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as Loss
from torch.nn.modules.loss import _Loss
from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import warnings
import logging



logger = logging.getLogger("interactive-trainer." + __name__)
SMOOTH = 1e-5

class BCEJaccardLoss(_Loss):
    def __init__(
        self,
        beta=1,
        class_weights=None,
        class_indexes=None,
        per_image=False,
        smooth=SMOOTH,
    ):
        super().__init__(name="bce_jaccard_loss")
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = smooth
        self.beta = beta

    def dice_loss(self, gt, pr):
        return 1 - F.f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None,
            **self.submodules,
        )

    def jacacard_loss(self, gt, pr):
        return 1 - F.iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None,
            **self.submodules,
        )

    def __call__(self, gt, pr):
        # blue channel
        ce_body = F.binary_crossentropy(
            gt[:, :, :, 2:3], pr[:, :, :, 2:3], **self.submodules
        )
        # green channel
        ce_border = F.binary_crossentropy(
            gt[:, :, :, 1:2], pr[:, :, :, 1:2], **self.submodules
        )

        dice_body = self.dice_loss(gt[:, :, :, 2:3], pr[:, :, :, 2:3])
        dice_border = self.dice_loss(gt[:, :, :, 1:2], pr[:, :, :, 1:2])
        return 0.6 * (ce_body + ce_border) + 0.2 * (dice_body + dice_border)


def zero_mean_unit_var(x):
    xm = x.mean()
    return (x - x.mean()) / x.std()


def load_unet_model(model_path=None, backbone="mobilenet_v2"):
    # disable warnings temporary
    warnings.filterwarnings("ignore")

    # preprocess_input = smp.get_preprocessing(backbone)

    if model_path:
        logger.info("model loaded from %s", model_path)
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        # define model
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            classes=3,
            activation="sigmoid"
        )
        logger.info("model built from scratch, backbone: %s", backbone)

    #model.compile("Adam", loss=Loss.JaccardLoss())

    warnings.resetwarnings()
    return model, zero_mean_unit_var

"""
def get_augmentor(target_size=128):
    crop_size = int(target_size * 1.415)
    return A.Compose(
        [
            A.RandomSizedCrop(
                [int(crop_size * 0.8), int(crop_size * 1.2)], crop_size, crop_size
            ),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=1),
            A.CenterCrop(target_size, target_size),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                ],
                p=0.1,
            ),
        ]
    )
"""

def get_augmentor(backbone="mobilenet_v2"):
    preprocess_input = get_preprocessing_fn(backbone, pretrained='imagenet')
    return preprocess_input

class UnetInteractiveModel:
    def __init__(
        self,
        model_dir=None,
        type="unet",
        resume=True,
        pretrained_model=None,
        save_freq=None,
        use_gpu=True,
        learning_rate=0.2,
        batch_size=2,
        backbone="mobilenet_v2",
        channels=[1, 2],
    ):
        assert type == "unet"
        assert model_dir is not None
        self.model_dir = model_dir
        """
        device, gpu = tf.keras.models.assign_device(True, use_gpu)
        # device, gpu = models.assign_device(True, use_gpu)
        if save_freq is None:
            if gpu:
                self.save_freq = 2000
            else:
                self.save_freq = 300
        else:
            self.save_freq = save_freq
        """
        self.learning_rate = learning_rate
        self.channels = channels
        self.batch_size = batch_size
        self.model_path = pretrained_model
        self.class_weight = {0: 0.0, 1: 100.0, 2: 1.0}
        tf.keras.backend.clear_session()

        # assert (
        #     _img.shape[2] == self.model.input_shape[3]
        # ), f"shape mismatch: { _img.shape[2] } != {self.model.input_shape[3]}"
        # assert (
        #     _mask.shape[2] == self.model.output_shape[3]
        # ), f"shape mismatch: { _mask.shape[2] } != {self.model.output_shape[3]}"

        # min_size = min(_img.shape[0], _img.shape[1])
        # if min_size >= 512:
        #     training_size = 256
        # elif min_size >= 256:
        #     training_size = 128
        # else:
        #     raise Exception(
        #         f"invalid input image size: {min_size}, it should not smaller than 256x256"
        #     )
        training_size = 256

        # load latest model if exists
        os.makedirs(self.model_dir, exist_ok=True)
        if resume:
            if resume == True:
                label = "latest"
            else:
                label = resume
            checkpoint = os.path.join(self.model_dir, f"model_{label}.h5")
            # only complain error if resume was set to a specific label
            if resume != True and not os.path.exists(checkpoint):
                raise Exception(
                    f"checkpoint file not found: {checkpoint}, if you want to start from scratch, please set resume to False."
                )
            elif not os.path.exists(checkpoint):
                checkpoint = None
            report_path = os.path.join(self.model_dir, f"reports_{label}.json")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    self.reports = json.load(f)
        else:
            checkpoint = None

        self.augmentor = get_augmentor()
        self.model, self.preprocess_input = load_unet_model(self.model_path, backbone=backbone)

    def augment(self, X, y):
        augmented = self.augmentor(image=X, mask=y)
        return augmented["image"], augmented["mask"]

    def train_on_batch(self, X, y):
        """train the model for one iteration
        Parameters
        --------------
        X: array [batch_size, width, height, channel]
            the input image with 2 channels

        y: array [batch_size, width, height, channel]
            the mask (a.k.a label) image with one unique pixel value for one object
            if the shape is [1, width, height], then y is the label image
            otherwise, it should have channel=4 where the 1st channel is the label image
            and the other 3 channels are the precomputed flow image

        Returns
        ------------------
        loss value
        """
        X = self.preprocess_input(np.asarray(X, dtype="float32"))
        y = np.asarray(y, dtype="float32")
        loss_metrics = self.model.train_on_batch(X, y)
        return loss_metrics

    def predict(self, X):
        """predict the model for one input image
        Parameters
        --------------
        X: array [batch_size, width, height, channel]
            the input image with 2 channels

        Returns
        ------------------
        array [batch_size, width, height, channel]
            the predicted label image
        """
        X = self.preprocess_input(X)
        mask = self.model.predict(X)
        mask[0, :, :, 0] = 0
        labels = label_cell2(mask[0, :, :, :])
        return labels

    def save(self, filename):
        """save the model
        Parameters
        --------------
        filename: string
            the file name to the model

        Returns
        ------------------
        saved path
        """
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(os.path.join(self.model_dir, f"model_{label}.h5"))
        with open(os.path.join(self.model_dir, f"reports_{label}.json"), "w") as f:
            json.dump(self.reports, f)

    def export(self, format, file_path):
        """export the model into different format
        Parameters
        --------------
        format: string
            the model format to be exported
        file_path: string
            the file path to the exported model

        Returns
        ------------------
        None
        """
        raise NotImplementedError
