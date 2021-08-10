import os
import tensorflow as tf
import albumentations as A

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import Loss
from segmentation_models_pytorch.base import functional as F

SMOOTH = 1e-5


class BCEJaccardLoss(Loss):
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


def load_unet_model(model_path=None, backbone="mobilenetv2"):
    # disable warnings temporary
    warnings.filterwarnings("ignore")

    # preprocess_input = smp.get_preprocessing(backbone)

    if model_path:
        logger.info("model loaded from %s", model_path)
        model = tf.keras.models.load_model(model_path, compile=False)
    else:
        # define model
        model = smp.Unet(
            backbone,
            encoder_weights="imagenet",
            classes=3,
            activation="sigmoid",
            layers=tf.keras.layers,
            models=tf.keras.models,
            backend=tf.keras.backend,
            utils=tf.keras.utils,
        )
        logger.info("model built from scratch, backbone: %s", backbone)

    model.compile("Adam", loss=BCEJaccardLoss())

    warnings.resetwarnings()
    return model, zero_mean_unit_var


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


class UnetInteractiveModel:
    def __init__(
        self,
        model_path=None,
        type="unet",
        resume=True,
        use_gpu=True,
        learning_rate=0.2,
        batch_size=2,
        backbone="mobilenetv2",
    ):
        # device, gpu = tf.keras.models.assign_device(True, use_gpu)
        self.learning_rate = learning_rate
        self.channels = [1, 2]
        self.batch_size = batch_size
        self.model_path = model_path
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

        self.augmentor = get_augmentor(target_size=training_size)
        self.model, self.preprocess_input = load_unet_model(model_path)

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
