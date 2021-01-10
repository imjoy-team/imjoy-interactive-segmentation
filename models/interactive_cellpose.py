import os
import urllib.request
import torch
import numpy as np
from pathlib import Path
from cellpose import utils, models, io, transforms, dynamics


class CellPoseInteractiveModel:
    def __init__(
        self,
        model_dir,
        pretrained_model=None,
        use_gpu=True,
        diam_mean=30.0,
        residual_on=1,
        learning_rate=0.02,
        batch_size=2,
        channels=(1, 2),
        resample=True,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        interp=True,
        default_diameter=30,
        style_on=0,
    ):
        device, gpu = models.assign_device(True, use_gpu)
        self.learning_rate = learning_rate
        self.channels = channels
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.resample = resample
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.interp = interp
        self.default_diameter = default_diameter
        self.model = models.CellposeModel(
            gpu=gpu,
            device=device,
            torch=True,
            pretrained_model=pretrained_model,
            diam_mean=diam_mean,
            residual_on=residual_on,
            style_on=style_on,
            concatenation=0,
        )
        os.makedirs(self.model_dir, exist_ok=True)
        # load pretrained model weights if not specified
        if pretrained_model is None:
            cp_model_dir = Path.home().joinpath(".cellpose", "models")
            os.makedirs(cp_model_dir, exist_ok=True)
            weights_path = cp_model_dir / "cytotorch_0"
            if not weights_path.exists():
                urllib.request.urlretrieve(
                    "https://www.cellpose.org/models/cytotorch_0", str(weights_path)
                )
            if not (cp_model_dir / "size_cytotorch_0.npy").exists():
                urllib.request.urlretrieve(
                    "https://www.cellpose.org/models/size_cytotorch_0.npy",
                    str(cp_model_dir / "size_cytotorch_0.npy"),
                )

            print("loading pretrained cellpose model from " + str(weights_path))
            if gpu:
                self.model.net.load_state_dict(
                    torch.load(str(weights_path)), strict=False
                )
            else:
                self.model.net.load_state_dict(
                    torch.load(str(weights_path), map_location=torch.device("cpu")),
                    strict=False,
                )
        LR = np.linspace(0, self.learning_rate, 10)
        max_n_epochs = 200
        if max_n_epochs > 250:
            LR = np.append(LR, self.learning_rate * np.ones(max_n_epochs - 100))
            for i in range(10):
                LR = np.append(LR, LR[-1] / 2 * np.ones(10))
        else:
            LR = np.append(LR, self.learning_rate * np.ones(max(0, max_n_epochs - 10)))
        self.LR = LR
        self._iterations = 0
        momentum = 0.9
        weight_decay = 0.00001
        self.model._set_optimizer(self.LR[0], momentum, weight_decay)
        self.model._set_criterion()

    def _get_LR(self):
        epoch = int(self._iterations / 1000)  # assuming we have 1000 samples
        if epoch > len(self.LR):
            return self.LR[-1]
        return self.LR[epoch]

    def transform_labels(self, label_image):
        """transform the labels which will be used as training target
        Parameters
        --------------
        label_image: array [width, height, channel]
            a label image

        Returns
        ------------------
        array [width, height, channel]
            the transformed label image
        """
        assert label_image.ndim == 3 and label_image.shape[2] == 1
        label_image = label_image[:, :, 0]
        veci = dynamics.masks_to_flows(label_image)[0]
        # concatenate flows with cell probability
        flows = np.concatenate(
            (np.stack([label_image, label_image > 0.5], axis=0), veci), axis=0
        ).astype(np.float32)
        return flows.transpose(1, 2, 0)

    def augment(self, images, labels):
        """augment the images and labels
        Parameters
        --------------
        images: array [batch_size, width, height, channel]
            a batch of input images

        labels: array [batch_size, width, height, channel]
            a batch of labels

        Returns
        ------------------
        (images, labels) both are: array [batch_size, width, height, channel]
            augmented images and labels
        """
        images = [images[i].transpose(2, 0, 1) for i in range(images.shape[0])]
        labels = [labels[i].transpose(2, 0, 1) for i in range(labels.shape[0])]

        nimg = len(images)
        # check that arrays are correct size
        if nimg != len(labels):
            raise ValueError("train images and labels not same length")
        if labels[0].ndim < 2 or images[0].ndim < 2:
            raise ValueError(
                "training images or labels are not at least two-dimensional"
            )

        # make data correct shape and normalize it so that 0 and 1 are 1st and 99th percentile of data
        images, _, _ = transforms.reshape_and_normalize_data(
            images, test_data=None, channels=self.channels, normalize=True
        )

        # compute average cell diameter
        diam_train = np.array(
            [utils.diameters(labels[k][0])[0] for k in range(len(labels))]
        )
        diam_train[diam_train < 5] = 5.0
        scale_range = 0.5
        rsc = diam_train / self.model.diam_mean
        imgi, lbl, scale = transforms.random_rotate_and_resize(
            images,
            Y=[label[1:] for label in labels],
            rescale=rsc,
            scale_range=scale_range,
            unet=False,
        )
        return imgi, lbl

    def train(
        self,
        images,
        labels,
        iterations=1,
        rescale=True,
    ):
        imgi, lbl = self.augment(images, labels)
        self.model._set_learning_rate(self._get_LR())
        train_loss = self.model._train_step(imgi, lbl)
        self._iterations += len(images)
        return train_loss

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
        assert X.shape[0] == y.shape[0] and X.ndim == 4

        return self.train(X, y)

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
        assert X.ndim == 4
        X = [X[i].transpose(2, 0, 1) for i in range(X.shape[0])]
        masks, flows, diams = self.model.eval(
            X,
            channels=self.channels,
            diameter=self.default_diameter,
            do_3D=False,
            net_avg=False,
            augment=False,
            resample=self.resample,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            batch_size=self.batch_size,
            interp=self.interp,
        )
        # io.masks_flows_to_seg(image, masks, flows, diams, image_name, channels)
        # io.save_masks(path, masks, flows, image_name, png=True, tif=False)
        return np.stack([np.expand_dims(mask, axis=2) for mask in masks], axis=0)

    def save(self, file_path):
        """save the model
        Parameters
        --------------
        file_path: string
            the model file path

        Returns
        ------------------
        None
        """
        self.model.net.save_model(file_path)

    def load(self, file_path):
        """load the model
        Parameters
        --------------
        file_path: string
            the model file path

        Returns
        ------------------
        None
        """
        self.model.net.load_state_dict(torch.load(file_path), strict=True)

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
