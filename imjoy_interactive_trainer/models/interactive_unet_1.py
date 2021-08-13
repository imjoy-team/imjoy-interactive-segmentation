import os
import urllib.request
import torch
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
 from torchvision import transforms as T

class UnetInteractiveModel:
    def __init__(
        self,
        model_dir=None,
        type="unet",
        backbone="mobilenet_v2",
        resume=True,
        pretrained_model=None,
        save_freq=None,
        use_gpu=True,
        learning_rate=0.001,
        batch_size=2,
        channels=(1, 2),
        resample=True,
        **kwargs,
    ):
        assert type == "unet"
        assert model_dir is not None
        if use_gpu:
            if not torch.cuda.is_available():
                print("No GPU found")
                self.device = torch.device('cpu')
                gpu = False
            else:
                self.device = torch.cuda.get_device_name(0)
                gpu = True
        else:
            gpu = False
        self.learning_rate = learning_rate
        self.channels = channels
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.resample = resample
        if save_freq is None:
            if gpu:
                self.save_freq = 2000
            else:
                self.save_freq = 300
        else:
            self.save_freq = save_freq
        if pretrained_model==None:

        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            classes=3,
            activation="sigmoid",
            **kwargs,
        )
        os.makedirs(self.model_dir, exist_ok=True)
        if resume:
            resume_weights_path = os.path.join(self.model_dir, "snapshot")
            if os.path.exists(resume_weights_path):
                print("Resuming model from " + resume_weights_path)
                self.load(resume_weights_path)
                # disable pretrained model
                pretrained_model = False
            else:
                print("Skipping resume, snapshot does not exist")
        # load pretrained model weights if not specified
        if pretrained_model is None:
            _model_dir = Path.home().joinpath(".cellpose", "models")
            os.makedirs(_model_dir, exist_ok=True)
            weights_path = _model_dir / (model_type + "torch_0")
            if not weights_path.exists():
                urllib.request.urlretrieve(
                    f"https://www.cellpose.org/models/{model_type}torch_0",
                    str(weights_path),
                )
            if not (_model_dir / f"size_{model_type}torch_0.npy").exists():
                urllib.request.urlretrieve(
                    f"https://www.cellpose.org/models/size_{model_type}torch_0.npy",
                    str(_model_dir / f"size_{model_type}torch_0.npy"),
                )

            print("loading pretrained cellpose model from " + str(weights_path))
            if gpu:
                self.model.net.load_state_dict(
                    torch.load(str(weights_path)), strict=False
                )
            else:
                self.model.net.load_state_dict(
                    torch.load(str(weights_path), map_location=self.device),
                    strict=False,
                )
        self._iterations = 0
        # Note: we are using Adam for adaptive learning rate which is different from the SDG used by cellpose
        # this support to make the training more robust to different settings
        self.model.optimizer = torch.optim.Adam(
            self.model.net.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        self.model._set_criterion()
        self.augmentator = smp.encoders.get_preprocessing_fn(backbone, pretrained='imagenet')


    def get_config(self):
        """augment the images and labels
        Parameters
        --------------
        None

        Returns
        ------------------
        config: dict
            a dictionary contains the following keys:
            1) `batch_size` the batch size for training
        """
        return {"batch_size": self.batch_size}

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
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        label_image = transform
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

        sample = self.augmentator(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']
        return imgi, lbl

    def train(
        self,
        images,
        labels,
        iterations=1,
        rescale=True,
    ):
        imgi, lbl = self.augment(images, labels)
        train_loss = self.model._train_step(imgi, lbl)
        self._iterations += len(images)
        if self._iterations % self.save_freq == 0:
            self.save(os.path.join(self.model_dir, "snapshot"))
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
        x_tensor = torch.from_numpy(X).to(self.device).unsqueeze(0)
        pr_mask = self.model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        return pr_mask

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
