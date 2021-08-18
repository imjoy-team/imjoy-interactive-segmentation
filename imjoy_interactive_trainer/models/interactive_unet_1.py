import os
import urllib.request

from numpy.lib.type_check import asfarray
import torch
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class UnetInteractiveModel:
    def __init__(
        self,
        model_dir=None,
        type="unet",
        backbone="mobilenet_v2",
        resume=True,
        pretrained_model="imagenet",
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
            self.device = torch.device('cpu')
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

        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=pretrained_model,
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
        self._iterations = 0
        # Note: we are using Adam for adaptive learning rate which is different from the SDG used by cellpose
        # this support to make the training more robust to different settings
        self.model.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
                
        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]
        preprocessing_fn = smp.encoders.get_preprocessing_fn(backbone, pretrained='imagenet')
        self.augmentator = A.Compose([
                                A.ShiftScaleRotate(),
                                A.RGBShift(),
                                A.Blur(),
                                A.GaussNoise(),
                                A.Lambda(image=preprocessing_fn),
                                A.Lambda(image=to_tensor, mask=to_tensor)
                            ],p=1)

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
            2) `optimizer` the optimizer for training
            3) `loss` the loss for training
            4) `lr` the learning rate
            5) 
        """
        return {"batch_size": self.batch_size,
        "optimizer": self.model.optimizer,
        "loss": self.loss,
        "lr": self.learning_rate
        }

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
        #transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        t_label_image = label_image.astype("float32")#.transpose(2, 0, 1).astype('float32')
        return t_label_image

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
        #images = [images[i].transpose(2, 0, 1) for i in range(images.shape[0])]
        #labels = [labels[i].transpose(2, 0, 1) for i in range(labels.shape[0])]

        nimg = len(images)
        # check that arrays are correct size
        if nimg != len(labels):
            raise ValueError("train images and labels not same length")
        if labels[0].ndim < 2 or images[0].ndim < 2:
            raise ValueError(
                "training images or labels are not at least two-dimensional"
            )
        imgi = []
        lbl = []
        for image, mask in zip(images, labels):
            print(image.shape, image.max(), mask.shape, mask.max())
            sample = self.augmentator(image=image, mask=mask)
            img, lb = sample['image'], sample['mask']
            # img = self.preprocess_input_fn(img)
            imgi += [img]
            lbl += [lb]
        imgi = torch.from_numpy(np.asarray(imgi)).to(self.device)#.unsqueeze(0)
        lbl = torch.from_numpy(np.asarray(lbl)).to(self.device)#.unsqueeze(0)
        print(f"Agument output dtype {type(imgi)}")
        return imgi, lbl

    def train(
        self,
        images,
        labels,
        iterations=1,
        rescale=True,
    ):
        imgi, lbl = self.augment(images, labels)
        prediction = self.model.forward(imgi)
        train_loss = self.loss(prediction, lbl)
        train_loss.backward()
        self.model.optimizer.step()
        self._iterations += len(images)
        if self._iterations % self.save_freq == 0:
            self.save(os.path.join(self.model_dir, "snapshot"))
        return train_loss.detach().numpy()

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
        #x_tensors = torch.from_numpy(X).to(self.device).unsqueeze(0)
        #y_tensors = torch.from_numpy(y).to(self.device).unsqueeze(0)
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
        assert X.ndim == 4 # len(X.shape) == 4 #.ndim == 4
        print(f"Input shape {X.shape}, type {torch.from_numpy(X).dtype}, max {X.max()}")
        x_tensor = torch.from_numpy(X.astype('float32')).permute(0,3,2,1).to(self.device)
        pr_mask = self.model.predict(x_tensor)
        pr_mask = (pr_mask.permute(0,3,2,1).cpu().numpy().round())
        print(f"predicted output shape {pr_mask.shape}, max {pr_mask.max()}")
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
        self.model.save_model(file_path)

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
        self.model.load_state_dict(torch.load(file_path), strict=True)

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
