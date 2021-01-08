import os
import urllib.request
import torch
from pathlib import Path
from cellpose import utils, models, io


class CellPoseInteractiveModel:
    def __init__(
        self,
        model_path=None,
        use_gpu=True,
        diam_mean=30.0,
        residual_on=1,
        learning_rate=0.2,
        batch_size=2,
        resample=True,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        interp=True,
        default_diameter=30,
        style_on=0,
    ):
        device, gpu = models.assign_device(True, use_gpu)
        self.learning_rate = learning_rate
        self.channels = [1, 2]
        self.batch_size = batch_size
        self.model_path = model_path
        self.resample = resample
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.interp = interp
        self.default_diameter = default_diameter
        self.save_path = (
            os.path.realpath(os.path.dirname(self.model_path))
            if self.model_path
            else None
        )
        self.model = models.CellposeModel(
            gpu=gpu,
            device=device,
            torch=True,
            pretrained_model=model_path,
            diam_mean=diam_mean,
            residual_on=residual_on,
            style_on=style_on,
            concatenation=0,
        )
        # load pretrained model weights if not specified
        if model_path is None:
            model_dir = Path.home().joinpath(".cellpose", "models")
            os.makedirs(model_dir, exist_ok=True)
            weights_path = model_dir / "cytotorch_0"
            if not weights_path.exists():
                urllib.request.urlretrieve(
                    "https://www.cellpose.org/models/cytotorch_0", str(weights_path)
                )
            if not (model_dir / "size_cytotorch_0.npy").exists():
                urllib.request.urlretrieve(
                    "https://www.cellpose.org/models/size_cytotorch_0.npy",
                    str(model_dir / "size_cytotorch_0.npy"),
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

    def train(self, train_data, test_data=None, iterations=None):
        images, labels = train_data
        test_images, test_labels = test_data if test_data is not None else (None, None)
        iterations = iterations if iterations is not None else len(images)
        n_epochs = iterations // len(images)
        n_epochs = max(1, n_epochs)
        cpmodel_path = self.model.train(
            images,
            labels,
            train_files=None,
            test_data=test_images,
            test_labels=test_labels,
            test_files=None,
            learning_rate=self.learning_rate,
            channels=self.channels,
            save_path=self.save_path,
            rescale=True,
            n_epochs=n_epochs,
            batch_size=self.batch_size,
        )

    def train_on_batch(self, X, y):
        """train the model for one iteration
        Parameters
        --------------
        X: array [batch_size, channel, width, height]
            the input image with 2 channels

        y: array [batch_size, channel, width, height]
            the mask (a.k.a label) image with one unique pixel value for one object
            if the shape is [1, width, height], then y is the label image
            otherwise, it should have channel=4 where the 1st channel is the label image
            and the other 3 channels are the precomputed flow image

        Returns
        ------------------
        None
        """
        assert X.shape[0] == y.shape[0]
        X_ = []
        y_ = []
        for i in range(X.shape[0]):
            X_.append(X[i])
            y_.append(y[i])
        self.train([X_, y_])

    def predict(self, X):
        """predict the model for one input image
        Parameters
        --------------
        X: array [channel, width, height]
            the input image with 2 channels

        Returns
        ------------------
        y_predict: the predicted label image
        """
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
        return masks

    def save(self, file_path):
        """save the model
        Parameters
        --------------
        model_path: string
            the file path to the model

        Returns
        ------------------
        None
        """
        self.model.net.save_model(file_path)

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
