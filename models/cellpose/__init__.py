import os
from cellpose import utils, models, io
from cellpose.utils2 import (
    load_train_test_data,
    get_image_folders,
    read_multi_channel_image,
)


class CellPoseInteractiveModel:
    def __init__(
        self,
        model_path=None,
        use_gpu=True,
        diam_mean=30.0,
        residual_on=1,
        learning_rate=0.2,
        batch_size=2,
        resample=False,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        interp=True,
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
        self.model = models.CellposeModel(
            gpu=gpu,
            device=device,
            torch=True,
            pretrained_model=model_path,
            diam_mean=diam_mean,
            residual_on=residual_on,
            style_on=0,
            concatenation=0,
        )

    def train_once(self, X, y):
        """
        X is a numpy array with shape, for example, [2, 512, 512] for a two channel image, the first channel is the cell, and the second channel for the nuclei
        y is the corresponding segmentation labels with the shape of [512, 512] (1 channel for the labels) or [4, 512, 512] (labels + 3 channels for flow image)
        """
        save_path = (
            os.path.realpath(os.path.dirname(self.model_path))
            if self.model_path
            else None
        )
        cpmodel_path = self.model.train(
            [X],
            [y],
            train_files=None,
            test_data=None,
            test_labels=None,
            test_files=None,
            learning_rate=self.learning_rate,
            channels=self.channels,
            save_path=save_path,
            rescale=True,
            n_epochs=1,
            batch_size=self.batch_size,
        )

    def predict(self, X, diameter=30):
        """
        X is a numpy array with shape, for example, [2, 512, 512] for a two channel image, the first channel is the cell, and the second channel for the nuclei
        """
        masks, flows, diams = self.model.eval(
            X,
            channels=self.channels,
            diameter=diameter,
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
        self.model.net.save_model(file_path)
