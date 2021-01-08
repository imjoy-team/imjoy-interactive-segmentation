import os
import tensorflow as tf
import segmentation_models as sm
from segmentation_models.base import Loss
from segmentation_models.base import functional as F

class UnetInteractiveModel:
    def __init__(
        self,
        model_path=None,
        use_gpu=True,
        learning_rate=0.2,
        batch_size=2,
        backbone="mobilenetv2",
    ):
        #device, gpu = tf.keras.models.assign_device(True, use_gpu)
        self.learning_rate = learning_rate
        self.channels = [1, 2]
        self.batch_size = batch_size
        self.model_path = model_path
        if self.model_path:
            #logger.info("model loaded from %s", model_path)
            self.model = tf.keras.models.load_model(model_path, compile=False)
        else:
            # define model
            self.model = sm.Unet(
                backbone,
                encoder_weights="imagenet",
                classes=3,
                activation="sigmoid",
                layers=tf.keras.layers,
                models=tf.keras.models,
                backend=tf.keras.backend,
                utils=tf.keras.utils,
            )
            #logger.info("model built from scratch, backbone: %s", backbone)

    def get_random_training_sample(X, Y):    
        return random.choice(zip(X,Y))

    def train_once(self, X, Y):
        batchX = []
        batchY = []
        for i in range(self.batch_size):
            x, y = self.get_random_training_sample(X,Y)
            #x, y, info = self.get_random_training_sample()
            augmented = self.augmentor(image=x, mask=y)
            batchX += [self.preprocess_input(augmented["image"])]
            batchY += [augmented["mask"]]
        loss_metrics = self.model.train_on_batch(
            np.asarray(batchX, dtype="float32"), np.asarray(batchY, dtype="float32")
        )
        return loss_metrics

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

    def predict(self, X):
        mask = self.model.predict(self.preprocess_input(np.expand_dims(image, axis=0)))
        mask[0, :, :, 0] = 0
        labels = np.flipud(label_cell2(mask[0, :, :, :]))
        # simplify_tol is removed, otherwise, some coordinates will be empty
        geojson = mask_to_geojson(labels, label=self.object_name, simplify_tol=None)
        return geojson, mask[0, :, :, :]

    def save_model(self, label="latest"):
        os.makedirs(self.model_dir, exist_ok=True)
        self.model.save(os.path.join(os.path.basename(self.model_path), f"model_{label}.h5"))
        with open(os.path.join(os.path.basename(self.model_path), f"reports_{label}.json"), "w") as f:
            json.dump(self.reports, f)
